
Below is a focused **Metal & MLX implementation research pack** to unblock the kernel work for **PagedAttention (decode and prefill)** and make it fit cleanly into the MLX stack you’ve already wired up (Python fallback, tests, KVBlockManager). I’ve prioritized (1) **Metal kernel design for attention**, (2) **streaming softmax across tiles**, and (3) **GQA/MQA handling** — with concrete guidance, code skeletons, and references.

---

## 0) What you already have (so we align precisely)

* **Python fallback op** `_paged_attention_reference` registered onto `mx.fast` when the native op is missing (parity-tested vs dense reference).
* **KVBlockManager** with copy-on-write (COW) tails for decode-time append.
* **TDD** in `python/tests/test_paged_kv.py` (allocator tests; dense vs paged attention parity), passing inside `mlx/.venv`.
* **To implement next (per your PRD)**:

  1. **Metal decode kernel** with streaming softmax, GQA mapping, function-constant block size.
  2. **C++ launcher/bindings** to register the op and prefer Metal when present.
  3. **Prefill writer** to fill K/V pages in chunked fashion.
  4. **Microbenches & perf validation**.

Everything below is scoped to those deliverables.

---

## 1) Metal kernels for attention on Apple GPUs — practical guidance

### 1.1 Threadgroup sizing & SIMD-group usage

* **Choose threadgroup sizes in multiples of the device’s thread execution width** (SIMD width) and keep total threads in a group high enough to hide latency. On Apple GPUs, optimal sizes are typically 128–512 threads per group; 256 is a safe starting point, then tune. Apple’s guidance is to **pick a size that is a multiple of the thread execution width** and balance register and threadgroup-memory pressure. ([Apple Developer][1])
* **Use SIMD‑group reductions** for per‑tile max/sum in softmax (e.g., `simdgroup_reduce_max`, `simdgroup_reduce_add`) instead of manual reductions — these map to efficient hardware collectives. These functions are documented in the Metal language & feature set materials; they’re available across modern Apple GPU feature sets (see SIMD-scope built-ins). ([Apple Developer][1])
* **Favor vectorized loads/stores** (e.g., `half4` / `float4`) where alignment permits. Keep **head_dim** strides aligned to 8–16 bytes to enable coalesced, natural alignment reads/writes; this matches best practice Apple calls out (alignment & vectorization strongly affect throughput). ([Apple Developer][1])

### 1.2 Threadgroup (LDS) memory & double-buffering

* **Stage K/V tiles in threadgroup memory** to amortize global memory bandwidth across the MMA (dot products with Q) and softmax accumulation per tile. Keep a **double-buffer** (ping/pong) to overlap global loads with compute on the previous tile.
* Use `threadgroup_barrier(mem_flags::mem_threadgroup)` to sequence producer/consumer phases cleanly. Apple’s compute best-practices and feature tables emphasize threadgroup memory reuse and barriers to maximize locality and hide latency. ([Apple Developer][1])

### 1.3 Avoiding pipeline recompiles (function constants & pipelines)

* **Parameterize tile sizes via function constants** (`MTLFunctionConstantValues`) and **prebuild a small set of compute pipelines** for the likely shapes (e.g., `BLOCK_SIZE ∈ {16, 32, 64}`; optionally `HEAD_DIM_TILE` choices). Reuse pipeline states rather than specializing at high frequency. Apple’s docs: define function constants and create pipelines from `MTLFunctionConstantValues` once; swap pipeline at dispatch time. ([Apple Developer][2])
* Use **argument buffers** to pass per-dispatch resource sets (block tables, page base pointers, strides) without re-encoding many bindings; this reduces CPU overhead. Apple recommends argument buffers to **reduce Metal command encoding cost** when resources vary per-dispatch. ([Apple Developer][3])

### 1.4 Heaps & allocations (KV residency)

* If you manage the KV pages more manually on the native side, **MTLHeap** can offer tighter control (placement, aliasing, reuse) — but MLX’s allocator likely already maps to device-coherent buffers. Apple’s heap docs are still useful if you add native KV buffers later. ([Stack Overflow][4])

---

## 2) Streaming softmax across tiles (decode) — algorithm details you can drop into MSL

**Goal:** Compute attention over a long, paged K/V sequence **without materializing the full score vector**, using a numerically stable **online (streaming) softmax** that fuses score computation and accumulation tile-by-tile.

**Reference algorithm** (same recurrence as FlashAttention/online softmax):

* Maintain per‑query running `(m, l, o)`:

  * `m` = running max of scores
  * `l` = running sum of `exp(score - m)`
  * `o` = running sum of `exp(score - m) * v`
* For each tile `T` of keys/values:
  Let `z_t` be the vector of scores for this tile (dot products `q·k_t` scaled).
  `m_new = max(m, max(z_t))`
  `alpha = exp(m - m_new)`
  `l = alpha * l + sum(exp(z_t - m_new))`
  `o = alpha * o + Σ exp(z_t - m_new) * v_t`
* At end: output `o / l`. This is the standard **online normalizer** used by FlashAttention; it’s numerically stable when you accumulate in FP32. ([Wikipedia][5])

**Why this is precisely what PagedAttention needs:** PagedAttention tiles sequence memory by **blocks**; you stream through the **block_table** for a request, bring one block at a time into shared/threadgroup memory, compute, and update the running `(m, l, o)`; no giant scores buffer is needed. The original PagedAttention paper focuses on **memory paging and cache design**; the streaming softmax is the compute analog that preserves exactness. ([Apple Developer][6])

**Tuning levers (Metal-specific):**

* Tile size along the **sequence** dimension (`BLOCK_SIZE`) is your main specialization constant. Start with 64 and 32; on smaller models and large concurrency, 32 may schedule better. Prebuild pipelines for both. ([Apple Developer][2])
* Tile size along **head_dim** (`HD_TILE`) determines vectorized loads; favor multiples of 8/16 bytes. Pick based on the model head dim (e.g., 128, 256) and register pressure.
* Always **accumulate in FP32** (`float` in MSL), even if Q/K/V are FP16/bfloat16 — this is the common practice for stability. (The online softmax reference explicitly derives the stable recurrence.) ([Wikipedia][5])

**Kernel‑level pseudocode (Metal)** — **decode path** (one Q per sequence step):

```metal
#include <metal_stdlib>
using namespace metal;

struct PagedParams {
  uint  head_dim;               // e.g., 128
  uint  block_size;             // e.g., 32/64 (function constant)
  uint  num_q_heads;            // per token
  uint  num_kv_heads;
  uint  kv_head_stride_elems;   // elements between heads in a page
  uint  page_stride_elems;      // elements between pages
  uint  max_blocks_per_seq;
  float scale;                  // 1/sqrt(head_dim) or model-provided
};

constant ushort BLOCK_SIZE [[ function_constant(0) ]];  // specialize {32, 64}
constant ushort HD_TILE    [[ function_constant(1) ]];  // specialize {8, 16}

kernel void paged_attention_decode(
  device const half  *q,                  // [B, num_q_heads, head_dim]
  device const half  *kv_pages,           // flat pool of K then V pages (or separate)
  device const uint  *block_table,        // [B, max_blocks_per_seq] -> page ids
  device const uint  *context_lens,       // [B]
  device       half  *out,                // [B, num_q_heads, head_dim]
  constant PagedParams &p,
  ushort3 tgpig [[threadgroup_position_in_grid]],
  ushort  tid   [[thread_index_in_threadgroup]]
) {
  // Map this threadgroup to (batch_idx, q_head_idx)
  const uint b = tgpig.y;
  const uint hq = tgpig.x;               // query head idx within [num_q_heads]
  const uint hk = /* map to kv head (GQA) */ (hq * p.num_kv_heads) / p.num_q_heads;

  // Each TG processes one (b, hq). Threads cooperate along head_dim and seq tiles.
  threadgroup half  K_tile[BLOCK_SIZE][HD_TILE];
  threadgroup half  V_tile[BLOCK_SIZE][HD_TILE];

  // Load q vector for (b, hq) in vectorized chunks; upcast to float for math.
  float q_reg[/*HD_TILE-rolled*/];  // keep per-thread chunk of head_dim in registers
  // ... (coalesced loads: reinterpret_cast<vector_halfN>)

  // Online softmax state (float for stability)
  float m = -INFINITY;
  float l = 0.0f;
  float o_accum[/*HD_TILE-rolled*/] = {0}; // partial V-weighted sum in regs

  const uint seq_len = context_lens[b];
  const uint num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (uint blk = 0; blk < num_blocks; ++blk) {
    const uint page_id = block_table[b * p.max_blocks_per_seq + blk];

    // Cooperative load of K and V block [BLOCK_SIZE, head_dim] for kv head = hk
    // into threadgroup memory, vectorized across HD_TILE dimension.
    // address = base(kv_pages) + page_id * p.page_stride_elems
    //         + hk * p.kv_head_stride_elems
    //         + row * head_dim + col
    // Double-buffer this in practice.

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute scores for the BLOCK_SIZE rows contributed by this tile:
    // Each thread computes dot(q, k_row) for its subset of head_dim, then SIMD-reduce
    // across threads to get the full dot.
    float z_local = /* partial dot for this thread */;
    // reduce within simdgroup, then across simdgroups if needed:
    float z = simdgroup_reduce_add(z_local); // then TG-wide reduction if >1 simdgroup

    // Apply scale
    z *= p.scale;

    // Compute tile max across rows: maintain running m/l/o via online recurrence
    float m_tile = simdgroup_reduce_max(z); // reduce over BLOCK_SIZE rows
    float m_new  = max(m, m_tile);
    float alpha  = exp(m - m_new);

    // Compute l update: sum(exp(z - m_new)) over rows
    float e = exp(z - m_new);              // per-row
    float l_tile = simdgroup_reduce_add(e);
    l = alpha * l + l_tile;

    // Compute o update: weighted sum over V_tile rows
    // o = alpha * o + Σ exp(z_i - m_new) * v_i
    // Accumulate in float; write back to half at the end.
    // Each thread multiplies its V chunk by e and accumulates to o_accum
    // then reduce across threads for the full head_dim vector if needed.
    // ...

    m = m_new;
  }

  // Normalize: out = o / l
  // Write back to out[b, hq, :]
}
```

This is deliberately **register- and TG‑centric**; in your actual kernel:

* Execute **two nested loops**: (1) over `blk` pages, (2) over head_dim tiles to keep vectorized loads.
* Use **SIMD‑group collectives** to reduce **dot** and **softmax** statistics. ([Apple Developer][1])
* Accumulate **`o_accum` in FP32** and cast at the end. ([Wikipedia][5])

**Further background specific to PagedAttention:**

* vLLM’s public docs outline the **block table / KV page** abstraction and why it enables continuous batching without defragmenting memory; useful for aligning your block-table interpretation and per-request iteration. ([VLLM Docs][7])
* The original paper is good for **page sizing** trade-offs and the serving pipeline design (it complements the compute kernel design you’re implementing here). ([Apple Developer][6])

---

## 3) GQA/MQA handling (head mapping) — MLX-friendly patterns

* **Mapping**: For GQA with `num_q_heads` and `num_kv_heads`, the **kv-head index for a given query head** is typically:

  ```cpp
  // C++ / Metal-friendly
  uint kv_head = (hq * num_kv_heads) / num_q_heads;
  ```

  This is the same mapping used widely (e.g., LLaMA-style GQA). Keep it **precomputed** on the CPU if you want to avoid the divide (cheap either way). (General practice described in LLaMA/vLLM implementations.) ([VLLM Docs][7])
* **Memory layout**: Lay out K and V **by kv-head**, contiguous along **head_dim**, and **page-major** along the sequence. That is:

  ```
  [page][kv_head][row_in_block][head_dim]
  ```

  with `row_in_block` ≡ token offset in that page (0..BLOCK_SIZE-1). This maximizes coalescing when you broadcast `q` (per `hq`) against many rows and then pull the corresponding **V rows** for the same kv-head.
* **MQA** (num_kv_heads = 1): This naturally falls out of the same codepath; the mapping collapses to `kv_head = 0`.

---

## 4) Prefill path (chunked writer) — what to implement and how to keep it fast

**Goal:** Populate paged K/V buffers during prefill **without creating a giant contiguous K/V**. Two options:

1. **Compute kernel** that receives `[B, T_chunk, num_heads, head_dim]` and **writes rows** into paged buffers by following the `block_table`. Works everywhere and lets you fuse simple transforms (e.g., cast to `half`).
2. **Blit encoder** (if you already have the tensors in the right layout) to copy chunks directly into the KV pages; typically less flexible than compute for format transforms.

**Implementation notes:**

* Use the **same layout & strides** as decode. Verify in tests that the decode kernel can immediately consume what prefill writes.
* If you expect both **prefill and decode concurrently**, consider a **separate Metal queue** for the writer or **batch the copy work** to larger chunks to reduce interleaving overhead. (In MLX this is hidden; you mainly want to ensure kernels are “fat” enough so schedulers see good occupancy.)

---

## 5) MLX integration (C++ launcher & Python binding)

### 5.1 Files to add

* `mlx/backend/metal/kernels/paged_attention.metal`   ← kernel(s)
* `mlx/backend/metal/paged_attention.cpp`             ← pipeline init + dispatch (C++)
* `python/src/fast.cpp` (or equivalent)               ← **expose `mx.fast.paged_attention`** via nanobind
* `CMakeLists.txt` updates to compile `.metal` → `.air` and link into `mlx.metallib` and include the new C++ TU.

**Function‑constant specializations** (build time):

```cpp
// paged_attention.cpp (sketch)
struct PagedKernelKey { uint block; uint hdtile; };
static std::map<PagedKernelKey, id<MTLComputePipelineState>> g_pipelines;

static id<MTLComputePipelineState>
get_pipeline(id<MTLLibrary> lib, uint block, uint hdtile) {
  PagedKernelKey key{block, hdtile};
  auto it = g_pipelines.find(key);
  if (it != g_pipelines.end()) return it->second;

  MTLFunctionConstantValues *fc = [MTLFunctionConstantValues new];
  uint16_t b = block, t = hdtile;
  [fc setConstantValue:&b type:MTLDataTypeUShort atIndex:0];
  [fc setConstantValue:&t type:MTLDataTypeUShort atIndex:1];

  NSError *err = nil;
  id<MTLFunction> f = [lib newFunctionWithName:@"paged_attention_decode"
                            constantValues:fc error:&err];
  id<MTLComputePipelineState> pso =
      [device newComputePipelineStateWithFunction:f error:&err];
  g_pipelines[key] = pso;
  return pso;
}
```

Apple recommends **function constants** with **reused pipeline states** to avoid recompiles at runtime. ([Apple Developer][2])

### 5.2 C++ op signature → Python `mx.fast`

Mirror your Python fallback signature; e.g.:

```cpp
// Inputs: q[B, Hq, D], paged K/V (as device buffers + strides), block_table[B, max_blocks],
//         context_lens[B], scale
// Output: out[B, Hq, D]
void paged_attention_metal(const Array& q,
                           const KVHandle& kv,  // encapsulates device ptrs, strides
                           const Array& block_table,
                           const Array& context_lens,
                           float scale,
                           Array& out);
```

Then bind in `python/src/fast.cpp`:

```cpp
m.def("paged_attention", [](Array q, KVHandle kv, Array block_table,
                            Array context_lens, float scale) {
  Array out = ...;  // allocate via MLX allocator
  paged_attention_metal(q, kv, block_table, context_lens, scale, out);
  return out;
});
```

(Use the same `Array` / device utilities your other fast ops use.)

---

## 6) Microbenchmarking & validation

* **Parity**: Your existing `test_paged_kv.py` dense-vs-paged check is the right baseline; keep tolerances `rtol=1e-4, atol=1e-4` (FP16/bf16 in/out, FP32 accum).
* **Microbenches**: Create **Metal-only** tests that run batched (B=64), `head_dim ∈ {128, 256}`, `BLOCK_SIZE ∈ {32, 64}`, `seq_len ∈ {1k, 2k, 4k}`, and measure **tokens/s** on decode. Tie into your bench harness to compare **steady state** throughput vs your open-loop static baseline.
* **Profiling**: Use **Xcode GPU Capture** to verify:

  * TG occupancy and threadgroup memory usage,
  * Pipeline switches (confirm only a small set of PSOs are used),
  * Global load efficiency (vectorized, coalesced),
  * Time in barriers and reductions.

---

## 7) What to lift from vLLM/FlashAttention (conceptually) and what to keep MLX‑native

* **From vLLM**: page/block-table abstraction and continuous batching semantics (admission/topping-up). Their public docs on the **PagedAttention KV cache** are directly applicable to how you iterate blocks in the kernel and the allocator logic you already wrote. ([VLLM Docs][7])
* **From FlashAttention**: the **online softmax recurrence** and tiling strategy. Even though you’re on Metal (not CUDA/Triton), the algorithmic structure is identical. ([Wikipedia][5])
* **Keep MLX-native**: memory ownership and op registration; use MLX’s allocator and Array handles; compile kernels into `mlx.metallib` (you saw `scaled_dot_product_attention.air` in MLX builds; your new kernel follows the same flow).

---

## 8) Metal‑centric pitfalls to avoid (learned the hard way)

1. **Excessive pipeline specialization**: If you bake too many function-constant combinations, you’ll pay in pipeline count and cache misses. Keep it to 2–3 block sizes and 1–2 head-dim tiles per model family. ([Apple Developer][2])
2. **TG memory overcommit**: Oversized tiles can push TG memory beyond device limits and reduce occupancy. Start conservative (e.g., `BLOCK_SIZE=32`, `HD_TILE=16`) and scale up while checking occupancy.
3. **Atomics for reductions**: Avoid atomics in softmax reductions; use **SIMD-group** and TG-level reductions, then only one scalar update of `(m, l)` per tile. ([Apple Developer][1])
4. **Accumulating in FP16**: Don’t — keep `m/l/o` in `float`. This is critical for long sequences and many tiles. ([Wikipedia][5])
5. **Misaligned K/V layout**: Ensure per‑row stride in head_dim is aligned for `half4/float4` vector ops; pad if necessary.

---

## 9) Concrete shape & launch recipe (decode)

* **Grid**: `tgpig.x = num_q_heads` (or `num_q_heads * BqPerTG` if you batch multiple heads per TG), `tgpig.y = B` (batch), `tgpig.z = 1`.
* **Threads per TG**: Start at 256 (e.g., 8 × 32) and test 128/512. Keep **HD_TILE × lanes per row** roughly equal to head_dim / (threads_per_row).
* **Function constants**: Build pipelines for `BLOCK_SIZE = 32, 64` and `HD_TILE = 8/16` and select at runtime based on head_dim & cache pressure. ([Apple Developer][2])

---

## 10) Prefill writer (compute kernel) — skeleton

```metal
kernel void paged_kv_prefill(
  device const half  *k_in,  // [B, T_chunk, kv_heads, head_dim] or model-native
  device const half  *v_in,
  device       half  *kv_pages,
  device const uint  *block_table, device const uint *write_offsets, // per sequence
  constant PagedParams &p, ushort3 tgpig [[threadgroup_position_in_grid]],
  ushort tid [[thread_index_in_threadgroup]]
) {
  // Map (b, kv_head) to this threadgroup; partition rows over threads
  // Compute page_id and row_in_page for each token row; vectorized store into pages
  // Optionally apply cast/quant transform before store.
}
```

For many models, prefill is bandwidth-bound; **vectorized stores** and **batched writes** (many rows per TG) matter most.

---

## 11) Checkpoints before you connect to `mlx-lm`

1. **Functional parity**: Your Python fallback vs Metal op across random seeds / batch sizes / GQA settings.
2. **Perf sanity**: Microbench shows **steady-state decode** ≥ your static baseline’s **per-token** time over the same token budget (with your new open-loop harness).
3. **Stress**: Long-context runs (e.g., Llama-3.2-3B), mixed kv-head counts (MQA/GQA), **block table fragmentation** (non-uniform page sequences).
4. **Integration**: Bind in `python/src/fast.cpp`; your existing test should seamlessly pick Metal path when the symbol exists, otherwise Python fallback.

---

## 12) Pinned references you can consult while coding

* **PagedAttention (original paper)** — KV paging design; informs page sizing & runtime behavior. ([Apple Developer][6])
* **vLLM docs on PagedAttention** — clear overview of block tables & KV cache design in serving. ([VLLM Docs][7])
* **Online/streaming softmax (FlashAttention background)** — numerically stable recurrence you’ll implement in MSL; same math, different backend. ([Wikipedia][5])
* **Metal function constants (avoid recompiles)** — `MTLFunctionConstantValues` and pipeline reuse. ([Apple Developer][2])
* **Metal argument buffers** — reduce binding overhead when resources vary per dispatch. ([Apple Developer][3])
* **Metal heaps (memory management)** — if you later manage KV storage natively. ([Stack Overflow][4])
* **Metal feature set & performance guidance** — SIMD‑group, threadgroup memory, and best practices. ([Apple Developer][1])
* **MLX docs site** — operator and backend organization; helps match file placement and binding patterns. ([X (formerly Twitter)][8])

---

## 13) Minimal “first light” plan (what to code in what order)

1. **Decode kernel**: Implement streaming softmax with a single `BLOCK_SIZE` (e.g., 32), FP32 accum, fixed `HD_TILE`; bind into C++ and Python.
2. **Correctess**: Pass your existing parity tests across random seeds; add a new test that constructs a **multi-page** block table with non‑trivial lengths.
3. **Specializations**: Add function constants and build **two** pipelines (`BLOCK_SIZE=32/64`); add simple runtime policy to pick based on `seq_len`/`head_dim`.
4. **Prefill writer**: Implement compute-based writer; parity-check that decode reads exactly what prefill wrote (end-to-end prefill→decode tests).
5. **Perf**: Microbench with your open-loop harness, then iterate threadgroup sizes and `HD_TILE`.

This is the shortest path to a robust, MLX‑native PagedAttention that you can then hook directly into your `mlx-lm` top-up scheduler.

---

### If you want a quick starter for the launcher + binding

* Base your new files alongside existing MLX Metal ops (you’ve seen `scaled_dot_product_attention.air` and the metal build step in your logs; mirror that pattern in CMake).
* Pre-create pipelines for your chosen function constants on first use and cache them in a static map.
* Expose the op via `python/src/fast.cpp`, preserving your fallback registration logic (Python op installs only when native is absent).

---

[1]: https://developer.apple.com/documentation/metal/mtlfunctionconstant?utm_source=chatgpt.com "MTLFunctionConstant | Apple Developer Documentation"
[2]: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf?utm_source=chatgpt.com "Metal Shading Language Specification"
[3]: https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Compute-Ctx/Compute-Ctx.html?utm_source=chatgpt.com "Data-Parallel Compute Processing"
[4]: https://stackoverflow.com/questions/78675006/performing-a-reduce-operation-with-metal?utm_source=chatgpt.com "Performing a reduce operation with Metal"
[5]: https://en.wikipedia.org/wiki/Softmax_function?utm_source=chatgpt.com "Softmax function"
[6]: https://developer.apple.com/documentation/metal/creating-threads-and-threadgroups?utm_source=chatgpt.com "Creating threads and threadgroups"
[7]: https://docs.vllm.ai/en/latest/api/vllm/attention/ops/paged_attn.html?utm_source=chatgpt.com "paged_attn - vLLM"
[8]: https://x.com/awnihannun/status/1827087059431125004 "x.com"
