
Below is a **hands‑on implementation guide** that maps a high‑performance, vLLM‑style PagedAttention (for **prefill** and **decode**) onto **MLX** as it exists today, and shows how to integrate it cleanly with your `mlx‑lm` changes (scheduler, open‑loop bench, continuous batching). It is grounded in the current MLX primitives and code layout you shared (Metal attention kernels, RoPE, gather ops, tests/benches), plus the PagedAttention design from the paper you attached. I’ve cited each place where we rely on existing MLX functionality or external references.

---

# MLX PagedAttention (Prefill & Decode) — Implementation Guide

**Targets**

* A fused **`mx.fast.paged_attention`** operator (Metal) with a Python wrapper, covering **prefill** (Q,K,V length ≥ 1) and **decode** (Q length = 1) paths.
* A first‑class **paged KV‑cache** (block allocator + block tables) with **copy‑on‑write** hooks exposed to the Python side.
* An incremental integration path that achieves **correctness using existing MLX ops** first (e.g., `gather_mm`/`gather_qmm`), then unlocks **maximum throughput** with a fused Metal kernel.
* Seamless drop‑in for your `mlx‑lm` runtime & bench harness (open‑loop, tokens‑target, continuous batching).

---

## 0) What already exists (and we will build on)

* **Llama Attention (docs example):** shows RoPE application, KV concatenation, and causal masking strategy using MLX tensors and `nn.RoPE`/`nn.Linear`. This is our reference shape & semantics for correctness.
* **Fast attention & Metal kernels:** MLX already ships a **fast scaled dot‑product attention** and a **Metal SDPA kernel**; the code organization and build system are a template for adding **paged attention**. ([ML Explore][1])
* **Gather GEMM ops & benches:** MLX provides `gather_mm`/`gather_qmm` and dedicated benches—ideal for a **reference (unfused) PagedAttention** using gather + tiled softmax before we land the fused kernel. Directory shows `gather_mm_bench.py`, `gather_qmm_bench.py`.
* **RoPE Metal kernels:** Existing RoPE kernels and instantiations to mirror when we wire RoPE offsets into paged decode.
* **Stream/memory tests & infra:** GPU tests and memory limit toggles indicate how to add new tests for allocator and kernel without deadlocks.

**External design reference:** PagedAttention’s **paged KV memory**, **block tables**, **copy‑on‑write (CoW)**, and **streaming softmax across tiles**. We will implement equivalent mechanics on MLX/Metal.  

---

## 1) API surface & files to add

> The file paths below follow MLX conventions you shipped (docs + kernels + benches) and the “fast” operator style. Adjust to the exact subdirs in your MLX fork as needed.

### 1.1 Python wrapper (public API)

**File:** `python/mlx/core/fast/_paged_attention.py` (imported as `mlx.core.fast.paged_attention`)

```python
# python/mlx/core/fast/_paged_attention.py
import mlx.core as mx

def paged_attention(
    q,                          # [B, H, Lq, D]  (Lq=1 in decode; Lq>=1 in prefill)
    k_cache, v_cache,          # Paged KV buffers, *not* concatenated
    block_tables,              # [B, max_blocks] int32 (indices of blocks for each seq)
    seq_lens,                  # [B] int32 (tokens already in cache per seq)
    block_size: int,
    causal: bool = True,
    scale: float | None = None,   # override or derive as 1/sqrt(D)
    attn_mask: mx.array | None = None,  # optional additive mask (e.g., prefix/cutoffs)
    rope_freqs: mx.array | None = None, # optional fused RoPE frequencies
    rope_offset: mx.array | None = None,# [B] offsets for decode
    dtype_out=None,
):
    """
    Return: out [B, H, Lq, D], plus optional stats for debug
    """
    # This function dispatches to Metal kernel if available,
    # otherwise falls back to a reference implementation (Section 2).
    ...
```

* Mirrors the style of **`mlx.core.fast.scaled_dot_product_attention`** (doc shows API & conventions), but takes **paged KV** and **block tables** rather than contiguous tensors. ([ML Explore][1])
* You’ll register the Metal kernel via **`mx.fast.metal_kernel()`** as in MLX docs for custom kernels. ([ML Explore][2])

### 1.2 C++/Metal kernel bindings

**Files:**

* `mlx/backend/metal/kernels/paged_attention.metal` — **kernel** (decode + prefill modes).
* `mlx/backend/metal/paged_attention.cpp` — **launcher** that binds buffers/strides/params.
* `python/mlx/core/fast/_fast_bindings.py` — register the op as `mlx.core.fast.paged_attention`.

Use the existing **SDPA vector kernel** and its launcher as a template for how to wire grid dimensions, buffer indices, and dtype specializations.

### 1.3 Paged KV memory manager (Python)

**Files:**

* `python/mlx/nn/_kv_cache.py` — **`PagedKVCache`**, **`BlockAllocator`**, **`BlockTable`** (Python side).
* Optional C++ helpers if Python becomes a bottleneck in block management.

Expose:

```python
class BlockAllocator:
    def __init__(self, num_layers, num_heads, head_dim, block_size, capacity_blocks, dtype):
        ...

    def alloc(self, num_blocks) -> mx.array:  # returns block_ids [num_blocks]
        ...

    def free(self, block_ids: mx.array) -> None:
        ...

class PagedKVCache:
    def __init__(self, allocator: BlockAllocator):
        self.alloc = allocator
        # per-layer K,V buffers: layout described in §3
        ...

    def ensure_capacity(self, seq_id, need_tokens): ...
    def append_kv(self, layer, seq_id, k_tok, v_tok): ...
    def block_table(self, seq_id) -> mx.array: ...
    def seq_len(self, seq_id) -> int: ...
```

The **paged KV** and **block table** are directly compatible with `fast.paged_attention`. CoW can be emulated in Python first (copy the table on fork) and optimized later.

---

## 2) Step‑0: Reference (unfused) PagedAttention with existing MLX ops

Before writing the fused kernel, get a **correct** operator using **existing MLX ops** (e.g., `gather_mm`/`gather_qmm`) and **streaming softmax**. This lets you validate shapes, memory layout, masks, and numerics **without** kernel debug time.

**Why this works well on MLX today:** MLX exposes gather‑GEMMs and vector SDPA pieces; you can build a tiled attention that iterates over the KV **blocks** of each sequence and aggregates the result with a **numerically stable streaming softmax** (see §4.2).

Skeleton (decode: `Lq=1`), **inside** the Python fallback of `fast.paged_attention`:

```python
def _paged_attn_reference(q, k_cache, v_cache, block_tables, seq_lens, block_size, scale):
    # q: [B, H, 1, D]; k_cache/v_cache: paged layout (see §3)
    B, H, _, D = q.shape
    scale = scale or (1.0 / (D ** 0.5))

    # Running streaming-softmax state per (B,H):
    m = mx.full((B, H, 1, 1), -mx.inf, dtype=q.dtype)   # running max
    l = mx.zeros((B, H, 1, 1), dtype=q.dtype)           # running sum of exp
    p = mx.zeros((B, H, 1, D), dtype=q.dtype)           # running numerator

    # Iterate per block; gather_qmm computes Q * K^T over an index set.
    # You can realize it via mx.core.gather_qmm / gather_mm (see benches in repo).
    # (Pseudo) iterate blocks: for each seq, block_tables[b] gives block ids [nb].
    max_blocks = block_tables.shape[1]
    for t in range(max_blocks):
        # mask blocks beyond seq_len[b] -> skip or apply -inf
        # gather K_t, V_t for this tile across batch+heads
        K_t = _gather_k_tile(k_cache, block_tables, t, block_size)   # [B,H,block_size,D]
        V_t = _gather_v_tile(v_cache, block_tables, t, block_size)   # [B,H,block_size,D]
        # logits: [B,H,1,block_size]
        logits_t = (q * scale) @ K_t.transpose(0,1,3,2)
        # causal trim if needed (for the last block only): zero out positions > seq_len[b]
        logits_t = _apply_causal_and_len_mask(logits_t, seq_lens, t, block_size)
        # streaming softmax update (see §4.2)
        m_new = mx.maximum(m, logits_t.max(axis=-1, keepdims=True))
        l = l * mx.exp(m - m_new) + mx.sum(mx.exp(logits_t - m_new), axis=-1, keepdims=True)
        p = p * mx.exp(m - m_new) + (mx.exp(logits_t - m_new) @ V_t)
        m = m_new

    out = p / l                     # [B,H,1,D]
    return out
```

* Use `gather_qmm`/`gather_mm` in your `_gather_*_tile` to avoid materializing large intermediate K/V tensors; the benchmarks in the repo show how to call them.
* This reference establishes **correctness** (unit tests, parity vs. `fast.scaled_dot_product_attention` on short contexts) before we write the fused kernel. ([ML Explore][1])

---

## 3) KV layout, block tables, and dtypes (MLX‑friendly)

**Goals:** (1) fixed‑size **blocks**; (2) **contiguous** per‑block storage for coalesced loads; (3) per‑seq **block tables**; (4) stable strides for all layers/heads.

**Recommended backing layout (Metal‑friendly):**

```
K: [L, H_qk, N_blocks_capacity, block_size, D]   # half/bfloat16
V: [L, H_v,  N_blocks_capacity, block_size, D]
```

* Strides per dimension should allow **per‑layer per‑head** contiguous block tiles.
* For **decode**, you write a single token’s K,V into the block at the current cursor; once the block fills, allocate the next block and update the **block_table[seq]**.
* For **prefill**, write in chunks of `block_size` (or partial final block) and update `seq_len` and `block_table[seq]`.
* Dtype: Keep KV in **fp16/bf16** (matching MLX attention/SDPA data paths) and compute softmax in **fp32** (accumulators) inside the kernel; the SDPA code shows the pattern.

**Block tables:**

* `block_tables`: `[B, max_blocks] int32` (per request in batch).
* `seq_lens`: `[B] int32` (kept in sync with how many tokens are actually valid in the last block).
* Copy‑on‑write for prefixes (when branching/speculative decode) = duplicate table (and per‑seq cursors) until divergence, then allocate fresh blocks for the new branch. (Exactly as in PagedAttention.)

---

## 4) Fused Metal kernel (`paged_attention.metal`)

### 4.1 Kernel modes

* **Decode (Lq=1):** One Q vector per (B,H) with **block loop** over K,V tiles referenced by `block_tables[b]`.
* **Prefill (Lq>1):** Small Q tiles (e.g., `Tq × D`) handled the same way, looping over block tables for each sequence with **causal masking** within the last block.

### 4.2 Streaming softmax across blocks

Implement **numerically stable streaming softmax**:

Let `S = softmax([L_1, L_2, ...])` where `L_j = (QK_j^T)` per block.

Maintain for each (b,h,q_row):

* running max `m`, sum `l`, and accumulator `p`:

```
m_new = max(m, max(L_j))
l    = l * exp(m - m_new) + sum( exp(L_j - m_new) )
p    = p * exp(m - m_new) + exp(L_j - m_new) @ V_j
m    = m_new
```

At end: `out = p / l`. (Same idea is used in FlashAttention; PagedAttention adds the **paged gather** and block tables.)

### 4.3 Kernel inputs/strides (suggested)

* Buffers: `q`, `kv`, `block_tables`, `seq_lens`, optional `mask`, optional `rope_freqs/offsets`.
* Constants: `B`, `H`, `D`, `block_size`, `max_blocks`, `dtype tags`, `causal flag`, `scale`.
* Stride arrays for Q, K, V for flexible layouts (like SDPA kernels do).

### 4.4 RoPE (optional fusion)

* For **decode**, apply **RoPE offset** corresponding to `seq_len[b]` to **Q and K** in‑kernel (avoid extra passes). The existing RoPE kernels show how frequency tables and offsets are handled in Metal (template instantiations, type dispatch).
* For **prefill**, either apply RoPE in Python (using `nn.RoPE`) before calling the kernel (simpler), or pass a `rope_freqs` buffer to fuse. The example LlamaAttention shows the RoPE call sites.

### 4.5 Grid/blocking

* One grid dimension over `(B × H × Lq)` rows of Q; inner loops over blocks.
* Vectorize over `D` where possible (the SDPA vector kernel shows patterns like `sdpa_vector_*`).

---

## 5) Memory manager (allocator & cache)

### 5.1 BlockAllocator

* Maintain **free lists** of block IDs.
* `alloc(n)` pulls `n` block IDs; `free(ids)` returns them (on EOS/abort).
* For CoW, when forking a sequence, **do not copy KV**; just **copy the block table**. Only on write to the last (incomplete) block do you allocate and write new data (classic CoW).

### 5.2 PagedKVCache

* Holds K,V backing tensors (once per model layer), created up‑front to **capacity_blocks**.
* Exposes **`append_kv(layer, seq_id, k_tok, v_tok)`**, which writes into the current block (using a per‑seq cursor) or allocates a new one if the block filled.
* Produces **`block_tables[seq_id]`** and **`seq_len[seq_id]`** for the kernel.

---

## 6) Integration into LlamaAttention (MLX style)

**Replace** the KV concatenation + standard SDPA call with a **paged branch** when `cache` is a `PagedKVCache`. The Llama example in MLX docs shows pre‑RoPE and post‑RoPE shapes; we keep those identical.

**File to edit (your model code):** wherever your attention runs (e.g., `llama.py` in `mlx‑lm`), wrap with:

```python
if isinstance(cache, PagedKVCache):
    # Compute Q,K,V projections & apply RoPE (or fuse; see §4.4)
    q = self.query_proj(x).reshape(B, Lq, H, Dh).transpose(0, 2, 1, 3)
    k = self.key_proj(x).reshape(B, Lq, H, Dh).transpose(0, 2, 1, 3)
    v = self.value_proj(x).reshape(B, Lq, H, Dh).transpose(0, 2, 1, 3)
    q, k = self.rope(q, offset=cache.seq_offsets(b)), self.rope(k, offset=cache.seq_offsets(b))

    if Lq > 1:                      # prefill
        cache.append_kv(layer_idx, seq_id, k, v)

    out = mx.fast.paged_attention(
        q=q,
        k_cache=cache.k[layer_idx],
        v_cache=cache.v[layer_idx],
        block_tables=cache.block_table(seq_id),  # shape [B, max_blocks]
        seq_lens=cache.seq_len(seq_id),          # [B]
        block_size=cache.block_size,
        causal=True,
        # optional fused RoPE params
    ).transpose(0, 2, 1, 3).reshape(B, Lq, H*Dh)

else:
    # fallback: standard fast SDPA path (unchanged)
    ...
```

* This mirrors the **existing shapes & RoPE usage** seen in MLX’s Llama example, but moves KV into the **paged cache** instead of concatenating tensors per step.

---

## 7) Serving/runtime hooks (connecting to your `mlx‑lm` changes)

1. **Prefill → Decode transition:**

   * On **prefill**, allocate blocks and write K,V via `append_kv`.
   * On **decode**, compute only the **last token’s Q,K,V**, `append_kv` for that token, then call `mx.fast.paged_attention` with `Lq=1`.

2. **Continuous batching (“top‑up”)**

   * Keep `active ≈ max_num_seqs`. Your open‑loop bench already supports this (`--open-loop`, `--tokens-target`). That avoids the “tail drain” and keeps decode batches high‑occupancy; your phase/tick logs show the penalty when not topped up.
   * Now, because the KV is **paged**, admitting new work **does not require moving old KV**; you just allocate blocks and append, as in vLLM.

3. **EOS & free**

   * When a request finishes, call `cache.free(block_table(seq))`. The blocks go back to the free list immediately.

---

## 8) Testing & validation

### 8.1 Unit tests

* **Allocator invariants:** allocate/free, fragmentation, batch alloc/free; add tests under `python/tests/` (similar to existing tests using `mx.gpu` & memory limits). Use the **multistream/memory** tests as examples for how to orchestrate GPU work & memory.
* **Paged vs. non‑paged parity:**

  * Build a short‑context harness where KV fits in one/two blocks and compare `mx.fast.paged_attention` vs `mx.fast.scaled_dot_product_attention` outputs (RMS error, per‑head). ([ML Explore][1])
  * Include both **prefill** and **decode** cases; vary `seq_len` and `block_size`.

### 8.2 Numerical stability

* Stress long contexts by increasing the number of blocks; verify that streaming softmax stays stable (no NaNs/Inf), matching the reference.

### 8.3 Benches

* Add `benchmarks/python/paged_attention_bench.py` modeled after `gather_mm_bench.py`, `sdpa_bench.py`. Compare:

  * reference gather‑based paged attention (Python fallback)
  * fused Metal paged attention
  * baseline fast SDPA with **concatenated KV** (for small contexts)

The repo already contains **rope** and **sdpa** benches you can mimic.

---

## 9) Performance plan (what moves the needle)

1. **Fused kernel first**: the gather reference establishes correctness; the **Metal** kernel removes Python loop and multiple gathers. Use vectorized loads and shared memory tiles as in SDPA vector variants.
2. **Block size autotune (server knob)**: expose `--block-size` (common choices are 16–64). The best setting is model/SoC dependent; make it a runtime option. (Design supports any fixed block size.)
3. **Fused RoPE for decode**: apply RoPE to Q,K inside the kernel using `seq_len[b]` as offset to avoid extra passes; follow the RoPE kernel’s **freq table** pattern.
4. **Avoid bucket spikes**: your logs show spikes from **bucket transitions**; paged attention reduces re‑bucketing pressure because the kernel handles ragged lengths via block tables. Keep **active** high via your **open‑loop** mode to minimize the ramp/tail share of wall‑time.
5. **Streams**: If you later parallelize Q/K/V projection and attention, reuse MLX stream APIs (see multistream tests). Careful to avoid deadlocks; see the existing test for patterns.

---

## 10) Minimal reference code (snippets)

### 10.1 Python fallback (excerpt)

```python
# python/mlx/core/fast/_paged_attention.py
import mlx.core as mx

def paged_attention(...):
    if _metal_kernel_is_available():
        return _paged_attention_metal(...)
    return _paged_attn_reference(...)

def _apply_causal_and_len_mask(logits_t, seq_lens, t, block_size):
    # logits_t: [B,H,1,Bt]; Bt <= block_size
    # Mask positions >= seq_lens[b] in the *last* block
    # and all future blocks beyond last valid block.
    ...
    return logits_t
```

### 10.2 Metal launcher skeleton

```cpp
// mlx/backend/metal/paged_attention.cpp
#include "metal_context.h"
#include "metal_helpers.h"
using namespace mlx::metal;

void paged_attention_launch(
    const Tensor& q,
    const Tensor& k_cache,
    const Tensor& v_cache,
    const Tensor& block_tables,
    const Tensor& seq_lens,
    /* params: block_size, scale, ... */,
    Tensor& out) {

  auto& ctx = metal::context();
  auto pipeline = ctx.get_pipeline("paged_attention_f16_d", {/*specializations*/});
  // encode buffers: q, k_cache, v_cache, block_tables, seq_lens, out, strides, consts
  // dispatch grid over (B*H*Lq)
}
```

### 10.3 Metal kernel skeleton (decode path)

```metal
// mlx/backend/metal/kernels/paged_attention.metal
#include <metal_stdlib>
using namespace metal;

kernel void paged_attention_decode(
  device const half* q,
  device const half* k_cache,
  device const half* v_cache,
  device const int*  block_tables,
  device const int*  seq_lens,
  constant int& B, constant int& H, constant int& D,
  constant int& block_size, constant int& max_blocks,
  constant float& scale, constant bool& causal,
  device half* out,
  uint tid [[thread_position_in_grid]]) {

  // map tid -> (b,h)
  // load q[b,h,:]  (vectorized)
  // streaming-softmax state: m, l, p (float accumulators)
  // for each block id in block_tables[b]:
  //   load K,V tile pointers via block id
  //   apply RoPE offset if fused
  //   logits = dot(q, K^T) * scale
  //   causal trim for last positions
  //   update streaming softmax (m,l,p)
  // write out[b,h,:] = p / l
}
```

---

## 11) Validation matrix

* **Parity tests**:

  * L=32/64/128, H×D small (e.g., 8×128), B=16/32/64 → compare to `fast.scaled_dot_product_attention` (contiguous KV). ([ML Explore][1])
* **Edge cases**:

  * `seq_len` exactly multiple of `block_size`; `seq_len % block_size = 1`; no KV (cold start); all EOS mid‑run.
* **Long‑context**:

  * Thousands of tokens → many blocks; verify numerical stability and memory reuse.
* **Throughput**:

  * Bench (prefill vs decode; mixed) at various concurrency and page sizes; confirm improvement when open‑loop keeps **active** high.

---

## 12) How this connects to your `mlx‑lm` scheduler & bench

* Your **open‑loop** harness already keeps decode **topped up** (fixes the earlier drain/tail effect). Now it can **fully exercise** paged attention’s strength: **no re‑concat**, just **append K/V + block tables**, and **constant‑shape kernel** across ragged sequences.
* Keep the tokens‑target methodology and report **steady‑state** tokens/s, plus **phase ramp/tail** shares to confirm reductions in spikes and tail time after switching to paged attention.

---

## 13) “Done” criteria (PR‑ready)

* `mlx.core.fast.paged_attention` public API + Python fallback + Metal fused op.
* `PagedKVCache` + `BlockAllocator` + unit tests (alloc/free; CoW; seq_len/block_table correctness).
* Parity tests vs. `fast.scaled_dot_product_attention` on small cases and mixed prefill/decode.
* Benches: reference vs. fused vs. baseline; plots + CSV; command lines.
* Integration hooks in your attention layer and runtime (feature flag `--paged-attn`).
* Documentation: one pager describing block layout, API, and gotchas (RoPE offsets, masks).

---

## 14) References (supporting the design)

* **MLX fast attention API** (pattern for our operator): “Scaled Dot Product Attention — MLX documentation.” ([ML Explore][1])
* **Custom Metal kernels in MLX** (registering via `mx.fast.metal_kernel()` and kernel authoring patterns). ([ML Explore][2])
* **Existing MLX Metal SDPA & RoPE kernels** (file layout, instantiation, dtype handling).
* **MLX Llama attention (docs example)** for shapes, RoPE offsets, causal mask style (used for parity).
* **Gather GEMM benches** (for reference implementation and micro‑tuning).
* **PagedAttention paper** (paged KV, block tables, CoW, streaming softmax across tiles).  

---

## 15) Suggested patch plan (small, reviewable PRs)

1. **PR‑1 (reference path):** Python‑only `PagedKVCache` + `BlockAllocator` + **reference** `paged_attention` using gather ops. Parity tests + small benches.
2. **PR‑2 (Metal fused decode):** `paged_attention.metal` (decode), launcher, wrapper; keep Python prefill path. Bench + doc update.
3. **PR‑3 (Metal fused prefill + RoPE fuse option):** add prefill mode + (optional) in‑kernel RoPE; more benches, ablations for block_size.
4. **PR‑4 (server integration):** enable in `mlx‑lm` runtime with your **open‑loop** scheduler; EOS/free paths; feature flags; end‑to‑end bench grids and plots.
5. **PR‑5 (polish):** CoW optimizations, allocator stats, minor kernel tuning (vector widths, LDS usage), masks & edge cases hardening.

Each PR includes **before/after tokens/s**, p50/p95, and exact commands—consistent with your benchmarking discipline.

---

### Closing note

The plan above stays tightly aligned with **the MLX stack that exists today** (fast attention API, Metal kernels, gather ops, RoPE, test/bench scaffolding) and with the **PagedAttention** design that makes continuous batching scale without KV re‑writes. Build correctness once with gather‑based reference, then switch to the fused kernel to harvest throughput—staying compatible with your `mlx‑lm` scheduler and open‑loop benches.

If you want, I can sketch the exact signatures for the **C++ launcher** and the **Metal argument buffers** next, using the SDPA kernel as a template, and draft a minimal **allocator test file** under `python/tests/` to copy into your repo.

[1]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html "mlx.core.fast.scaled_dot_product_attention — MLX 0.29.4 documentation"
[2]: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.rope.html "mlx.core.fast.rope — MLX 0.29.4 documentation"
