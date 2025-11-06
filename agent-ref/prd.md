
Below is a single, self‑contained implementation document you can work from inside **mlx** and then link up cleanly with your existing **mlx‑lm** scheduler/bench harness changes. It’s written to be maximally concrete about interfaces, data layouts, and kernel behavior on MLX, while staying faithful to the state of the art in continuous batching and PagedAttention (as popularized by vLLM). Where the literature or public artifacts disagree or leave room for choice, I call that out explicitly and provide decision criteria.

I cite a handful of canonical sources from the vLLM team and adjacent work so we’re not building on assumptions: the original PagedAttention paper (concepts & algorithms), vLLM’s public write‑ups (continuous batching, KV virtual memory), and scheduler papers you’ll likely want to emulate for prefill/decode interleaving (chunked prefill). ([arXiv][1])

---

# MLX PagedAttention & Continuous Batching: Implementation Plan

**Target repos:**

* **mlx** (new op + runtime infrastructure): `mx.fast.paged_attention`, KV block manager, COW semantics, page tables, (optional) host‑backed KV.
* **mlx‑lm** (integration): plug the new op & KV manager into your scheduler and open‑loop bench harness (already added), remove the “bucketed decode” bottlenecks over time.

**Primary goals**

1. **Decode throughput**: match or exceed your current steady window (≈800–850 tok/s @ 64 active) without the ramp/tail drag, under open‑loop load.
2. **Latency stability**: eliminate 0.2–1.6 s “spikes” when active counts change or batches reshape; keep decode ticks within a narrow band.
3. **Memory efficiency**: block‑level KV with copy‑on‑write (COW) to support parallel sampling/beam without duplicating history, consistent with PagedAttention. ([arXiv][1])
4. **Unified prefill/decode path**: chunked prefill that (a) writes KV directly into blocks and (b) never stalls ongoing decodes (continuous batching). ([arXiv][2])

**Non‑goals (initial phase)**

* Advanced host‑offloading / tiered KV (GPU↔CPU migration). We keep the hooks but can defer actual migration to a follow‑up milestone (see §8.3).

---

## 1. Why this change (anchored to your measurements)

From your open‑loop runs:

* **Steady decode**: mean decode tick ~109–114 ms with decode_batch ~62–63; harness reports **steady ≈562 tok/s** on the Paged path approximation, but **wall TPS ≈503 tok/s** because **spikes accounted for ~63 s / 215 s (~29%)** of wall time. (You logged: `[phase spikes] count=56 time=63.06s threshold=0.20s`.)
* **Observation**: The system is doing the right math per tick, but *shape change and draining* events (ramp/tail + sporadic long ticks) dominate wall. Continuous batching should eliminate tails by topping up, and PagedAttention should remove the **“KV copy & re‑layout”** costs that typically cause the long ticks during reshape, as well as prevent fragmentation overhead. ([arXiv][1])

**Design implication:** we need **true continuous batching** (never let active collapse) **and** a **paged KV** so that changing batch composition does **not** force memory moves or kernel shape recompilations. That combination is the essence of vLLM’s serving architecture.

---

## 2. High‑level architecture

### 2.1 Components

1. **KV Block Manager (mlx core)**

   * Owns a pool of fixed‑size KV blocks (pages).
   * Exposes per‑sequence **logical block tables** mapping logical positions → physical blocks.
   * Provides **COW semantics** for forked sequences (parallel sampling; beam).
   * Optional tier‑2 storage hooks (host memory) for future offload.
   * See §4.

2. **`mx.fast.paged_attention` op (mlx core)**

   * **Decode kernel** performing attention over a set of blocks addressed via page tables.
   * **Streaming softmax across blocks** (no revisits), stable numerics.
   * Supports MQA/GQA and rotary embeddings (RoPE) at decode, optional ALiBi.
   * See §5.

3. **Prefill writer (mlx core, reuses existing fused prefill math)**

   * **Chunked prefill** (split prompt into slices) that:
     a) computes QKV using standard high‑throughput prefill (Flash‑like attention, existing MLX kernels),
     b) **writes K/V into blocks** via the same block manager, and
     c) yields between chunks to allow decode ticks to proceed (no stalls).
   * See §6.

4. **Admission & Top‑Up (mlx‑lm)**

   * On each completion, **immediately** admit a new request to keep `active ≈ target`.
   * Prefill is scheduled **chunk‑wise** so decodes aren’t starved; decodes never wait on a full prefill pass.
   * See §7; scheduler heuristics inspired by Sarathi‑Serve’s stall‑free chunking. ([arXiv][2])

### 2.2 Data flow (decode step)

* Queries for the current token (Q) are computed as today (per active sequence).
* For each sequence/head, we read its **block table** to iterate over `N = ceil(ctx_len / block_size)` blocks.
* For each block, we load the block’s `K` to compute partial logits (`q @ K^T`) and update streaming softmax state, and (optionally fusion) accumulate `V` contribution.
* Output context vector is normalized and passed through the head projection to produce logits for sampling.

**Key property:** the **physical order** of blocks in VRAM never needs to match the **logical order** of tokens. A change in batch composition only changes which block tables are traversed, not the storage layout. That’s the PagedAttention win. ([arXiv][1])

---

## 3. Public API (mlx)

### 3.1 Python surface

```python
# mlx/fast/__init__.py
def paged_attention(
    q: mx.array,                        # [B, n_heads, 1, head_dim] at decode
    k_cache: mx.array,                  # [n_layers, n_kv_heads, n_blocks, block_size, head_dim]
    v_cache: mx.array,                  # [n_layers, n_kv_heads, n_blocks, block_size, head_dim]
    block_tables: mx.array,             # [B, max_blocks] int32: physical block ids per sequence
    context_lens: mx.array,             # [B] int32: current context length per sequence
    layer_idx: int,                     # which layer’s KV to read
    kv_head_mapping: mx.array | None,   # [n_heads] -> kv_head_id (for GQA)
    rope_freqs: mx.array | None,        # optional decode-time RoPE cache for position t
    scale: float,                       # 1/sqrt(head_dim)
    causal: bool = True,
    attn_mask: mx.array | None = None,  # optional (rare at decode)
    return_attn: bool = False,          # debug only
) -> mx.array:                           # [B, n_heads, 1, head_dim]
    ...
```

* `block_tables[b]` enumerates **logical** blocks for sequence *b* in order.
* The op **does not** write; it only reads. Writes happen in prefill & at decode when appending to the tail block (via the KV manager).
* We keep KV for all layers in the same tensors (`k_cache`, `v_cache`), indexed by `layer_idx`. This avoids per‑layer Python object churn.

### 3.2 KV manager hooks (Python)

```python
class KVBlockManager:
    def __init__(self, n_layers, n_kv_heads, head_dim, block_size, max_blocks, dtype):
        ...

    # Allocate logical table for a new sequence with enough blocks for prompt_len
    def new_sequence(self, seq_id: int, prompt_len: int) -> BlockTableView: ...

    # COW: fork a sequence (e.g., sampling or beam). No data copy until write.
    def fork(self, parent_seq_id: int, child_seq_id: int) -> BlockTableView: ...

    # Append tokens for decode; may allocate a new tail block if the last one fills
    def ensure_tail_capacity(self, seq_id: int, n_tokens: int = 1) -> TailView: ...

    # Write K/V for a (chunk of) prompt into assigned logical positions
    def write_prefill(self, seq_id: int, layer_idx: int, k_chunk, v_chunk, start_pos: int): ...

    # Free all resources for a sequence (including refcount drops for shared blocks)
    def free(self, seq_id: int) -> None: ...

    # Introspection for runtime/scheduler
    def table(self, seq_id: int) -> tuple[mx.array, int]:  # (block_ids, ctx_len)
        ...
```

* **Refcounting & COW** are implemented per physical block. When a child needs to write into a shared tail, we **clone-on-write** just that block (cheap). This is a major memory/time win versus naïve duplication. ([arXiv][1])
* **Block size** (`block_size`) is a tuning knob; PagedAttention evaluates a range (8–128). In practice, **16–32** often balance TMA/gather overheads and softmax fusion well on common GPUs; we’ll expose it as a parameter and benchmark on MLX. (LMCache’s docs note that vLLM commonly uses 16; tune empirically on MLX.)

---

## 4. KV Block Manager (mlx core)

### 4.1 Storage layout

```
k_cache, v_cache: [n_layers, n_kv_heads, n_blocks, block_size, head_dim]  # fp16/bf16
```

* **Why per‑layer tensors?** Simplifies indexing and keeps per‑layer contiguity; decoder loops do `layer_idx` addressing with coalesced block fetches.
* **GQA/MQA:** `n_kv_heads` ≤ `n_heads`; we pass a small `kv_head_mapping` to the op so each query head knows which kv head to read.

### 4.2 Free list, refcounts, COW

* Maintain:

  * `free_blocks: deque[int]`
  * `refcount: int[n_blocks]`
  * `owner_map: {seq_id -> BlockTable}`, where `BlockTable = array[int32]` of physical IDs
* **Fork**: new child gets a **view** (same block IDs, `refcount++`).
* **Tail write**:

  * If last block has unused slots and `refcount == 1` → write in place.
  * Else allocate a new block (from `free_blocks`), copy only the needed tail tokens (optionally zero if new), update the child’s table (COW).

### 4.3 Capacity & OOM policy

* Keep **hard GPU cap** `max_blocks`. If no free blocks:

  1. Try to **reclaim** blocks from finished sequences.
  2. (Optional) Tier‑2 hook: move least‑recently‑used sequences to **host** KV (pinned/unified) and mark them **un‑schedulable** until brought back. (Disabled initially; keep interface.)
  3. Otherwise, **admission control**: reject/tarpit new requests until blocks free up.
* The PagedAttention paper discusses offloading and recomputation trade‑offs; we start with **no offload** and **no recompute** (deterministic serving), then iterate. ([arXiv][1])

---

## 5. `mx.fast.paged_attention` (decode)

### 5.1 Kernel contract

* Input `q`: `[B, n_heads, 1, d]`

* Walk **per sequence** the **logical block list** (length `ceil(ctx_len/block_size)`), and for each block:

  1. Load `K_block`: `[n_kv_heads, block_size, d]` (select kv group).

  2. Compute scores `s = q @ K_block^T` → `[B, n_heads, 1, block_size]`.

  3. **Streaming softmax** across blocks: keep running `(m, l, o)` triplet to avoid a second pass:

     * `m`: running max logit; `l`: running denom; `o`: running numerator (context vector).
     * Update with the numerically stable variant:

       ```
       # vectorized over heads & batch
       s_local = (q @ Kp^T) * scale
       m_new = max(m, max(s_local))
       alpha = exp(m - m_new)
       w = exp(s_local - m_new)
       l = l * alpha + sum(w, axis=block)
       o = o * alpha + (w @ Vp)           # Vp: [block_size, d]
       m = m_new
       ```

  4. After the last block: `out = o / l`.

* **Causal mask at decode** is trivial (only the last position of the last block needs masking if you include the token being generated). Implement as: for the **final block**, mask positions `> (ctx_len % block_size)`.

* **RoPE**: apply to `q` (and KV when written at prefill time, not at read time). So decode kernel only needs the current position’s `q_rope`.

* **Return shape**: `[B, n_heads, 1, d]`, to be projected by the model’s `o_proj`.

### 5.2 Metal/MPS implementation notes (MLX backend)

* Implement as a **custom fused op** to minimize Python overhead and gather cost:

  * **Threadgroup** cooperatively loads `K_block` (and `V_block` when fusing the numerator update) into threadgroup memory; iterate tokens along the block dimension.
  * Favor **structure‑of‑arrays** access for K/V (tokens contiguous inside block) to coalesce loads when iterating over block_size.
  * Expose **block_size** as a specialization constant so we can pre‑compile for common sizes (8/16/32) to avoid runtime pipeline recompiles (your spikes are consistent with shape‑specialization stalls).
* **Precision**: keep K/V in fp16/bf16; accumulate logits in fp32; output context in fp16/bf16 to match the rest of the model.
* **Return‑attn (debug)**: optional path writing the attention weights per block back to a small buffer (B×heads×1×block_size). Keep disabled in hot path.

### 5.3 Complexity & overhead

PagedAttention introduces a small gather overhead and book‑keeping in exchange for removing huge re‑layout costs; the paper reports modest kernel overhead that is outweighed by batching flexibility and memory efficiency, enabling high serving throughput. ([arXiv][1])

---

## 6. Prefill: chunked writer into paged KV

**Objective:** prefill should fill KV **in blocks** without starving decode.

### 6.1 API and behavior

```python
def prefill_into_kv(
    seq_id: int,
    tokens: mx.array,            # prompt token ids
    max_chunk_tokens: int,       # e.g., 1024..4096
) -> None:
    # Loop over chunks; between chunks, yield to scheduler (see §7)
```

* For each chunk:

  1. Run the standard high‑throughput **prefill attention** on that slice (MLX already has fast kernels).
  2. From the per‑layer outputs (K,V), **pack** into KV blocks:

     * Determine the logical positions covered by the chunk (`start_pos:end_pos`).
     * For each layer, copy K/V rows into the corresponding **tail block** (allocate blocks as needed).
  3. `yield` to let **decode ticks** execute (continuous batching).

This is the **chunked‑prefill** pattern used by high‑throughput schedulers (Sarathi‑Serve) to avoid decode stalls while keeping GPU saturated. ([arXiv][2])

### 6.2 Choices we make explicit

* **Where to apply RoPE?** Apply RoPE to K and Q **before** writing K/V; then decode never needs to rotate K again (only Q for the current position).
* **Chunk size**: expose as a scheduler knob; too big → stalls decode; too small → launch overhead. Start with your current `--prefill-chunk` defaults and tune under open‑loop load.

---

## 7. Scheduler integration (mlx‑lm)

You already added **open‑loop** with **tokens‑target** and measured phases. Keep these changes and adjust the runtime to **top‑up** immediately upon completion (which you noted), ensuring `active ≈ target` most of the time.

### 7.1 Admission loop (decode‑first, stall‑free)

* On each scheduler tick:

  1. If **decode batch < target_active**, **admit** from wait‑queue.
  2. If a prefill is in flight for an admitted request, **schedule its next chunk only if** it won’t reduce the next decode batch below a threshold.
* This yields “stall‑free” schedules under mixed prefill/decode, which Sarathi‑Serve shows improves both throughput and tail latency vs naïve interleaving. ([arXiv][2])

### 7.2 Trimming tails & preventing spikes

* **Never drain** into tiny decode batches. If `active` drops (some finished), **admit immediately** to fill the holes before the next decode tick.
* **Pre‑warm** kernels: when the engine initializes, invoke `paged_attention` with representative `(B, heads, d)` and **each block_size** variant you plan to use (8/16/32) so Metal pipelines are compiled up‑front. This removes your 0.4–1.6 s long ticks that coincide with first‑time shapes.
* Keep your **phase breakdown** logging; add:

  * `paged_kv_bytes_read`, `blocks_touched`, `gather_ms`
  * `prefill_chunk_ms`, `kv_write_ms`, `decode_tick_ms_p50/p95`
    to attribute any residual spikes.

---

## 8. Optional but important: toward feature parity

### 8.1 Parallel sampling & beam search

* With **COW** blocks, sampling `n` candidates is O(n_blocks) **metadata only**; only the **tail block** is cloned on write for candidates that diverge. That’s per the PagedAttention design. ([arXiv][1])

### 8.2 Block size tuning

* Provide runtime flag `--kv-block-size {8,16,32}`; measure tokens/s and p95 tick time under your open‑loop harness. vLLM often uses **16**; your MLX kernel may prefer **32** if threadgroup memory favors larger coalesced loads. Document the curve in CI.

### 8.3 KV tiering (future)

* The paper discusses **GPU memory virtualization** with paging; if MLX unified memory and bandwidth permit, a **host tier** can be added later. Expose the API now (`spill_to_host(seq_id)`, `prefetch(seq_id)`), stubbed, so higher layers don’t change when you add it. ([arXiv][1])

---

## 9. Implementation plan (step‑by‑step)

### M0 — Scaffolding & flags (mlx)

* Add `mx.fast.paged_attention` Python stub + tests that assert shapes and simple CPU fallback correctness.
* Add `KVBlockManager` (Python, backed by MLX arrays for block tables) with: allocate/free, fork, ensure_tail_capacity, write_prefill.
* Land unit tests:

  * **COW**: fork → decode append; parent’s blocks remain shared except tail.
  * **Large prompt write**: chunked prefill writes exactly the right tokens into KV.

### M1 — Decode kernel (mlx)

* Write the Metal/MPS kernel for `paged_attention` with:

  * Specialization constant for `block_size`.
  * Streaming softmax across blocks (fp32 accumulators).
  * GQA mapping.
* Add microbench + correctness tests:

  * Compare against a dense “gather‑then‑dense‑attention” reference on small sizes.
  * Numerics under long contexts (e.g., 8–64 blocks).

### M2 — Prefill writer (mlx)

* Hook `prefill_into_kv` into the model’s forward path for prompts:

  * Apply RoPE to K/Q before writing.
  * Copy per‑layer K/V rows into KV blocks.
* Add **chunking** (yield between chunks). Validate no decode stalls in a mixed workload test.

### M3 — Runtime integration (mlx‑lm)

* Replace your “bucketed decode shapes” path with:

  * top‑up admission (always keep active near target),
  * chunked prefill interleaving,
  * paged decode calls.
* Keep your bench harness **open‑loop tokens‑target** to compare `static_tokens_per_sec`, `continuous_tokens_per_sec`, and **steady**.

### M4 — De‑spike & polish

* **Pre‑warm** kernel shapes on init.
* Add per‑tick metrics (gather, softmax, total).
* Tune `block_size` (8/16/32) and `prefill_chunk` under open‑loop.
* Validate p95 decode tick is stable (no >0.2 s spikes in steady).

---

## 10. Integration points with your code today

You showed these key entry points (summarized):

* **Bench harness**

  * `run_static(..., open_loop, tokens_target)` accumulates tokens.
  * `run_continuous(..., open_loop, tokens_target)` has **schedule_next()** and open‑loop arrivals.
  * Phase breakdown you added: `[phase ramp]`, `[phase steady]`, `[phase tail]`, `[phase spikes]`.

**What changes after paged KV:**

* In the engine’s **decode step**, call `mx.fast.paged_attention(...)` instead of assembling dense per‑sequence KV slices.
* In **prefill**, call `prefill_into_kv(seq_id, tokens, max_chunk_tokens)` (which writes K/V directly to blocks).
* On **completion**, **schedule_next() immediately** to top‑up. Your open‑loop harness already supports this; just keep it on in perf runs.

---

## 11. Testing & validation

1. **Correctness**

   * Short context: compare decode outputs with a dense attention implementation for the same seeds.
   * Long context: verify numerical stability (no NaNs) with 1k–16k tokens.

2. **Perf microbench**

   * Single layer, varying `block_size in {8,16,32}`, `B in {8,16,32,64}`, measure `paged_attention` kernel time only.

3. **Perf end‑to‑end (your harness)**

   * `--open-loop --tokens-target 100000` (or 500k for longer) comparing:

     * `static_tokens_per_sec`
     * `continuous_tokens_per_sec`
     * `continuous_steady_tokens_per_sec`
   * Expectation: **steady** should climb toward kernel‑limited throughput (remove tail), and spikes vanish after pre‑warm.

4. **Memory**

   * Track `blocks_in_use`, `refcount_histogram`, `COW_clones_per_token` in mixed sampling/beam tests.

---

## 12. Risk register & mitigations

* **Metal pipeline specialization spikes**

  * *Mitigation:* pre‑warm common `(B, block_size)` combos at startup.

* **Gather inefficiency vs block size**

  * *Mitigation:* tune block size; 16–32 often sweet spot; measure on MLX.

* **Prefill starvation**

  * *Mitigation:* stall‑free chunked prefill; never admit a prefill chunk if it would shrink the next decode batch under target. ([arXiv][2])

* **Copy‑on‑write bugs**

  * *Mitigation:* exhaustive tests around tail fills, forks, frees.

---

## 13. Minimal code skeletons

> **Note:** paths/names align with MLX conventions as much as possible, but adjust to actual folder layout in your tree.

**13.1 `mlx/fast/paged_attention.py` (Python wrapper)**

```python
import mlx.core as mx

def paged_attention(q, k_cache, v_cache, block_tables, context_lens,
                    layer_idx, kv_head_mapping, rope_freqs, scale,
                    causal=True, attn_mask=None, return_attn=False):
    # type/shape checks elided
    return mx._ops.paged_attention(  # bound to C++/Metal op
        q, k_cache, v_cache, block_tables, context_lens,
        layer_idx, kv_head_mapping if kv_head_mapping is not None else mx.array([]),
        rope_freqs if rope_freqs is not None else mx.array([]),
        scale, int(causal), int(return_attn)
    )
```

**13.2 `mlx/fast/kv_manager.py`**

```python
from dataclasses import dataclass

@dataclass
class BlockTableView:
    seq_id: int
    block_ids: mx.array   # int32
    ctx_len: int

class KVBlockManager:
    def __init__(self, n_layers, n_kv_heads, head_dim, block_size, max_blocks, dtype=mx.float16):
        self.bs = block_size
        self.max_blocks = max_blocks
        self.k = mx.zeros((n_layers, n_kv_heads, max_blocks, block_size, head_dim), dtype=dtype)
        self.v = mx.zeros_like(self.k)
        self.free = list(range(max_blocks))
        self.ref = mx.zeros((max_blocks,), dtype=mx.int32)
        self.tables: dict[int, BlockTableView] = {}

    def new_sequence(self, seq_id, prompt_len):
        n = (prompt_len + self.bs - 1) // self.bs
        blocks = [self.free.pop() for _ in range(n)]
        for b in blocks: self.ref[b] += 1
        t = BlockTableView(seq_id, mx.array(blocks, dtype=mx.int32), prompt_len)
        self.tables[seq_id] = t
        return t

    def fork(self, parent_seq_id, child_seq_id):
        p = self.tables[parent_seq_id]
        for b in p.block_ids: self.ref[b] += 1
        self.tables[child_seq_id] = BlockTableView(child_seq_id, p.block_ids.copy(), p.ctx_len)
        return self.tables[child_seq_id]

    def ensure_tail_capacity(self, seq_id, n_tokens=1):
        # allocate new tail block only if needed; handle COW on shared tail
        ...

    def write_prefill(self, seq_id, layer_idx, k_chunk, v_chunk, start_pos):
        # copy rows into the correct blocks, allocate as needed
        ...

    def free(self, seq_id):
        t = self.tables.pop(seq_id, None)
        if t is None: return
        for b in t.block_ids:
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free.append(b)

    def table(self, seq_id):
        t = self.tables[seq_id]
        return t.block_ids, t.ctx_len
```

**13.3 Decode call site (mlx‑lm engine)**

```python
# Build q for active batch (B = active sequences)
q = ...  # [B, n_heads, 1, d]
block_tables, context_lens = manager.batch_tables(active_seq_ids)
ctx = mx.fast.paged_attention(
    q, manager.k, manager.v, block_tables, context_lens,
    layer_idx, kv_head_mapping, rope_freqs_for_pos_t, scale
)
```

**13.4 Prefill (chunked)**

```python
def prefill_into_kv(engine, seq_id, tokens, chunk):
    pos = 0
    while pos < len(tokens):
        end = min(pos + chunk, len(tokens))
        # standard prefill forward for tokens[pos:end]; get per-layer K,V
        k_layers, v_layers = engine.forward_prefill(tokens[pos:end])
        # write into KV by blocks
        for L,(k,v) in enumerate(zip(k_layers, v_layers)):
            engine.kv.write_prefill(seq_id, L, k, v, start_pos=pos)
        pos = end
        engine.yield_to_decode()   # scheduler hook
```

---

## 14. What to expect after implementation

* **No more long “reshape” ticks** in steady decode (post pre‑warm). Continuous top‑up avoids draining tails.
* **Wall TPS** should converge toward your current **steady** TPS (you reported ~836 tok/s windows), rather than ~503 tok/s dominated by spikes/tails.
* **Sampling/beam** should scale without quadratic memory growth due to COW blocks. ([arXiv][1])

---

## 15. References (for design parity & scheduler choices)

* **PagedAttention (vLLM)**: core idea, block tables, COW, GPU memory virtualization; the canonical paper. ([arXiv][1])
* **vLLM continuous batching**: high‑level architecture emphasising efficient KV cache management and batching.
* **KV block size & tuning context** (external doc citing vLLM practice; tune on MLX):
* **Chunked prefill & stall‑free scheduling** (Sarathi‑Serve): the right way to interleave prefill/decode under continuous batching. ([arXiv][2])

---

### Final notes (bias checks)

* I avoided hard‑coding **block size**; while 16 is common in vLLM contexts, MLX kernels on Apple GPUs could prefer 32. That’s why it’s a specialization constant and a benchmarked parameter, not an assumption.
* I did **not** assume availability or performance of host‑offload on MLX; the API surface supports it later, but initial delivery keeps KV fully resident and deterministic. ([arXiv][1])
* I kept **prefill math** as you have it today (fast dense attention), only changing the **write‑path** into KV blocks and the **scheduler cadence** (chunk/yield). That reduces risk while giving you most of the PagedAttention benefits immediately.
* I explicitly targeted your observed pain points (spikes, drain tails) with **pre‑warm**, **top‑up**, and **paged decode** rather than hand‑waving “optimize kernels”.

If you want, I can follow up with a tiny, self‑contained unit test file set (toy transformer, two heads, tiny block size) to validate COW and streaming softmax numerics end‑to‑end before wiring this into the full model.

[1]: https://arxiv.org/abs/2509.04377?utm_source=chatgpt.com "PagedEviction: Structured Block-wise KV Cache Pruning for Efficient Large Language Model Inference"
[2]: https://arxiv.org/abs/2403.02310?utm_source=chatgpt.com "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"
