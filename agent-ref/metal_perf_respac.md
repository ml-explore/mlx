
Below is a practical guide for **what to use, when, and how**, plus what to look for to compare your PagedAttention kernels versus the existing MLX SDPA kernels.

---

## The three Xcode tools you should use (and why)

### 1) **Xcode GPU Frame Capture** (a.k.a. GPU Frame Debugger)

**When:** Deep inspection of one or a few dispatches; verifying kernel resource usage and specialization.
**Why:** Lets you open a compute dispatch, confirm:

* **Which compute pipeline & function‑constant specialization** was used (e.g., `BLOCK_SIZE=32/64`, your `HD_TILE`).
* **Threadgroup size** vs device **threadExecutionWidth**, **maxTotalThreadsPerThreadgroup**.
* **Threadgroup/shared memory** allocation size (are you over‑allocating?).
* **Resource bindings** (argument buffers, buffer offsets/strides) and whether you’re loading/storing in the layout you think you are.
* **Barrier counts** and sequencing (sanity‑check your ping‑pong loads for K/V tiles).

> Use this after each major kernel change to avoid “it runs but with the wrong specialization” problems, and to confirm your TG memory footprint and group size are what you expect.

**How:** Run your microbench, hit the “Capture GPU” button (or use `MTLCaptureManager` programmatically) and inspect the captured dispatch.

---

### 2) **Metal System Trace** (Instruments)

**When:** End‑to‑end timeline analysis across CPU and GPU with **minimal overhead**.
**Why:** Shows:

* Command buffer/encoder scheduling, queue depth, CPU submission time, driver latency.
* Overlap between prefill, decode, and other work.
* Spikes and stalls (e.g., a few 800–1600 ms outliers like the “spikes” your bench is reporting).

**How:** `xctrace record --template "Metal System Trace" --time-limit 10s --attach <pid>` (or use Instruments GUI).
**What to look for:**

* **Regular cadence** of your paged_attention dispatches in steady state.
* **No frequent pipeline recompiles** (see few unique pipeline objects; pipeline creation should be outside the hot path).
* **Minimal command buffer bubbles** (idle gaps) once steady state begins.
* If you see oscillations, correlate them with your bench’s “spikes” phase.

---

### 3) **GPU Counters (Instruments → GPU Counters)**

**When:** You want *numbers* to drive tile and threadgroup tuning.
**Why:** Counters expose whether you’re **compute‑bound** or **memory‑bound** and whether you’re wasting time in **barriers** or **under‑occupancy**. Look for:

* **Threadgroup / SIMD occupancy**: Is your TG size too small or register/TG‑mem pressure capping active warps?
* **ALU utilization vs. memory throughput**: If memory throughput is pegged but ALU is low, reduce per‑tile bytes (smaller BLOCK_SIZE or better vectorized loads) or increase reuse (bigger HD tile per shared‑load).
* **L2/DRAM read/write BW**: Are K/V loads saturating memory? Vectorize (`half4/float4`) and align head‑dim strides.
* **Barrier / synchronization stalls**: Too many TG barriers => rework the inner loop to double‑buffer K/V loads more effectively.

> Use GPU Counters after you have a working kernel to choose between `BLOCK_SIZE=32` vs `64` and to pick a threadgroup size (128/256/512 threads) that maximizes occupancy without blowing TG memory.

---

## Low‑overhead timing you should add to your launcher (for every run)

Relying only on Instruments is cumbersome for inner‑loop tuning. Add **programmatic GPU timing** so your bench can print per‑dispatch GPU ms alongside tokens/sec:

* After the command buffer completes, read **`MTLCommandBuffer.gpuStartTime`** and **`gpuEndTime`** (exposed in Swift/ObjC). `gpuEndTime - gpuStartTime` is the **actual on‑GPU time** for all work in that command buffer.
* Wrap each paged_attention dispatch in its own command buffer (or stamp with **os_signpost** markers) so you can attribute GPU time to the op.
* Dump **CSV**: `tokens, seq_len, heads, head_dim, BLOCK_SIZE, TG_SIZE, gpu_ms` to correlate with your open‑loop bench phases.

This gives you **stable numbers** for CI/regression and aligns with your bench’s **steady vs ramp/tail** breakdown.

---

## How to compare against “standards” in MLX

Use **both** macro- and micro‑level comparisons:

1. **Macro (end‑to‑end)**: Run your open‑loop harness with a fixed token budget and compare **tokens/sec** between:

   * `mx.fast.scaled_dot_product_attention` (existing MLX SDPA path) and
   * `mx.fast.paged_attention` (your Metal kernel),
     under identical shapes (`B, Hq, Hkv, D`, seq length distribution, block size).

2. **Micro (kernel‑level)**: In a controlled microbench (single op, prewarmed):

   * Log **GPU ms/dispatch**, **occupancy**, **ALU vs memory utilization**, and **bytes read/written per token**.
   * Aim to reduce the gap between **steady‑state tokens/sec** and the microbench’s kernel‑only ceiling by removing submission/fragmentation overhead (pipeline caching, argument buffers, prebuilt resources).

3. **Numerical parity**: Keep your existing parity tests (dense vs paged) and run them through the **Metal** path; accumulate in FP32, cast to FP16/BF16 on store.

---

## What to measure and how to act on it (checklist)

**Per dispatch (programmatic):**

* GPU time (ms), TG size, BLOCK_SIZE, HD_TILE.
* Bytes touched (model‑derived): read K page + V page + Q; write O.
* Derived **operational intensity**: FLOPs / byte. If low, expect **memory‑bound** → shrink BLOCK_SIZE, improve vectorization; if high, expect **compute‑bound** → consider larger TG or more ILP.

**In GPU Counters:**

* **Occupancy**: Too low → increase TG size (e.g., 128 → 256) or reduce TG‑mem (smaller tiles).
* **ALU vs Mem**: If ALU underutilized and Mem near peak → memory bound. If vice‑versa → compute bound; look for unvectorized math or serialized reductions.
* **Barrier / Stall**: If high → reduce number of barriers per tile (double‑buffer K/V loads more aggressively).

**In System Trace timeline:**

* **Pipeline churn**: Prebuild pipelines for `{BLOCK_SIZE ∈ {32,64}}×{HD_TILE ∈ {8,16}}` and cache them.
* **CPU side bubbles**: Move resource/descriptor setup out of the hot loop (argument buffers help).

---

## Other tools you might consider (but Xcode is primary)

* **`os_signpost` + Instruments (Points of Interest)**: Correlate CPU phases in your allocator/scheduler (admit, prefill write, decode, retire) with GPU dispatches. This helps explain ramp/tail vs steady‑state.
* **`powermetrics`** (system‑level): Sanity‑check thermal throttling or DVFS during long runs (tokens/sec droop over minutes).
* **Command‑line Instruments (`xctrace`)**: Scriptable captures in CI to produce counter timelines and CSVs without opening the GUI.

There aren’t Nsight‑class third‑party profilers for Apple GPUs. The **Xcode toolchain is the authoritative source** for Metal‑specific performance data.

---

## Practical workflow we recommend

1. **Tight inner loop**: Add GPU timing to your C++ launcher and log per‑dispatch ms → see changes immediately when flipping `BLOCK_SIZE`, TG size, or load patterns.
2. **Once per day**: Capture a **Metal System Trace** during a steady‑state open‑loop run to check for pipeline churn or CPU submission gaps.
3. **Before committing a tile/tg change**: Take a **GPU Frame Capture** of a representative dispatch to confirm specialization constants, TG memory usage, and resource bindings.
4. **When performance regresses**: Use **GPU Counters** to see whether the shift is compute vs memory vs synchronization bound.

---

## TL;DR

* **Yes**: Xcode GPU Frame Capture, Metal System Trace, and GPU Counters are the right—and effectively the only—serious tools for **Metal** kernel performance.
* Combine them with **programmatic GPU timing** in your launcher and your **open‑loop throughput harness** to get both **per‑kernel truth** and **end‑to‑end tokens/sec**.
* Use the data to tune **BLOCK_SIZE**, **threadgroup size**, **vectorization**, and **double‑buffering**, and to confirm no **pipeline recompiles** or **CPU submission bubbles** are polluting your steady‑state.

