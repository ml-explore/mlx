# ROCm mlx investigation — are there confirmed issues?

**Date:** 2026-07-19  
**Repo:** [NripeshN/mlx](https://github.com/NripeshN/mlx) branch **`rocm-support`**  
**Local:** `/home/antmi/mlx`  
**mlx SHA:** `0dadb703d77301af29405cf7e12627efb88a6d0f`  
**Consumer:** lemon-mlx-engine (FetchContent same SHA), model `LemonMLXE/Qwen3.6-35B-A3B-MTP-mlx-4bit`, **gfx1150**

---

## Executive verdict

| Question | Answer |
|----------|--------|
| Is there a **confirmed mlx ROCm failure mode** on this machine? | **YES — dual large-model process load → SIGSEGV (exit 139)** |
| Is there a confirmed **single-process correctness** bug blocking eager decode? | **NO** (exclusive loads pass; engine product path green) |
| Should we open a **code PR** that “fixes GDN/copy” without more isolation? | **NO** |
| Should we open a **GitHub issue** (robustness / OOM→SEGV)? | **OPTIONAL / YES if documenting** — frame as dual-process overcommit SEGV, not “copy kernel math wrong” |
| Must product wait on mlx tip? | **NO** for eager single-process; ops rule: one process |

---

## Candidate inventory (from engine loops + this pack)

| ID | Symptom | Where stack dies | Confirmed? | Ownership |
|----|---------|------------------|------------|-----------|
| **M1** | 2nd process load SIGSEGV after MTP-skip | `hipLaunchKernel` ← mlx `copy_contiguous` / bf16→f32 | **YES** (this pack dual A/B) | mlx ROCm **or** HIP driver OOM path + allocator; **triggered by ops dual-load** |
| **M2** | Exclusive chat/server load | — | **PASS** 2/2 chat exclusive | — |
| **M3** | MTP `There is no Stream(cpu, N)` | `mlx/backend/cpu/encoder.cpp` | **Reproduced earlier in engine with `--use-mtp`**; **not re-tested here** (MTP deferred) | mlx stream TLS / engine MTP thread hygiene — **open if MTP productized** |
| **M4** | Pure-graph / capture desync | rocm `device.cpp` capture/arena | **Not exercised** (eager only) | deferred |
| **M5** | Engine ChatSession / stop / thinking | — | Engine-owned; fixed elsewhere | **not mlx** |

---

## Exclusive vs dual-load matrix (this run)

**Harness:** engine binaries linked against this mlx SHA (same `0dadb703`).  
**Env:** `MLX_DECODE_GRAPH_PURE_OFF=1`, no `MLX_LOAD_MTP_HEAD`, no quant fuse.

| Run | Result | Log |
|-----|--------|-----|
| exclusive chat #1 | **PASS** Model loaded | `logs/excl-chat-1.log` |
| exclusive chat #2 | **PASS** Model loaded | `logs/excl-chat-2.log` |
| server up (holds ~18 GB) | **PASS** health | `logs/dual-server.log` |
| chat load **while server holds model** | **FAIL_SEGV rc=139** after MTP-skip line | `logs/dual-chat.log` |

**Interpretation:** Failure correlates with **second ~18 GB model** on unified-memory APU (~8.5 GiB VRAM report), not with exclusive first-forward GDN. Stack class matches historical gdb (copy → HIP), but root is **resource overcommit / ungraceful failure**, not proven single-stream kernel logic bug.

---

## Code map (mlx tree)

| Component | Path |
|-----------|------|
| Contiguous copy kernels | `mlx/backend/rocm/copy/copy_contiguous.hip` |
| Copy dispatch | `mlx/backend/rocm/copy.hip` |
| CPU stream missing | `mlx/backend/cpu/encoder.cpp` (“There is no Stream(cpu, {})…”) |
| ROCm eval / graph decode | `mlx/backend/rocm/eval.cpp`, `device.cpp` |

Historical engine gdb (not re-run here):  
`Qwen35MoEGatedDeltaNet` → `eval` → `copy_contiguous` → `hipLaunchKernel` in `libamdhip64`.

---

## Decision matrix (for maintainers)

| Action | Do it? | Condition |
|--------|--------|-----------|
| Document dual-load SEGV in **this repo** | **Yes** | This RESULTS.md |
| Open **issue** on NripeshN/mlx | **Optional** | Title like: *ROCm gfx1150: SIGSEGV on second concurrent large model load (copy_contiguous / hipLaunchKernel)* — include dual A/B, pins, “exclusive OK” |
| Open **fix PR** without repro isolation | **No** | No minimal pure-mlx unit repro yet; dual-load is multi-process |
| Blame product “gibberish” on mlx FA kernels | **No** | Loop6 engine gate: stop/C1/thinking floor pass exclusive |
| Pursue MTP Stream(cpu) | **Later** | Only when product enables MTP |

---

## Recommended issue body (if filing)

```text
Title: [ROCm] SIGSEGV (hipLaunchKernel) when loading a second large model while first process holds ~18GB (gfx1150)

mlx: 0dadb703 (rocm-support)
GPU: gfx1150 AMD Radeon 890M
Repro:
  1) start process A: load Qwen3.6-35B-A3B-MTP-mlx-4bit (lemon-mlx-engine server)
  2) while A healthy, process B: same model via chat CLI
  3) B dies exit 139 after device bind / early load (copy_contiguous path)
Control: exclusive single process loads succeed repeatedly.
Ask: should allocator/HIP return OOM cleanly instead of SEGV?
```

---

## Product / ops (not mlx code)

On this APU: **never** run two full 35B MLX processes. Engine docs (loop6) already state this.

---

## Supervisors (quintuple)

| Role | Verdict |
|------|---------|
| Decode/ROCm | Dual-load SEGV confirmed; exclusive path clean |
| QA | A/B exclusive vs dual is required evidence |
| Ownership | M1 = mlx/HIP robustness + ops; M5 = engine; M3 MTP deferred |
| Product | Unblocked single-process; do not block on mlx PR |
| Clear Thought | Falsified “exclusive chat always broken”; confirmed dual-load |

*End of investigation pack. Next loop: optional issue draft PR or pure-mlx microbench for OOM handling.*
