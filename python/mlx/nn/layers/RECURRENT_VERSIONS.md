# GRU and LSTM: Versioned Implementations and Audit

This document locks versions of the GRU/LSTM improvements and explains how to revert or compare.

## Environment: `MLX_RNN_IMPL`

Set before importing `mlx.nn` to choose the implementation:

| Value     | Behavior |
|----------|----------|
| `legacy` | **Previous (Python-only)** – no Metal fast cells; pure Python ops per step. Use to compare or revert. |
| `fast`   | **Current default** – use `mx.fast.gru_cell` / `mx.fast.lstm_cell` when available and initial hidden/cell are set. |

Example (revert to previous behavior):

```bash
export MLX_RNN_IMPL=legacy
python your_script.py
```

Example (equality test – legacy vs fast outputs must match):

```bash
cd python && python -m pytest tests/test_recurrent_fast_vs_legacy.py -v
```

Example (speed benchmark – run twice and compare times):

```bash
cd benchmarks/python
MLX_RNN_IMPL=legacy python recurrent_bench.py
MLX_RNN_IMPL=fast   python recurrent_bench.py
```

---

## Audit: Previous (legacy) vs fast_cell

### GRU

**Previous (legacy) – Python-only path**

- One matmul at start: `x = addmm(b, x, Wx.T)` → shape `(..., L, 3*H)`.
- Per step:
  - Slice `x_rz`, `x_n` from `x`.
  - `h_proj = hidden @ Wh.T` (recurrent matmul).
  - Add `bhn` to `h_proj_n` only; split and add to `rz`.
  - `rz = sigmoid(rz)`; split into `r`, `z`.
  - `n = tanh(x_n + r * h_proj_n)`.
  - `hidden = (1 - z) * n + z * hidden`.
- Many small ops and kernel launches per step.

**Fast path (when `MLX_RNN_IMPL` is `fast` and `mx.fast.gru_cell` exists)**

- Same initial matmul; precompute `bhn_padded` once (zeros for r,z, `bhn` for n) to avoid per-step concat.
- Per step:
  - `input_proj = x[..., idx, :]`, `h_proj = hidden @ Wh.T + bhn_padded`.
  - `hidden = mx.fast.gru_cell(input_proj, h_proj, hidden)` (single Metal kernel: gates + update).
- Fewer kernel launches; fused gating and update on Metal.

**Speed comparison (measured: seq=40, hidden=200, batch=32)**  
- Full layer legacy: nn.GRU ~0.31 ms, nn.LSTM ~0.18 ms.  
- Full layer fast: nn.GRU ~0.08 ms, nn.LSTM ~0.08 ms.  
- **Speedup: GRU ~3.9x, LSTM ~2.25x** with fast_cell vs legacy.  
- Cell-level: `mx.fast.gru_cell` ~2x faster than Python GRUCell (e.g. 0.05 ms vs 0.10 ms for 64×200, 20 steps).

---

### LSTM

**Previous (legacy) – Python-only path**

- One matmul at start: `x = addmm(bias, x, Wx.T)` → shape `(..., L, 4*H)`.
- Per step:
  - `input_proj = x[..., idx, :]`; `ifgo = input_proj + hidden @ Wh.T`.
  - Split into `i, f, g, o`; sigmoid/tanh.
  - `cell = f*cell + i*g`; `hidden = o*tanh(cell)`.
- Many small ops per step.

**Fast path (when `MLX_RNN_IMPL` is `fast` or `fast_v2` and `mx.fast.lstm_cell` exists)**

- Same initial matmul.
- Per step:
  - `input_proj = x[..., idx, :]`, `hidden_proj = hidden @ Wh.T`.
  - `cell, hidden = mx.fast.lstm_cell(input_proj, hidden_proj, cell, hidden)` (single Metal kernel).
- Fused gates and state update on Metal.

**Speed comparison (measured: seq=40, hidden=200, batch=32)**  
- Full layer legacy: nn.LSTM ~0.18 ms.  
- Full layer fast: nn.LSTM ~0.08 ms (**~2.25x speedup**).  
- Cell-level: `mx.fast.lstm_cell` ~1.8x faster than Python LSTMCell.

---

## Progress and version lock

| Version  | GRU full layer      | LSTM full layer     | Revert |
|----------|--------------------|---------------------|--------|
| legacy   | ~0.31 ms (Python)  | ~0.18 ms (Python)   | `MLX_RNN_IMPL=legacy` |
| fast     | ~0.08 ms (Metal)   | ~0.08 ms (Metal)    | default; no env |

**Caution:** To “go back in time”, set `MLX_RNN_IMPL=legacy` before any `import mlx.nn`. The implementation is chosen at import time.

---

## Possible future improvements (fast_v2 or later)

1. **Precomputed input projection**  
   Input projection `W_ih @ x` is already done once for the full sequence at layer entry. No extra precompute needed for current layout.

2. **Batched recurrent matmul**  
   `h_proj = hidden @ Wh.T` cannot be batched across steps (hidden changes each step). A fused “multi-step” kernel could be added later and gated under `fast_v2`.

3. **float16 / mixed precision**  
   Run recurrent layer in float16 where acceptable for additional speed.

4. **Larger threadgroups / tuning**  
   Further tuning of `fast_gru_cell` / `fast_lstm_cell` Metal kernels for typical batch/hidden sizes.
