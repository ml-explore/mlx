# Copyright © 2023-2024 Apple Inc.

"""
Compare nn.GRU and nn.LSTM speed: legacy vs fast, on GPU and CPU.
Runs recurrent_bench.py for each (impl, device) and prints tables + speedups.

Run from benchmarks/python:
  python compare_recurrent_speed.py
"""

import os
import subprocess
import sys


def run_bench(impl: str, device: str, num_runs: int = 3):
    """Run recurrent_bench.py num_runs times; return median (gru_ms, lstm_ms) for stability."""
    env = {**os.environ, "MLX_RNN_IMPL": impl}
    cwd = os.path.dirname(os.path.abspath(__file__)) or "."
    gru_list, lstm_list = [], []

    for _ in range(num_runs):
        result = subprocess.run(
            [sys.executable, "recurrent_bench.py", "--device", device],
            env=env,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=120,
        )
        if result.returncode != 0:
            print(result.stderr or result.stdout, file=sys.stderr)
            return float("nan"), float("nan")

        gru_ms = lstm_ms = float("nan")
        for line in result.stdout.splitlines():
            if "nn.GRU(" in line:
                try:
                    gru_ms = float(line.split(":")[-1].strip().replace(" ms", ""))
                except ValueError:
                    pass
            if "nn.LSTM(" in line:
                try:
                    lstm_ms = float(line.split(":")[-1].strip().replace(" ms", ""))
                except ValueError:
                    pass
        if gru_ms == gru_ms:
            gru_list.append(gru_ms)
        if lstm_ms == lstm_ms:
            lstm_list.append(lstm_ms)

    gru_med = float("nan") if not gru_list else sorted(gru_list)[len(gru_list) // 2]
    lstm_med = float("nan") if not lstm_list else sorted(lstm_list)[len(lstm_list) // 2]
    return gru_med, lstm_med


def main():
    print("Comparing nn.GRU / nn.LSTM: legacy (Python-only) vs fast (Metal)")
    print("(batch=32, seq=40, input=128, hidden=200, h0/c0 set)")
    print(
        "(median of 3 runs per config; each run = median of 5×100 iters for stability)"
    )
    print()

    gru_legacy_gpu, lstm_legacy_gpu = run_bench("legacy", "gpu")
    gru_fast_gpu, lstm_fast_gpu = run_bench("fast", "gpu")
    gru_legacy_cpu, lstm_legacy_cpu = run_bench("legacy", "cpu")
    gru_fast_cpu, lstm_fast_cpu = run_bench("fast", "cpu")

    def fmt(v):
        return f"{v:.3f}" if v == v and v != float("inf") else "N/A"

    print("=" * 70)
    print("Full layer timings (ms)")
    print("=" * 70)
    print(
        f"  {'Layer':<12}  {'legacy GPU':>12}  {'fast GPU':>12}  {'legacy CPU':>12}  {'fast CPU':>12}"
    )
    print(
        f"  {'nn.GRU':<12}  {fmt(gru_legacy_gpu):>12}  {fmt(gru_fast_gpu):>12}  {fmt(gru_legacy_cpu):>12}  {fmt(gru_fast_cpu):>12}"
    )
    print(
        f"  {'nn.LSTM':<12}  {fmt(lstm_legacy_gpu):>12}  {fmt(lstm_fast_gpu):>12}  {fmt(lstm_legacy_cpu):>12}  {fmt(lstm_fast_cpu):>12}"
    )
    print()

    print("Speedup (fast vs legacy) on same device:")
    if gru_fast_gpu > 0 and not (gru_legacy_gpu != gru_legacy_gpu):
        print(f"  GPU  GRU:  {gru_legacy_gpu / gru_fast_gpu:.2f}x")
    if lstm_fast_gpu > 0 and not (lstm_legacy_gpu != lstm_legacy_gpu):
        print(f"  GPU  LSTM: {lstm_legacy_gpu / lstm_fast_gpu:.2f}x")
    if gru_fast_cpu > 0 and not (gru_legacy_cpu != gru_legacy_cpu):
        print(f"  CPU  GRU:  {gru_legacy_cpu / gru_fast_cpu:.2f}x")
    if lstm_fast_cpu > 0 and not (lstm_legacy_cpu != lstm_legacy_cpu):
        print(f"  CPU  LSTM: {lstm_legacy_cpu / lstm_fast_cpu:.2f}x")
    if gru_fast_cpu != gru_fast_cpu or lstm_fast_cpu != lstm_fast_cpu:
        print(
            "  (fast on CPU: Metal kernels are GPU-only; fast path falls back or N/A)"
        )
    print()

    print(
        "Legacy: Python-only. Fast: Metal kernels on GPU; on CPU fast uses fallback (may be N/A)."
    )
    print("To use legacy: export MLX_RNN_IMPL=legacy")
    print("See python/mlx/nn/layers/RECURRENT_VERSIONS.md for details.")


if __name__ == "__main__":
    main()
