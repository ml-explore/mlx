# Copyright © 2023-2024 Apple Inc.

"""
Compare nn.GRU and nn.LSTM speed: legacy (Python-only) vs fast (Metal).
Runs recurrent_bench.py twice (MLX_RNN_IMPL=legacy then =fast) and prints speedup.

Run from benchmarks/python:
  python compare_recurrent_speed.py
"""

import os
import subprocess
import sys


def run_bench(impl: str):
    """Run recurrent_bench.py with given MLX_RNN_IMPL; return (gru_ms, lstm_ms)."""
    env = {**os.environ, "MLX_RNN_IMPL": impl}
    result = subprocess.run(
        [sys.executable, "recurrent_bench.py"],
        env=env,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)) or ".",
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
    return gru_ms, lstm_ms


def main():
    print("Comparing nn.GRU / nn.LSTM: legacy (Python-only) vs fast (Metal)")
    print()

    gru_legacy, lstm_legacy = run_bench("legacy")
    gru_fast, lstm_fast = run_bench("fast")

    print("=" * 60)
    print("Full layer timings (batch=32, seq=40, input=64, hidden=200)")
    print("=" * 60)
    print(f"  {'Version':<10}  nn.GRU (ms)  nn.LSTM (ms)")
    print(f"  {'legacy':<10}  {gru_legacy:>10.3f}  {lstm_legacy:>10.3f}")
    print(f"  {'fast':<10}  {gru_fast:>10.3f}  {lstm_fast:>10.3f}")
    if not (gru_legacy != gru_legacy or gru_fast != gru_fast) and gru_fast > 0:
        print(f"  GRU speedup (fast vs legacy):  {gru_legacy / gru_fast:.2f}x")
    if not (lstm_legacy != lstm_legacy or lstm_fast != lstm_fast) and lstm_fast > 0:
        print(f"  LSTM speedup (fast vs legacy): {lstm_legacy / lstm_fast:.2f}x")
    print()
    print("To use legacy: export MLX_RNN_IMPL=legacy")
    print("See python/mlx/nn/layers/RECURRENT_VERSIONS.md for details.")


if __name__ == "__main__":
    main()
