# Copyright © 2023-2024 Apple Inc.

"""
Benchmark nn.GRU and nn.LSTM: legacy (Python-only) vs fast (Metal) implementation.

Run twice and compare times:
  MLX_RNN_IMPL=legacy python recurrent_bench.py
  MLX_RNN_IMPL=fast   python recurrent_bench.py

Implementation is chosen at import time via MLX_RNN_IMPL.
"""

import os

import mlx.core as mx
import mlx.nn as nn
from time_utils import measure_runtime


def main():
    impl = os.environ.get("MLX_RNN_IMPL", "fast").strip().lower()
    if impl not in ("legacy", "fast", "fast_v2"):
        impl = "fast"

    # Match RECURRENT_VERSIONS.md: seq=40, hidden=200, batch=32
    batch, seq_len, input_size, hidden_size = 32, 40, 64, 200
    mx.random.seed(0)

    gru = nn.GRU(input_size, hidden_size)
    lstm = nn.LSTM(input_size, hidden_size)
    x = mx.random.normal(shape=(batch, seq_len, input_size))
    mx.eval(x)

    def gru_forward():
        out = gru(x)
        mx.eval(out)
        return out

    def lstm_forward():
        out, _ = lstm(x)
        mx.eval(out)
        return out

    gru_ms = measure_runtime(gru_forward)
    lstm_ms = measure_runtime(lstm_forward)

    print(f"MLX_RNN_IMPL={impl}")
    print(f"  nn.GRU({batch}, {seq_len}, {input_size} -> {hidden_size}): {gru_ms:.3f} ms")
    print(f"  nn.LSTM({batch}, {seq_len}, {input_size} -> {hidden_size}): {lstm_ms:.3f} ms")


if __name__ == "__main__":
    main()
