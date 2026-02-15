# Copyright © 2023-2024 Apple Inc.

"""
Benchmark nn.GRU and nn.LSTM: legacy (Python-only) vs fast (Metal) implementation.

Run with device and implementation:
  MLX_RNN_IMPL=legacy python recurrent_bench.py [--device gpu|cpu]
  MLX_RNN_IMPL=fast   python recurrent_bench.py [--device gpu|cpu]

Matches backup: h0/c0 set so fast path from step 1, batch=32, seq=40, input=128, hidden=200.
"""

import argparse
import os

import mlx.core as mx
import mlx.nn as nn
from time_utils import measure_runtime


def main():
    p = argparse.ArgumentParser(
        description="Benchmark nn.GRU / nn.LSTM (legacy vs fast)"
    )
    p.add_argument(
        "--device", choices=("gpu", "cpu"), default="gpu", help="Device to run on"
    )
    args = p.parse_args()

    impl = os.environ.get("MLX_RNN_IMPL", "fast").strip().lower()
    if impl not in ("legacy", "fast", "fast_v2"):
        impl = "fast"

    device = mx.gpu if args.device == "gpu" else mx.cpu
    mx.set_default_device(device)

    # Match backup benchmark_attentivefp_and_gru.py --layers and RECURRENT_VERSIONS.md
    batch, seq_len, input_size, hidden_size = 32, 40, 128, 200
    mx.random.seed(0)

    gru = nn.GRU(input_size, hidden_size, bias=True)
    lstm = nn.LSTM(input_size, hidden_size, bias=True)
    x = mx.random.normal(shape=(batch, seq_len, input_size)).astype(mx.float32)
    h0 = mx.zeros((batch, hidden_size), dtype=mx.float32)
    c0 = mx.zeros((batch, hidden_size), dtype=mx.float32)
    mx.eval(x, h0, c0)

    def gru_forward():
        out = gru(x, h0)
        mx.eval(out)
        return out

    def lstm_forward():
        out, _ = lstm(x, h0, c0)
        mx.eval(out)
        return out

    gru_ms = measure_runtime(gru_forward)
    lstm_ms = measure_runtime(lstm_forward)

    print(f"MLX_RNN_IMPL={impl} device={args.device}")
    print(
        f"  nn.GRU({batch}, {seq_len}, {input_size} -> {hidden_size}): {gru_ms:.3f} ms"
    )
    print(
        f"  nn.LSTM({batch}, {seq_len}, {input_size} -> {hidden_size}): {lstm_ms:.3f} ms"
    )


if __name__ == "__main__":
    main()
