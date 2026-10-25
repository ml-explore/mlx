# Copyright © 2026 Apple Inc.

import argparse

import mlx.core as mx
from time_utils import time_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark fused matrix multiplication and bias."
    )
    parser.add_argument(
        "--shape", type=int, nargs=3, default=(1, 4096, 4096), metavar=("M", "K", "N")
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--transpose-a", action="store_true")
    parser.add_argument("--transpose-b", action="store_true")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    m, k, n = args.shape
    dtype = getattr(mx, args.dtype)
    mx.random.seed(0)
    a = mx.random.normal((k, m) if args.transpose_a else (m, k)).astype(dtype)
    b = mx.random.normal((n, k) if args.transpose_b else (k, n)).astype(dtype)
    c = mx.random.normal((n,)).astype(dtype)
    a, b = (a.T if args.transpose_a else a), (b.T if args.transpose_b else b)
    mx.eval(a, b, c)

    def run():
        return [
            mx.addmm(c, a, b, args.alpha, args.beta) for _ in range(args.batch_size)
        ]

    time_fn(run, msg=f"{args.batch_size} addmm operations")
