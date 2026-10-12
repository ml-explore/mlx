# Copyright © 2026 Apple Inc.

"""Standalone M==1 fp16/bf16 matvec microbenchmark.

Measures ``y[1, N] = x[1, K] @ mat[N, K].T`` for representative dense
projection dimensions.

Reports the median amortized microseconds per matvec across timed rounds.
Point ``PYTHONPATH`` at the importable package for the build being measured:

    PYTHONPATH=/path/to/package python gemv_m1_bench.py --out parent.json

Use gemv_m1_compare.py for sequential alternating parent/PR measurements.
"""

import argparse
import json
import statistics
import sys
import time

import mlx.core as mx

# (out_features N, in_features K) matvec shapes spanning the decode path:
# square hidden, packed QKV, FFN up/down, and vocab-head widths. All K % 4 == 0
# and contiguous (so the vec4-alignment predicate in gemv_wide_config holds).
SHAPES = [
    (3072, 3072),  # O-proj / square hidden
    (4096, 3072),  # QKV-ish fan-out
    (6144, 3072),  # packed QKV
    (8192, 3072),  # wide projection
    (3072, 8192),  # FFN down-proj
    (4608, 3584),  # from the PR candidate note
    (32768, 3072),  # vocab-head slice
    (65536, 3072),  # vocab_wide threshold
]


def bench_one(N, K, dtype, rounds, iters, warmup):
    # Cycle distinct matrices to reduce cache reuse. For the default shapes
    # and iteration count the pool exceeds 192 MiB. This approximates weight
    # streaming; cache residency and DRAM traffic are not directly measured.
    bytes_per = N * K * (2 if dtype in (mx.float16, mx.bfloat16) else 4)
    pool = max(8, (192 * 1024 * 1024) // bytes_per + 1)
    pool = min(pool, iters)
    mats = [mx.random.normal((N, K)).astype(dtype) for _ in range(pool)]
    x = mx.random.normal((1, K)).astype(dtype)
    mx.eval(*mats, x)

    # Chain `iters` M==1 matvecs into one lazy graph and time a single eval, so
    # Python graph construction is outside the timer. Evaluation includes
    # scheduling, rescale/reduction operations, and synchronization. The
    # per-iter input rescale keeps every matvec distinct (no CSE collapse).
    def one_round():
        xs = x
        acc = mx.zeros((1,), dtype)
        for i in range(iters):
            xs = xs * (1.0 + 1e-4 * (i % 7))
            y = xs @ mats[i % pool].T
            acc = acc + y[:, :1]
        return acc

    # warm the pipeline / kernel cache
    for _ in range(warmup):
        mx.eval(one_round())
    mx.synchronize()

    round_times = []
    for _ in range(rounds):
        g = one_round()
        t0 = time.perf_counter()
        mx.eval(g)
        mx.synchronize()
        round_times.append((time.perf_counter() - t0) * 1e6 / iters)  # us/call
    return statistics.median(round_times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--rounds", type=int, default=11)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dtype = getattr(mx, args.dtype)
    info = mx.device_info()
    print(
        f"mlx {mx.__version__} | {info.get('device_name')} "
        f"({info.get('architecture')}) | {args.dtype} | "
        f"median of {args.rounds} round averages, {args.iters} iters/round",
        file=sys.stderr,
    )

    results = []
    for N, K in SHAPES:
        us = bench_one(N, K, dtype, args.rounds, args.iters, args.warmup)
        results.append({"N": N, "K": K, "us": us})
        print(f"  N={N:>6} K={K:>5}  {us:8.1f} us", file=sys.stderr)

    payload = {
        "mlx_version": mx.__version__,
        "device": info.get("device_name"),
        "architecture": info.get("architecture"),
        "dtype": args.dtype,
        "rounds": args.rounds,
        "iters": args.iters,
        "results": results,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
