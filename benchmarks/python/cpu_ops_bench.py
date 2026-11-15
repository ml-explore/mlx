# Copyright © 2026 Apple Inc.
"""Micro-benchmarks for hot CPU-backend ops.

Each op is evaluated `--warmup` times, then timed for `--rounds` rounds of
`--iters` evaluations. Reports the median per-call time across rounds and the
spread (min..max of the per-round means).

Usage:
    python benchmarks/python/cpu_ops_bench.py [--filter substr] [--json out.json]
"""

import argparse
import json
import statistics
import time

import mlx.core as mx


def make_cases():
    mx.random.seed(0)
    k = mx.random.key(0)
    big = mx.random.normal((4096, 4096), key=k)
    med = mx.random.normal((1024, 1024), key=k)
    small = mx.random.normal((256, 256), key=k)
    big2 = mx.random.normal((4096, 4096), key=mx.random.key(1))
    med2 = mx.random.normal((1024, 1024), key=mx.random.key(1))
    row = mx.random.normal((4096,), key=k)
    w = mx.ones((4096,))
    b = mx.zeros((4096,))
    batt = mx.random.normal((8, 32, 512, 512), key=k)  # attention scores
    big_bf16 = big.astype(mx.bfloat16)
    cond = big > 0
    table = mx.random.normal((32000, 1024), key=k)
    idx = mx.random.randint(0, 32000, (4096,), key=k)
    sort_in = mx.random.normal((1024, 4096), key=k)
    sort_small = mx.random.normal((64, 1024), key=k)
    mx.eval(
        big,
        med,
        small,
        big2,
        med2,
        row,
        w,
        b,
        batt,
        big_bf16,
        cond,
        table,
        idx,
        sort_in,
        sort_small,
    )

    cases = [
        ("add f32 256x256", lambda: small + small),
        ("add f32 1024x1024", lambda: med + med2),
        ("add f32 4096x4096", lambda: big + big2),
        ("add bcast row 4096x4096", lambda: big + row),
        ("mul scalar 4096x4096", lambda: big * 2.0),
        ("exp f32 1024x1024", lambda: mx.exp(med)),
        ("exp f32 4096x4096", lambda: mx.exp(big)),
        ("tanh f32 4096x4096", lambda: mx.tanh(big)),
        ("where 4096x4096", lambda: mx.where(cond, big, big2)),
        ("astype f32->bf16 4096x4096", lambda: big.astype(mx.bfloat16)),
        ("sum all 4096x4096", lambda: mx.sum(big)),
        ("sum axis=-1 4096x4096", lambda: mx.sum(big, axis=-1)),
        ("sum axis=0 4096x4096", lambda: mx.sum(big, axis=0)),
        ("max axis=-1 4096x4096", lambda: mx.max(big, axis=-1)),
        ("mean axis=-1 1024x1024", lambda: mx.mean(med, axis=-1)),
        ("softmax f32 4096x4096", lambda: mx.softmax(big, axis=-1)),
        ("softmax f32 8x32x512x512", lambda: mx.softmax(batt, axis=-1)),
        ("softmax bf16 4096x4096", lambda: mx.softmax(big_bf16, axis=-1)),
        ("rms_norm 4096x4096", lambda: mx.fast.rms_norm(big, w, 1e-5)),
        ("layer_norm 4096x4096", lambda: mx.fast.layer_norm(big, w, b, 1e-5)),
        ("transpose copy 4096x4096", lambda: mx.contiguous(big.T)),
        ("transpose copy 1024x1024", lambda: mx.contiguous(med.T)),
        ("swapaxes copy 8x32x512x512", lambda: mx.contiguous(mx.swapaxes(batt, 1, 2))),
        ("gather rows 4096 of 32000x1024", lambda: table[idx]),
        ("sort axis=-1 1024x4096", lambda: mx.sort(sort_in, axis=-1)),
        ("argsort axis=-1 1024x4096", lambda: mx.argsort(sort_in, axis=-1)),
        ("sort axis=-1 64x1024", lambda: mx.sort(sort_small, axis=-1)),
    ]
    return cases


def bench(fn, warmup, rounds, iters):
    for _ in range(warmup):
        mx.eval(fn())
    means = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            mx.eval(fn())
        means.append((time.perf_counter() - t0) / iters * 1e3)
    return statistics.median(means), min(means), max(means)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--filter", default="")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--json", default="")
    args = p.parse_args()
    mx.set_default_device(mx.cpu)
    results = {}
    for name, fn in make_cases():
        if args.filter and args.filter not in name:
            continue
        med, lo, hi = bench(fn, args.warmup, args.rounds, args.iters)
        results[name] = {"median_ms": med, "min_ms": lo, "max_ms": hi}
        print(f"{name:34s} {med:9.3f} ms  [{lo:.3f} .. {hi:.3f}]", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
