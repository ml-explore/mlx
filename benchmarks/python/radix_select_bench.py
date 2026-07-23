#!/usr/bin/env python3
"""Benchmark and verify MLX argpartition/partition (radix select)."""

import argparse
import time

import mlx.core as mx
import numpy as np


def _resolve_dtype(name):
    dt = getattr(mx, name, None) or getattr(mx, name + "_", None)
    if dt is None or not isinstance(dt, mx.Dtype):
        raise ValueError(f"Unknown dtype: {name}")
    return dt


# Must match partition.cu dispatch constants.
MAX_RADIX_ITEMS_PER_THREAD = 64
RADIX_SIZE = 32
WARP_SIZE = 32
SMEM_BUDGET = 48 * 1024

BLOCK_THRESHOLDS = [(256, 32), (512, 64), (1024, 128)]
ITEMS_BUCKETS = (1, 2, 4, 8, 12, 16, 24, 64)


def _dtype_name(dtype):
    return str(dtype).split(".")[-1]


def _key_bytes(dtype):
    return dtype.size


def _block_threads(axis_size):
    for threshold, threads in BLOCK_THRESHOLDS:
        if axis_size <= threshold:
            return threads
    return 256


def _items_per_thread(axis_size, block_threads):
    needed = (axis_size + block_threads - 1) // block_threads
    for b in ITEMS_BUCKETS:
        if needed <= b:
            return b
    return None


def _smem_bytes(key_size, block_threads, items_per_thread):
    tile = block_threads * items_per_thread
    warps = block_threads // WARP_SIZE
    return tile * key_size + RADIX_SIZE * 4 + (2 + 3 * warps + 6) * 4


def max_small_kernel_axis(dtype):
    """Largest axis size the radix small kernel can handle for dtype."""
    ks = _key_bytes(dtype)
    best = 0
    for v in range(1, 256 * MAX_RADIX_ITEMS_PER_THREAD + 1):
        bt = _block_threads(v)
        ipt = _items_per_thread(v, bt)
        if ipt is None or ipt > MAX_RADIX_ITEMS_PER_THREAD:
            continue
        if _smem_bytes(ks, bt, ipt) <= SMEM_BUDGET:
            best = v
    return best


def parse_dtypes(s):
    return [_resolve_dtype(n.strip().lower()) for n in s.split(",") if n.strip()]


def verify_correctness(b, v, k, dtype=mx.float32):
    x = mx.floor(mx.random.uniform(shape=(b, v)) * 257.0).astype(dtype)
    mx.eval(x)

    indices = mx.argpartition(x, kth=k, axis=-1)
    mx.eval(indices)

    x_np = np.array(x.astype(mx.float32)) if dtype == mx.bfloat16 else np.array(x)
    idx = np.array(indices)
    is_float = np.issubdtype(x_np.dtype, np.floating)

    for i in range(b):
        row, ri = x_np[i], idx[i]
        assert np.unique(ri).size == v, f"Row {i}: not a permutation"

        pv = row[ri]
        pivot, left, right = pv[k], pv[:k], pv[k + 1 :]

        if is_float and np.isnan(pivot):
            assert np.all(np.isnan(pv[k:])), f"Row {i}: non-NaN after NaN pivot"
            continue

        if is_float:
            assert np.all(
                (~np.isnan(left)) & (left <= pivot)
            ), f"Row {i}: left violation"
            assert np.all(
                np.isnan(right) | (right >= pivot)
            ), f"Row {i}: right violation"
        else:
            assert np.all(left <= pivot), f"Row {i}: left violation"
            assert np.all(right >= pivot), f"Row {i}: right violation"

        less = np.count_nonzero(row < pivot)
        leq = np.count_nonzero(row <= pivot)
        assert less <= k < leq, f"Row {i}: rank inconsistent"


def verify_determinism(b, v, k, dtype=mx.float32):
    x = mx.zeros((b, v), dtype=dtype)
    mx.eval(x)

    outputs = []
    for _ in range(8):
        idx = mx.argpartition(x, kth=k, axis=-1)
        mx.eval(idx)
        outputs.append(np.array(idx))

    assert len({o.tobytes() for o in outputs}) == 1, "Non-deterministic tie ordering"

    expected = mx.argsort(x, axis=-1)
    mx.eval(expected)
    assert np.array_equal(
        outputs[0], np.array(expected)
    ), "Tie order differs from argsort"


def _bench(x, fn, warmup=10, iters=50):
    for _ in range(warmup):
        mx.eval(fn(x))
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(x))
    return (time.perf_counter() - t0) / iters * 1000


def sweep(dtype, k_ratio=0.004, warmup=10, iters=50, verify=False):
    limit = max_small_kernel_axis(dtype)
    name = _dtype_name(dtype)

    vocabs = [
        32,
        64,
        96,
        160,
        256,
        384,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
    ]
    batches = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    vocabs = [v for v in vocabs if v <= limit]

    if not vocabs:
        print(f"No vocab sizes in range for {name} (limit={limit}).")
        return

    print(f"\n**{name}** k=v*{k_ratio:.3f}  small-kernel-limit={limit}\n")
    print("| batch |", " | ".join(f"v={v}" for v in vocabs), "|")
    print("|------:|", " | ".join("---:" for _ in vocabs), "|")

    for b in batches:
        cells = []
        for v in vocabs:
            k = max(1, int(v * k_ratio))
            try:
                x = mx.random.uniform(shape=(b, v)).astype(dtype)
                mx.eval(x)
                ap = _bench(
                    x, lambda a: mx.argpartition(a, kth=k, axis=-1), warmup, iters
                )
                ar = _bench(x, lambda a: mx.argsort(a, axis=-1), warmup, iters)
                if verify:
                    verify_correctness(b, v, k, dtype)
                    verify_determinism(b, v, k, dtype)
                cells.append(f"{ar / ap:.2f}x")
            except Exception:
                cells.append("ERR")
        print(f"| {b} |", " | ".join(cells), "|")


def main():
    p = argparse.ArgumentParser(description="Benchmark MLX radix select")
    p.add_argument("--verify", action="store_true", help="Run correctness checks")
    p.add_argument(
        "--sweep", action="store_true", help="Sweep batch x vocab for small kernel"
    )
    p.add_argument(
        "--dtypes",
        default="bfloat16,float32",
        help="Comma-separated dtypes (default: bfloat16,float32)",
    )
    args = p.parse_args()
    dtypes = parse_dtypes(args.dtypes)

    configs = [
        (2048, 8192, 32),
        (2048, 4096, 32),
        (1024, 4096, 16),
        (512, 2048, 64),
        (256, 1024, 32),
        (128, 512, 16),
        (1, 128000, 64),
        (1, 512, 32),
        (16, 8192, 32),
        (32, 8192, 32),
        (64, 8192, 32),
    ]

    if args.verify:
        print("# Correctness\n")
        for dtype in dtypes:
            name = _dtype_name(dtype)
            for b, v, k in configs:
                try:
                    verify_correctness(b, v, k, dtype)
                    verify_determinism(b, v, k, dtype)
                    print(f"  PASS  b={b} v={v} k={k} {name}")
                except AssertionError as e:
                    print(f"  FAIL  b={b} v={v} k={k} {name}: {e}")

    if args.sweep:
        print("# Sweep (speedup vs argsort)\n")
        for dtype in dtypes:
            sweep(dtype, verify=args.verify)

    if not args.verify and not args.sweep:
        print("# Benchmark (speedup vs argsort)\n")
        for dtype in dtypes:
            name = _dtype_name(dtype)
            print(f"\n**{name}**\n")
            print("| config | argpartition | argsort | speedup |")
            print("|--------|------------:|--------:|--------:|")
            for b, v, k in configs:
                try:
                    x = mx.random.uniform(shape=(b, v)).astype(dtype)
                    mx.eval(x)
                    ap = _bench(x, lambda a: mx.argpartition(a, kth=k, axis=-1), 3, 50)
                    ar = _bench(x, lambda a: mx.argsort(a, axis=-1), 3, 50)
                    print(
                        f"| b={b} v={v} k={k} | {ap:.3f}ms | {ar:.3f}ms | {ar/ap:.2f}x |"
                    )
                except Exception as e:
                    print(f"| b={b} v={v} k={k} | ERR | ERR | {e} |")


if __name__ == "__main__":
    main()
