#!/usr/bin/env python3
"""
Benchmark script for MLX argpartition/partition operations.
Compares radix select implementation against full argsort.
"""

import time

import mlx.core as mx
import numpy as np

GREEN = "\033[92m"
YELLOW = "\033[33m"
RED = "\033[91m"
RESET = "\033[0m"


def color_speedup(speedup):
    s = f"{speedup:>5.2f}x"
    if 0.9 <= speedup <= 1.1:
        return f"{YELLOW}{s}{RESET}"
    elif speedup > 1.1:
        return f"{GREEN}{s}{RESET}"
    else:
        return f"{RED}{s}{RESET}"


def benchmark_argpartition(b, v, k, dtype=mx.bfloat16, warmup=5, iterations=100):
    x = mx.random.uniform(shape=(b, v)).astype(dtype)
    mx.eval(x)
    for _ in range(warmup):
        mx.eval(mx.argpartition(x, kth=k, axis=-1))
    start = time.perf_counter()
    for _ in range(iterations):
        mx.eval(mx.argpartition(x, kth=k, axis=-1))
    return (time.perf_counter() - start) / iterations * 1000


def benchmark_argsort(b, v, dtype=mx.bfloat16, warmup=5, iterations=100):
    x = mx.random.uniform(shape=(b, v)).astype(dtype)
    mx.eval(x)
    for _ in range(warmup):
        mx.eval(mx.argsort(x, axis=-1))
    start = time.perf_counter()
    for _ in range(iterations):
        mx.eval(mx.argsort(x, axis=-1))
    return (time.perf_counter() - start) / iterations * 1000


def verify_correctness(b, v, k):
    x = mx.random.uniform(shape=(b, v)).astype(mx.float32)
    mx.eval(x)
    indices = mx.argpartition(x, kth=k, axis=-1)
    mx.eval(indices)
    x_np = np.array(x)
    indices_np = np.array(indices)
    for i in range(b):
        pv = x_np[i, indices_np[i]]
        assert np.all(pv[:k] <= pv[k]), f"Row {i}: elements before k not all <= kth"
        assert np.all(pv[k + 1 :] >= pv[k]), f"Row {i}: elements after k not all >= kth"
    return True


def sweep_boundary(dtype=mx.bfloat16, k_ratio=0.004, warmup=10, iterations=50):
    dtype_name = str(dtype).split(".")[-1]
    print(f"\nDtype={dtype_name}  k=vocab*{k_ratio:.3f}")
    print()

    batch_sizes = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048]
    vocab_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    col_w = 10
    print(f"{'':>8}", end="")
    for v in vocab_sizes:
        label = f"v={v}"
        print(f"  {label:^{col_w}}", end="")
    print()

    for b in batch_sizes:
        print(f"b={b:<6}", end="")
        for v in vocab_sizes:
            k = max(1, int(v * k_ratio))
            try:
                x = mx.random.uniform(shape=(b, v)).astype(dtype)
                mx.eval(x)
                for _ in range(warmup):
                    mx.eval(mx.argpartition(x, kth=k, axis=-1))
                start = time.perf_counter()
                for _ in range(iterations):
                    mx.eval(mx.argpartition(x, kth=k, axis=-1))
                radix_ms = (time.perf_counter() - start) / iterations * 1000
                for _ in range(warmup):
                    mx.eval(mx.argsort(x, axis=-1))
                start = time.perf_counter()
                for _ in range(iterations):
                    mx.eval(mx.argsort(x, axis=-1))
                argsort_ms = (time.perf_counter() - start) / iterations * 1000

                speedup = argsort_ms / radix_ms
                cell = color_speedup(speedup)
                # pad accounting for invisible ANSI codes
                print(f"  {cell:^{col_w + len(GREEN) + len(RESET)}}", end="")
            except Exception:
                print(f"  {'ERR':^{col_w}}", end="")
        print()


def main():
    print("=" * 70)
    print("MLX Radix Select Benchmark")
    print("=" * 70)

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

    dtypes = [(mx.bfloat16, "bfloat16"), (mx.float32, "float32")]

    print("\n1. Correctness Verification")
    print("-" * 40)
    for b, v, k in configs:
        try:
            verify_correctness(b, v, k)
            print(f"  {GREEN}[PASS]{RESET} b={b}, v={v}, k={k}")
        except AssertionError as e:
            print(f"  {RED}[FAIL]{RESET} b={b}, v={v}, k={k}: {e}")

    print("\n2. Performance Benchmarks")
    print("-" * 70)

    for dtype, dtype_name in dtypes:
        print(f"\nDtype: {dtype_name}")
        print(
            f"{'Config':<25} {'ArgPartition':>14} {'ArgSort':>12} {'Speedup':>10}"
        )
        print("-" * 80)

        for b, v, k in configs:
            try:
                argpart_ms = benchmark_argpartition(b, v, k, dtype, warmup=3, iterations=50)
                argsort_ms = benchmark_argsort(
                    b, v, dtype, warmup=3, iterations=50
                )
                speedup = argsort_ms / argpart_ms
                config_str = f"b={b}, v={v}, k={k}"
                print(
                    f"{config_str:<25} {argpart_ms:>12.3f}ms"
                    f" {argsort_ms:>10.3f}ms  {color_speedup(speedup)}"
                )
            except Exception as e:
                print(f"b={b}, v={v}, k={k}: Error - {e}")

    print("\n3. Boundary Sweep")
    print("-" * 70)
    # sweep_boundary(mx.bool_)
    sweep_boundary(mx.bfloat16)
    # sweep_boundary(mx.float16)
    sweep_boundary(mx.float32)
    # sweep_boundary(mx.float64)
    # sweep_boundary(mx.int8)
    # sweep_boundary(mx.int16)
    # sweep_boundary(mx.int32)
    # sweep_boundary(mx.int64)
    # sweep_boundary(mx.uint8)
    # sweep_boundary(mx.uint16)
    # sweep_boundary(mx.uint32)
    # sweep_boundary(mx.uint64)

    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
