#!/usr/bin/env python3
"""
Benchmark script for MLX argpartition/partition operations.
Compares radix select implementation against full sort.
"""

import time

import mlx.core as mx
import numpy as np


def benchmark_argpartition(b, v, k, dtype=mx.bfloat16, warmup=5, iterations=100):
    """Benchmark argpartition operation."""
    # Create random data
    x = mx.random.uniform(shape=(b, v)).astype(dtype)
    mx.eval(x)

    # Warmup
    for _ in range(warmup):
        result = mx.argpartition(x, kth=k, axis=-1)
        mx.eval(result)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = mx.argpartition(x, kth=k, axis=-1)
        mx.eval(result)
    end = time.perf_counter()

    avg_ms = (end - start) / iterations * 1000
    return avg_ms


def benchmark_partition(b, v, k, dtype=mx.bfloat16, warmup=5, iterations=100):
    """Benchmark partition operation."""
    # Create random data
    x = mx.random.uniform(shape=(b, v)).astype(dtype)
    mx.eval(x)

    # Warmup
    for _ in range(warmup):
        result = mx.partition(x, kth=k, axis=-1)
        mx.eval(result)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = mx.partition(x, kth=k, axis=-1)
        mx.eval(result)
    end = time.perf_counter()

    avg_ms = (end - start) / iterations * 1000
    return avg_ms


def benchmark_sort(b, v, dtype=mx.bfloat16, warmup=5, iterations=100):
    """Benchmark full sort operation for comparison."""
    # Create random data
    x = mx.random.uniform(shape=(b, v)).astype(dtype)
    mx.eval(x)

    # Warmup
    for _ in range(warmup):
        result = mx.sort(x, axis=-1)
        mx.eval(result)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        result = mx.sort(x, axis=-1)
        mx.eval(result)
    end = time.perf_counter()

    avg_ms = (end - start) / iterations * 1000
    return avg_ms


def verify_correctness(b, v, k, dtype=mx.float32):
    """Verify that argpartition produces correct results."""
    # Use float32 for verification since bfloat16 has numpy conversion issues
    x = mx.random.uniform(shape=(b, v)).astype(mx.float32)
    mx.eval(x)

    # Get argpartition result
    indices = mx.argpartition(x, kth=k, axis=-1)
    mx.eval(indices)

    # Convert to numpy for verification
    x_np = np.array(x)
    indices_np = np.array(indices)

    # Verify: for each row, the k-th element should be in its sorted position
    for i in range(b):
        # Get the values at the partitioned indices
        partitioned_values = x_np[i, indices_np[i]]

        # The k-th element should be the k-th smallest
        kth_value = partitioned_values[k]

        # All elements before k should be <= kth_value
        assert np.all(
            partitioned_values[:k] <= kth_value
        ), f"Row {i}: elements before k are not all <= kth"

        # All elements after k should be >= kth_value
        assert np.all(
            partitioned_values[k + 1 :] >= kth_value
        ), f"Row {i}: elements after k are not all >= kth"

    return True


def main():
    print("=" * 70)
    print("MLX Radix Select Benchmark")
    print("=" * 70)

    # Test configurations - including the problematic cases
    configs = [
        # (batch, vocab, k) - Standard cases
        (2048, 8192, 32),  # High batch, large vocab - radix should win
        (2048, 4096, 32),  # High batch, medium vocab - radix should win
        (1024, 4096, 16),
        (512, 2048, 64),
        (256, 1024, 32),
        (128, 512, 16),
        # Problematic cases - low batch, large vocab
        (1, 128000, 64),  # Single row, very large - sort should win
        (1, 512, 32),  # Single row, small - radix should win
        (16, 8192, 32),  # Few rows, large - sort should win
        (32, 8192, 32),  # Boundary case
        (64, 8192, 32),  # Above threshold - radix should win
    ]

    dtypes = [
        (mx.bfloat16, "bfloat16"),
        (mx.float32, "float32"),
    ]

    print("\n1. Correctness Verification")
    print("-" * 40)
    for b, v, k in [(2048, 4096, 32), (1, 128000, 64), (16, 8192, 32)]:
        try:
            verify_correctness(b, v, k)
            print(f"  [PASS] b={b}, v={v}, k={k}")
        except AssertionError as e:
            print(f"  [FAIL] b={b}, v={v}, k={k}: {e}")

    print("\n2. Performance Benchmarks")
    print("-" * 70)

    for dtype, dtype_name in dtypes:
        print(f"\nDtype: {dtype_name}")
        print(
            f"{'Config':<25} {'ArgPartition':<15} {'Partition':<15} {'Sort':<15} {'Speedup':<10}"
        )
        print("-" * 80)

        for b, v, k in configs:
            try:
                argpart_ms = benchmark_argpartition(
                    b, v, k, dtype, warmup=3, iterations=50
                )
                part_ms = benchmark_partition(b, v, k, dtype, warmup=3, iterations=50)
                sort_ms = benchmark_sort(b, v, dtype, warmup=3, iterations=50)
                speedup = sort_ms / argpart_ms

                config_str = f"b={b}, v={v}, k={k}"
                # Mark cases where we expect sort to be used
                note = ""
                if b <= 32 and v > 8192:
                    note = " (sort path)"
                print(
                    f"{config_str:<25} {argpart_ms:>12.3f}ms {part_ms:>12.3f}ms {sort_ms:>12.3f}ms {speedup:>8.2f}x{note}"
                )
            except Exception as e:
                print(f"b={b}, v={v}, k={k}: Error - {e}")

    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)
    print("\nNotes:")
    print("- Cases with b<=32 and v>8192 use sort (optimal for this workload)")
    print("- Cases with high batch count use radix select (optimal for parallelism)")
    print("- Speedup > 1.0 means partition is faster than sort")


if __name__ == "__main__":
    main()
