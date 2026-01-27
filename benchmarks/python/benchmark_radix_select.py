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
                # Dynamic threshold logic:
                # 1. Small arrays: merge sort (radix overhead too high)
                # 2. Large arrays with low batch: merge sort (can't saturate GPU)
                type_bits = 16 if dtype == mx.bfloat16 else 32
                num_passes = (type_bits + 7) // 8
                min_size_for_radix = 1024 * num_passes

                elements_per_thread = (v + 255) // 256
                work_per_thread = elements_per_thread * (num_passes + 2)
                active_threads = b * 256

                uses_sort = (v < min_size_for_radix) or (
                    work_per_thread > 64 and active_threads < 8192
                )
                note = " (sort path)" if uses_sort else ""
                print(
                    f"{config_str:<25} {argpart_ms:>12.3f}ms {part_ms:>12.3f}ms {sort_ms:>12.3f}ms {speedup:>8.2f}x{note}"
                )
            except Exception as e:
                print(f"b={b}, v={v}, k={k}: Error - {e}")

    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)
    print("\nNotes:")
    print("- Algorithm selection is dynamic based on workload characteristics:")
    print(
        "  - Small arrays (< 1024 * num_passes): merge sort (radix overhead too high)"
    )
    print("  - Large arrays with low batch: merge sort (can't saturate GPU)")
    print("  - Otherwise: radix select")
    print("- Speedup > 1.0 means partition is faster than sort")


if __name__ == "__main__":
    main()
