# Copyright © 2023 Apple Inc.

import argparse
import time

import mlx.core as mx
from time_utils import time_fn


def time_svd_square():
    """Benchmark SVD on square matrices of various sizes."""
    print("Benchmarking SVD on square matrices...")

    sizes = [64, 128, 256, 512]

    for size in sizes:
        print(f"\n--- {size}x{size} matrix ---")

        # Create random matrix
        a = mx.random.normal(shape=(size, size))
        mx.eval(a)

        # Benchmark singular values only
        print(f"SVD (values only):")
        time_fn(lambda x: mx.linalg.svd(x, compute_uv=False), a)

        # Benchmark full SVD
        print(f"SVD (full decomposition):")
        time_fn(lambda x: mx.linalg.svd(x, compute_uv=True), a)


def time_svd_rectangular():
    """Benchmark SVD on rectangular matrices."""
    print("\nBenchmarking SVD on rectangular matrices...")

    shapes = [(128, 64), (64, 128), (256, 128), (128, 256)]

    for m, n in shapes:
        print(f"\n--- {m}x{n} matrix ---")

        # Create random matrix
        a = mx.random.normal(shape=(m, n))
        mx.eval(a)

        # Benchmark full SVD
        print(f"SVD (full decomposition):")
        time_fn(lambda x: mx.linalg.svd(x, compute_uv=True), a)


def time_svd_batch():
    """Benchmark SVD on batched matrices."""
    print("\nBenchmarking SVD on batched matrices...")

    batch_configs = [
        (4, 64, 64),
        (8, 32, 32),
        (16, 16, 16),
    ]

    for batch_size, m, n in batch_configs:
        print(f"\n--- Batch of {batch_size} {m}x{n} matrices ---")

        # Create batch of random matrices
        a = mx.random.normal(shape=(batch_size, m, n))
        mx.eval(a)

        # Benchmark full SVD
        print(f"Batched SVD (full decomposition):")
        time_fn(lambda x: mx.linalg.svd(x, compute_uv=True), a)


def compare_cpu_gpu():
    """Compare CPU vs GPU performance for SVD."""
    print("\nComparing CPU vs GPU performance...")

    sizes = [64, 128, 256]

    for size in sizes:
        print(f"\n--- {size}x{size} matrix comparison ---")

        # Create random matrix
        a_cpu = mx.random.normal(shape=(size, size))
        mx.set_default_device(mx.cpu)
        mx.eval(a_cpu)

        a_gpu = mx.array(a_cpu)
        mx.set_default_device(mx.gpu)
        mx.eval(a_gpu)

        # Time CPU SVD
        mx.set_default_device(mx.cpu)
        print("CPU SVD:")
        start_time = time.time()
        u_cpu, s_cpu, vt_cpu = mx.linalg.svd(a_cpu, compute_uv=True)
        mx.eval(u_cpu, s_cpu, vt_cpu)
        cpu_time = time.time() - start_time

        # Time GPU SVD
        mx.set_default_device(mx.gpu)
        print("GPU SVD:")
        start_time = time.time()
        u_gpu, s_gpu, vt_gpu = mx.linalg.svd(a_gpu, compute_uv=True)
        mx.eval(u_gpu, s_gpu, vt_gpu)
        gpu_time = time.time() - start_time

        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Verify results are close
        mx.set_default_device(mx.cpu)
        s_cpu_sorted = mx.sort(s_cpu)
        mx.set_default_device(mx.gpu)
        s_gpu_sorted = mx.sort(s_gpu)
        mx.eval(s_cpu_sorted, s_gpu_sorted)

        # Convert to CPU for comparison
        mx.set_default_device(mx.cpu)
        s_gpu_cpu = mx.array(s_gpu_sorted)
        mx.eval(s_gpu_cpu)

        diff = mx.max(mx.abs(s_cpu_sorted - s_gpu_cpu))
        mx.eval(diff)
        print(f"Max singular value difference: {diff.item():.2e}")


def time_svd_special_matrices():
    """Benchmark SVD on special matrices (identity, diagonal, etc.)."""
    print("\nBenchmarking SVD on special matrices...")

    size = 256

    # Identity matrix
    print(f"\n--- {size}x{size} identity matrix ---")
    identity = mx.eye(size)
    mx.eval(identity)
    time_fn(lambda x: mx.linalg.svd(x, compute_uv=True), identity)

    # Diagonal matrix
    print(f"\n--- {size}x{size} diagonal matrix ---")
    diag_vals = mx.random.uniform(shape=(size,))
    diagonal = mx.diag(diag_vals)
    mx.eval(diagonal)
    time_fn(lambda x: mx.linalg.svd(x, compute_uv=True), diagonal)

    # Zero matrix
    print(f"\n--- {size}x{size} zero matrix ---")
    zero_matrix = mx.zeros((size, size))
    mx.eval(zero_matrix)
    time_fn(lambda x: mx.linalg.svd(x, compute_uv=True), zero_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLX SVD benchmarks.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    parser.add_argument(
        "--compare", action="store_true", help="Compare CPU vs GPU performance."
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmarks.")
    args = parser.parse_args()

    if args.gpu:
        mx.set_default_device(mx.gpu)
        print("Using GPU (Metal) backend")
    else:
        mx.set_default_device(mx.cpu)
        print("Using CPU backend")

    if args.compare:
        compare_cpu_gpu()
    elif args.all:
        time_svd_square()
        time_svd_rectangular()
        time_svd_batch()
        time_svd_special_matrices()
        if mx.metal.is_available():
            compare_cpu_gpu()
    else:
        time_svd_square()
        if args.gpu and mx.metal.is_available():
            time_svd_rectangular()
            time_svd_batch()
