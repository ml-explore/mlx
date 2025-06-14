#!/usr/bin/env python3
"""
Benchmark script for SVD operations comparing CPU vs Metal performance.
This benchmark should be run before and after the Metal SVD implementation
to measure performance improvements.
"""

import time
from typing import Dict, List, Tuple

import mlx.core as mx
import numpy as np


def benchmark_svd_sizes() -> List[Tuple[int, int]]:
    """Return list of matrix sizes to benchmark."""
    return [
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 1024),
    ]


def create_test_matrix(m: int, n: int, dtype=mx.float32) -> mx.array:
    """Create a test matrix with known properties for SVD."""
    # Create a matrix with controlled singular values for consistent benchmarking
    np.random.seed(42)  # Fixed seed for reproducible results

    # Create matrix with known rank and condition number
    U = np.random.randn(m, min(m, n)).astype(np.float32)
    V = np.random.randn(min(m, n), n).astype(np.float32)

    # Create diagonal matrix with decreasing singular values
    s = np.logspace(0, -3, min(m, n)).astype(np.float32)
    S = np.diag(s)

    # Construct A = U @ S @ V
    if m >= n:
        A = U @ S @ V
    else:
        A = U @ S @ V[:m, :]

    return mx.array(A, dtype=dtype)


def benchmark_svd_operation(
    matrix: mx.array,
    compute_uv: bool = True,
    device: str = "gpu",
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
) -> Dict[str, float]:
    """Benchmark SVD operation with proper warmup and timing."""

    # Set device
    if device == "cpu":
        mx.set_default_device(mx.cpu)
    else:
        mx.set_default_device(mx.gpu)

    # Move matrix to target device
    matrix = mx.array(matrix, copy=True)

    # Warmup runs
    for _ in range(warmup_runs):
        if compute_uv:
            u, s, vt = mx.linalg.svd(matrix, compute_uv=True)
            mx.eval(u, s, vt)
        else:
            s = mx.linalg.svd(matrix, compute_uv=False)
            mx.eval(s)

    # Benchmark runs
    times = []
    for _ in range(benchmark_runs):
        start_time = time.perf_counter()

        if compute_uv:
            u, s, vt = mx.linalg.svd(matrix, compute_uv=True)
            mx.eval(u, s, vt)
        else:
            s = mx.linalg.svd(matrix, compute_uv=False)
            mx.eval(s)

        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
    }


def run_comprehensive_benchmark():
    """Run comprehensive SVD benchmark comparing CPU and GPU performance."""

    print("MLX SVD Performance Benchmark")
    print("=" * 50)
    print(f"Device: {mx.default_device()}")
    print(f"MLX Version: {mx.__version__ if hasattr(mx, '__version__') else 'Unknown'}")
    print()

    sizes = benchmark_svd_sizes()
    results = []

    # Test both singular values only and full SVD
    for compute_uv in [False, True]:
        mode = "Full SVD" if compute_uv else "Singular Values Only"
        print(f"\n{mode}")
        print("-" * 30)
        print(
            f"{'Size':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10} {'Status'}"
        )
        print("-" * 60)

        for m, n in sizes:
            matrix = create_test_matrix(m, n)

            try:
                # CPU benchmark
                cpu_stats = benchmark_svd_operation(matrix, compute_uv, "cpu")
                cpu_time = cpu_stats["mean_time"] * 1000  # Convert to ms

                # GPU benchmark
                try:
                    gpu_stats = benchmark_svd_operation(matrix, compute_uv, "gpu")
                    gpu_time = gpu_stats["mean_time"] * 1000  # Convert to ms
                    speedup = cpu_time / gpu_time
                    status = "✓"
                except Exception as e:
                    gpu_time = float("inf")
                    speedup = 0.0
                    status = f"✗ ({str(e)[:20]}...)"

                print(
                    f"{m}x{n:<8} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:<10.2f} {status}"
                )

                results.append(
                    {
                        "size": (m, n),
                        "compute_uv": compute_uv,
                        "cpu_time": cpu_time,
                        "gpu_time": gpu_time,
                        "speedup": speedup,
                        "status": status,
                    }
                )

            except Exception as e:
                print(
                    f"{m}x{n:<8} {'ERROR':<12} {'ERROR':<12} {'N/A':<10} ✗ {str(e)[:30]}..."
                )

    # Summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    successful_results = [r for r in results if r["speedup"] > 0]
    if successful_results:
        speedups = [r["speedup"] for r in successful_results]
        print(f"Average Speedup: {np.mean(speedups):.2f}x")
        print(f"Max Speedup: {np.max(speedups):.2f}x")
        print(f"Min Speedup: {np.min(speedups):.2f}x")
        print(f"Successful Tests: {len(successful_results)}/{len(results)}")
    else:
        print("No successful GPU tests completed.")

    return results


def benchmark_batch_processing():
    """Benchmark batch processing capabilities."""
    print("\n" + "=" * 50)
    print("BATCH PROCESSING BENCHMARK")
    print("=" * 50)

    matrix_size = (128, 128)
    batch_sizes = [1, 2, 4, 8, 16, 32]

    print(f"{'Batch Size':<12} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print("-" * 50)

    for batch_size in batch_sizes:
        # Create batch of matrices
        matrices = []
        for _ in range(batch_size):
            matrices.append(create_test_matrix(*matrix_size))

        batch_matrix = mx.stack(matrices, axis=0)

        try:
            cpu_stats = benchmark_svd_operation(
                batch_matrix, True, "cpu", warmup_runs=2, benchmark_runs=5
            )
            gpu_stats = benchmark_svd_operation(
                batch_matrix, True, "gpu", warmup_runs=2, benchmark_runs=5
            )

            cpu_time = cpu_stats["mean_time"] * 1000
            gpu_time = gpu_stats["mean_time"] * 1000
            speedup = cpu_time / gpu_time

            print(
                f"{batch_size:<12} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:<10.2f}"
            )

        except Exception as e:
            print(f"{batch_size:<12} {'ERROR':<12} {'ERROR':<12} {'N/A':<10}")


def verify_correctness():
    """Verify that GPU results match CPU results."""
    print("\n" + "=" * 50)
    print("CORRECTNESS VERIFICATION")
    print("=" * 50)

    test_sizes = [(64, 64), (128, 128), (100, 150)]

    for m, n in test_sizes:
        matrix = create_test_matrix(m, n)

        # CPU computation
        mx.set_default_device(mx.cpu)
        cpu_matrix = mx.array(matrix, copy=True)
        u_cpu, s_cpu, vt_cpu = mx.linalg.svd(cpu_matrix, compute_uv=True)
        mx.eval(u_cpu, s_cpu, vt_cpu)

        # GPU computation
        try:
            mx.set_default_device(mx.gpu)
            gpu_matrix = mx.array(matrix, copy=True)
            u_gpu, s_gpu, vt_gpu = mx.linalg.svd(gpu_matrix, compute_uv=True)
            mx.eval(u_gpu, s_gpu, vt_gpu)

            # Compare singular values (most important)
            s_diff = mx.abs(s_cpu - s_gpu)
            max_s_diff = mx.max(s_diff).item()

            # Reconstruction test
            reconstructed_cpu = u_cpu @ mx.diag(s_cpu) @ vt_cpu
            reconstructed_gpu = u_gpu @ mx.diag(s_gpu) @ vt_gpu

            recon_diff = mx.abs(cpu_matrix - reconstructed_cpu)
            max_recon_diff_cpu = mx.max(recon_diff).item()

            recon_diff = mx.abs(gpu_matrix - reconstructed_gpu)
            max_recon_diff_gpu = mx.max(recon_diff).item()

            print(f"Size {m}x{n}:")
            print(f"  Max singular value difference: {max_s_diff:.2e}")
            print(f"  Max reconstruction error (CPU): {max_recon_diff_cpu:.2e}")
            print(f"  Max reconstruction error (GPU): {max_recon_diff_gpu:.2e}")

            if max_s_diff < 1e-5 and max_recon_diff_gpu < 1e-5:
                print(f"  Status: ✓ PASS")
            else:
                print(f"  Status: ✗ FAIL")

        except Exception as e:
            print(f"Size {m}x{n}: ✗ ERROR - {str(e)}")


if __name__ == "__main__":
    print("Starting MLX SVD Benchmark...")
    print("This benchmark compares CPU vs GPU performance for SVD operations.")
    print("Run this before and after implementing Metal SVD to measure improvements.\n")

    # Run all benchmarks
    results = run_comprehensive_benchmark()
    benchmark_batch_processing()
    verify_correctness()

    print("\nBenchmark completed!")
    print("Save these results to compare with post-implementation performance.")
