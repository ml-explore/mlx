#!/usr/bin/env python3
# Copyright © 2024 Apple Inc.

"""
Performance benchmark for MLX optimized collective communications.

This script benchmarks the performance of distributed operations with different
algorithms and data sizes to demonstrate speedups from optimizations.

Run with MPI:
    mpirun -n <num_procs> python benchmark_performance.py

Examples:
    # Run with 4 processes
    mpirun -n 4 python benchmark_performance.py

    # Run with specific algorithms
    mpirun -n 4 python benchmark_performance.py --algo default,ring,tree

    # Run with custom sizes
    mpirun -n 4 python benchmark_performance.py --sizes 1024,65536,262144
"""

import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Tuple

try:
    import mlx.core as mx
except ImportError:
    print("Error: MLX is not installed. Please install it first.")
    sys.exit(1)


def warmup_group():
    """Initialize distributed group and warmup."""
    try:
        world = mx.distributed.init()
        return world
    except Exception as e:
        print(f"Warning: Distributed not available: {e}")
        return None


def time_function(fn, num_warmup=3, num_iters=20):
    """
    Time a function with warmup and multiple iterations.

    Args:
        fn: Function to time
        num_warmup: Number of warmup iterations
        num_iters: Number of benchmark iterations

    Returns:
        tuple: (mean_time_ms, std_time_ms, min_time_ms, max_time_ms)
    """
    # Warmup
    for _ in range(num_warmup):
        result = fn()
        mx.eval(result)

    # Benchmark
    times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        result = fn()
        mx.eval(result)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    mean_time = sum(times) / len(times)
    variance = sum((t - mean_time) ** 2 for t in times) / len(times)
    std_time = variance**0.5
    min_time = min(times)
    max_time = max(times)

    return mean_time, std_time, min_time, max_time


def benchmark_all_reduce(
    sizes: List[int],
    algorithms: List[str] = ["default", "linear", "ring", "recursive_doubling", "tree"],
    op: str = "sum",
    warmup_group_=bool,
) -> Dict:
    """
    Benchmark all-reduce with different algorithms and sizes.

    Args:
        sizes: List of array sizes in elements
        algorithms: List of algorithm names to benchmark
        op: Reduction operation ("sum", "max", "min")

    Returns:
        Dict with benchmark results
    """
    world = warmup_group_ if isinstance(warmup_group_, mx.distributed.Group) else None

    results = []

    for algo in algorithms:
        print(f"  Benchmarking all_reduce with {algo} algorithm...")

        for size in sizes:
            # Create array
            x = mx.random.normal(shape=(size,), dtype=mx.float32)

            def run_all_reduce():
                return mx.distributed.all_reduce_opt(x, op=op, algo=algo)

            try:
                mean_time, std_time, min_time, max_time = time_function(run_all_reduce)

                # Calculate metrics
                size_bytes = size * 4  # float32 = 4 bytes
                if world and world.size() > 1:
                    # Calculate bandwidth with actual process count
                    num_processes = world.size()
                    total_bytes = size_bytes * num_processes
                    bandwidth_gbps = (total_bytes / 1e9) / (mean_time / 1000)
                    ops_per_sec = 1000 / mean_time
                else:
                    # Single process - no real communication
                    bandwidth_gbps = 0.0
                    ops_per_sec = 1000 / mean_time
                    num_processes = 1

                result_entry = {
                    "algorithm": algo,
                    "size_elements": size,
                    "size_bytes": size_bytes,
                    "num_processes": num_processes,
                    "latency_ms": round(mean_time, 6),
                    "std_ms": round(std_time, 6),
                    "min_ms": round(min_time, 6),
                    "max_ms": round(max_time, 6),
                    "bandwidth_gbps": (
                        round(bandwidth_gbps, 4) if bandwidth_gbps > 0 else None
                    ),
                    "ops_per_sec": round(ops_per_sec, 2),
                }
                results.append(result_entry)

                # Print progress (rank 0 only)
                if world is None or world.rank() == 0:
                    size_mb = size_bytes / (1024 * 1024)
                    if bandwidth_gbps:
                        print(
                            f"    Size {size_mb:.4f} MB: {mean_time:.3f} ms, "
                            f"{bandwidth_gbps:.2f} GB/s"
                        )
                    else:
                        print(f"    Size {size_mb:.4f} MB: {mean_time:.3f} ms")

            except Exception as e:
                if world is None or world.rank() == 0:
                    print(f"    Error with size {size} using {algo}: {e}")

    return {"operation": "all_reduce", "reduction_op": op, "results": results}


def benchmark_all_gather(
    sizes: List[int],
    algorithms: List[str] = ["default", "linear", "ring", "recursive_doubling", "tree"],
    warmup_group_: Optional[mx.distributed.Group] = None,
) -> Dict:
    """
    Benchmark all-gather with different algorithms and sizes.

    Args:
        sizes: List of array sizes in elements
        algorithms: List of algorithm names to benchmark

    Returns:
        Dict with benchmark results
    """
    world = warmup_group_ if isinstance(warmup_group_, mx.distributed.Group) else None

    results = []

    for algo in algorithms:
        print(f"  Benchmarking all_gather with {algo} algorithm...")

        for size in sizes:
            x = mx.random.normal(shape=(size,), dtype=mx.float32)

            def run_all_gather():
                return mx.distributed.all_gather_opt(x, algo=algo)

            try:
                mean_time, std_time, min_time, max_time = time_function(run_all_gather)

                size_bytes = size * 4
                if world and world.size() > 1:
                    num_processes = world.size()
                    # All-gather: each rank has size, total is size * num_processes
                    total_bytes = size_bytes * num_processes
                    bandwidth_gbps = (total_bytes / 1e9) / (mean_time / 1000)
                    ops_per_sec = 1000 / mean_time
                else:
                    bandwidth_gbps = 0.0
                    ops_per_sec = 1000 / mean_time
                    num_processes = 1

                result_entry = {
                    "algorithm": algo,
                    "size_elements": size,
                    "size_bytes": size_bytes,
                    "num_processes": num_processes,
                    "latency_ms": round(mean_time, 6),
                    "std_ms": round(std_time, 6),
                    "min_ms": round(min_time, 6),
                    "max_ms": round(max_time, 6),
                    "bandwidth_gbps": (
                        round(bandwidth_gbps, 4) if bandwidth_gbps > 0 else None
                    ),
                    "ops_per_sec": round(ops_per_sec, 2),
                }
                results.append(result_entry)

                if world is None or world.rank() == 0:
                    size_mb = size_bytes / (1024 * 1024)
                    if bandwidth_gbps:
                        print(
                            f"    Size {size_mb:.4f} MB: {mean_time:.3f} ms, "
                            f"{bandwidth_gbps:.2f} GB/s"
                        )
                    else:
                        print(f"    Size {size_mb:.4f} MB: {mean_time:.3f} ms")

            except Exception as e:
                if world is None or world.rank() == 0:
                    print(f"    Error with size {size} using {algo}: {e}")

    return {"operation": "all_gather", "results": results}


def benchmark_reduce_scatter(
    sizes: List[int],
    algorithms: List[str] = ["default", "linear", "ring", "recursive_doubling", "tree"],
    op: str = "sum",
    warmup_group_: Optional[mx.distributed.Group] = None,
) -> Dict:
    """
    Benchmark reduce-scatter with different algorithms and sizes.

    Args:
        sizes: List of array sizes in elements (will be adjusted to be divisible)
        algorithms: List of algorithm names to benchmark
        op: Reduction operation

    Returns:
        Dict with benchmark results
    """
    world = warmup_group_ if isinstance(warmup_group_, mx.distributed.Group) else None
    num_processes = world.size() if world else 1

    results = []

    for algo in algorithms:
        print(f"  Benchmarking reduce_scatter with {algo} algorithm...")

        for size in sizes:
            # Ensure size is divisible by num_processes
            if num_processes > 1 and size % num_processes != 0:
                adjusted_size = ((size // num_processes) + 1) * num_processes
            else:
                adjusted_size = size

            x = mx.random.normal(shape=(adjusted_size,), dtype=mx.float32)

            def run_reduce_scatter():
                return mx.distributed.reduce_scatter_opt(x, op=op, algo=algo)

            try:
                mean_time, std_time, min_time, max_time = time_function(
                    run_reduce_scatter
                )

                size_bytes = adjusted_size * 4
                if num_processes > 1:
                    # Reduce-scatter: each rank receives size/num_processes
                    send_bytes = size_bytes // num_processes
                    total_bytes = send_bytes * num_processes
                    bandwidth_gbps = (total_bytes / 1e9) / (mean_time / 1000)
                    ops_per_sec = 1000 / mean_time
                else:
                    bandwidth_gbps = 0.0
                    ops_per_sec = 1000 / mean_time

                result_entry = {
                    "algorithm": algo,
                    "size_elements": adjusted_size,
                    "original_size_elements": size,
                    "size_bytes": size_bytes,
                    "num_processes": num_processes,
                    "latency_ms": round(mean_time, 6),
                    "std_ms": round(std_time, 6),
                    "min_ms": round(min_time, 6),
                    "max_ms": round(max_time, 6),
                    "bandwidth_gbps": (
                        round(bandwidth_gbps, 4) if bandwidth_gbps > 0 else None
                    ),
                    "ops_per_sec": round(ops_per_sec, 2),
                }
                results.append(result_entry)

                if world is None or world.rank() == 0:
                    size_mb = size_bytes / (1024 * 1024)
                    if bandwidth_gbps:
                        print(
                            f"    Size {size_mb:.4f} MB: {mean_time:.3f} ms, "
                            f"{bandwidth_gbps:.2f} GB/s"
                        )
                    else:
                        print(f"    Size {size_mb:.4f} MB: {mean_time:.3f} ms")

            except Exception as e:
                if world is None or world.rank() == 0:
                    print(f"    Error with size {size} using {algo}: {e}")

    return {"operation": "reduce_scatter", "reduction_op": op, "results": results}


def benchmark_pipeline(
    num_stages: int = 4,
    sizes: List[int] = [1024, 65536],
    warmup_group_: Optional[mx.distributed.Group] = None,
) -> Dict:
    """
    Benchmark pipeline parallelism with different sizes.

    Args:
        num_stages: Number of pipeline stages
        sizes: List of input sizes

    Returns:
        Dict with benchmark results
    """
    world = warmup_group_ if isinstance(warmup_group_, mx.distributed.Group) else None
    num_processes = world.size() if world else 1

    results = []

    print(f"  Benchmarking pipeline with {num_stages} stages...")

    for size in sizes:
        input_array = mx.random.normal(shape=(size,), dtype=mx.float32)

        # Create pipeline stages
        def stage_fn(i):
            def fn(x):
                # Simulate computation
                for _ in range(3):
                    x = mx.sin(x)
                return x

            return fn

        stages = [
            mx.distributed.PipelineStage(i, num_stages, stage_fn(i))
            for i in range(num_stages)
        ]

        def run_pipeline():
            return mx.distributed.execute_pipeline(stages, input_array)

        try:
            mean_time, std_time, min_time, max_time = time_function(run_pipeline)

            size_bytes = size * 4
            ops_per_sec = 1000 / mean_time

            result_entry = {
                "num_stages": num_stages,
                "size_elements": size,
                "size_bytes": size_bytes,
                "num_processes": num_processes,
                "latency_ms": round(mean_time, 6),
                "std_ms": round(std_time, 6),
                "min_ms": round(min_time, 6),
                "max_ms": round(max_time, 6),
                "bandwidth_gbps": None,
                "ops_per_sec": round(ops_per_sec, 2),
            }
            results.append(result_entry)

            if world is None or world.rank() == 0:
                print(
                    f"    Size {size/1024:.2f} K: {mean_time:.3f} ms, "
                    f"{ops_per_sec:.1f} ops/s"
                )

        except Exception as e:
            if world is None or world.rank() == 0:
                print(f"    Error with size {size}: {e}")

    return {"operation": "pipeline", "num_stages": num_stages, "results": results}


def run_full_benchmark(
    sizes: Optional[List[int]] = None,
    algorithms: Optional[List[str]] = None,
    warmup_group_: Optional[mx.distributed.Group] = None,
) -> Dict:
    """
    Run the full benchmark suite.

    Args:
        sizes: List of sizes to benchmark
        algorithms: List of algorithms to test

    Returns:
        Dict with all benchmark results
    """
    world = warmup_group_ if isinstance(warmup_group_, mx.distributed.Group) else None
    num_processes = world.size() if world else 1

    # Default sizes (in elements)
    if sizes is None:
        sizes = [1024, 65536, 262144, 1048576]  # 4KB, 256KB, 1MB, 4MB

    # Default algorithms
    if algorithms is None:
        algorithms = ["default", "linear", "ring", "recursive_doubling", "tree"]

    if world and world.rank() == 0:
        print("=" * 80)
        print("MLX OPTIMIZED COLLECTIVES PERFORMANCE BENCHMARK")
        print("=" * 80)
        print(f"Number of processes: {num_processes}")
        print(f"Test sizes (elements): {sizes}")
        print(f"Algorithms: {algorithms}")

    all_results = {}

    # Run benchmarks
    if world and world.rank() == 0:
        print("\n" + "=" * 80)
        print("1. ALL-REDUCE BENCHMARK")
        print("=" * 80)

    all_results["all_reduce_sum"] = benchmark_all_reduce(
        sizes=sizes, algorithms=algorithms, op="sum", warmup_group_=world
    )

    if world and world.rank() == 0:
        print("\n" + "=" * 80)
        print("2. ALL-GATHER BENCHMARK")
        print("=" * 80)

    all_results["all_gather"] = benchmark_all_gather(
        sizes=sizes, algorithms=algorithms, warmup_group_=world
    )

    if world and world.rank() == 0:
        print("\n" + "=" * 80)
        print("3. REDUCE-SCATTER BENCHMARK")
        print("=" * 80)

    all_results["reduce_scatter_sum"] = benchmark_reduce_scatter(
        sizes=sizes, algorithms=algorithms, op="sum", warmup_group_=world
    )

    # Pipeline benchmarks (smaller sizes only)
    pipeline_sizes = [1024, 65536]

    if world and world.rank() == 0:
        print("\n" + "=" * 80)
        print("4. PIPELINE PARALLELISM BENCHMARK")
        print("=" * 80)

    all_results["pipeline"] = benchmark_pipeline(
        num_stages=4, sizes=pipeline_sizes, warmup_group_=world
    )

    # Add summary statistics
    if world is None or world.rank() == 0:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

    return all_results


def print_summary(results: Dict, world=None):
    """Print formatted summary of benchmark results."""
    if world and world.rank() != 0:
        return

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Group by operation
    ops = ["all_reduce_sum", "all_gather", "reduce_scatter_sum", "pipeline"]

    for op in ops:
        if op not in results:
            continue

        op_results = results[op]

        print(f"\n{op.upper().replace('_', ' ')}")
        print("-" * 80)

        # Print header
        if op == "pipeline":
            print(
                f"{'Stages':<10} {'Size (KB)':>12} {'Latency (ms)':>14} {'Ops/sec':>12}"
            )
        else:
            print(
                f"{'Algorithm':<20} {'Size (MB)':>12} {'Latency (ms)':>14} "
                f"{'Bandwidth (GB/s)':>16} {'Ops/sec':>12}"
            )

        print("-" * 80)

        # Sort by size
        sorted_results = sorted(
            op_results["results"],
            key=lambda x: (x.get("size_elements", 0), x.get("algorithm", "")),
        )

        for r in sorted_results:
            if op == "pipeline":
                size_kb = r["size_elements"] / 1024
                print(
                    f"{r['num_stages']:<10} {size_kb:>12.2f} "
                    f"{r['latency_ms']:>14.3f} {r['ops_per_sec']:>12.0f}"
                )
            else:
                size_mb = r["size_elements"] * 4 / (1024 * 1024)
                bandwidth = r.get("bandwidth_gbps")
                if bandwidth:
                    print(
                        f"{r['algorithm']:<20} {size_mb:>12.4f} "
                        f"{r['latency_ms']:>14.3f} {bandwidth:>16.2f} "
                        f"{r['ops_per_sec']:>12.0f}"
                    )
                else:
                    print(
                        f"{r['algorithm']:<20} {size_mb:>12.4f} "
                        f"{r['latency_ms']:>14.3f} {'N/A':>16} "
                        f"{r['ops_per_sec']:>12.0f}"
                    )

        # Find best algorithm for each size
        if op != "pipeline" and len(op_results["results"]) > 0:
            print("\nBest algorithms by size:")

            sizes = {}
            for r in op_results["results"]:
                size_key = r["size_elements"]
                if (
                    size_key not in sizes
                    or r["latency_ms"] < sizes[size_key]["latency_ms"]
                ):
                    sizes[size_key] = r

            for size_key, best in sorted(sizes.items()):
                speedup = 1.0
                if "ring" in str(op_results["results"]) and best["algorithm"] != "ring":
                    # Calculate speedup over ring
                    for r in op_results["results"]:
                        if r["size_elements"] == size_key and r["algorithm"] == "ring":
                            speedup = r["latency_ms"] / best["latency_ms"]
                            break

                size_mb = size_key * 4 / (1024 * 1024)
                print(
                    f"  Size {size_mb:.4f} MB: {best['algorithm']} ({speedup:.2f}x speedup)"
                )


def save_results(results: Dict, filename: str = "benchmark_performance.json"):
    """Save benchmark results to JSON file."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="MLX Optimized Collectives Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark suite with default settings
  mpirun -n 4 python benchmark_performance.py
  
  # Run with specific algorithms
  mpirun -n 2 python benchmark_performance.py --algo default,ring,tree
  
  # Run with custom sizes
  mpirun -n 4 python benchmark_performance.py --sizes 1024,65536,262144
  
  # Save results to JSON
  mpirun -n 4 python benchmark_performance.py --output my_benchmark.json
  
  # Run single operation
  mpirun -n 4 python benchmark_performance.py --op all_reduce
  
Note:
  For distributed benchmarks, run with mpirun.
  Example: mpirun -n 4 python benchmark_performance.py
        """,
    )

    parser.add_argument(
        "--op",
        "-o",
        type=str,
        default=None,
        help="Operation to benchmark: all_reduce, all_gather, reduce_scatter, pipeline, or None for full suite",
    )

    parser.add_argument(
        "--algo",
        "-a",
        type=str,
        default="default,ring,tree",
        help="Algorithms to benchmark (comma-separated): default,linear,ring,recursive_doubling,tree",
    )

    parser.add_argument(
        "--sizes",
        "-s",
        type=str,
        default="1024,65536,262144,1048576",
        help="Sizes to benchmark (comma-separated in elements)",
    )

    parser.add_argument(
        "--output",
        "-O",
        type=str,
        default="benchmark_performance.json",
        help="Output file for JSON results",
    )

    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print results without running benchmarks",
    )

    args = parser.parse_args()

    # Parse algorithms
    algorithms = [a.strip() for a in args.algo.split(",")]

    # Parse sizes
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    # Initialize distributed
    world = warmup_group()
    num_processes = world.size() if world else 1

    # Check if we're running with MPI
    if num_processes == 1:
        print(
            "Warning: Running in single-process mode. Performance numbers may not reflect real distributed performance."
        )

    # Run benchmarks
    if args.op:
        # Single operation benchmark
        if args.op == "all_reduce":
            results = {
                "all_reduce": benchmark_all_reduce(
                    sizes=sizes, algorithms=algorithms, warmup_group_=world
                )
            }
        elif args.op == "all_gather":
            results = {
                "all_gather": benchmark_all_gather(
                    sizes=sizes, algorithms=algorithms, warmup_group_=world
                )
            }
        elif args.op == "reduce_scatter":
            results = {
                "reduce_scatter": benchmark_reduce_scatter(
                    sizes=sizes, algorithms=algorithms, warmup_group_=world
                )
            }
        elif args.op == "pipeline":
            results = {"pipeline": benchmark_pipeline(sizes=sizes, warmup_group_=world)}
        else:
            print(f"Unknown operation: {args.op}")
            parser.print_help()
            return
    else:
        # Full benchmark suite
        results = run_full_benchmark(
            sizes=sizes, algorithms=algorithms, warmup_group_=world
        )

    # Print summary
    print_summary(results, world)

    # Save to JSON
    save_results(results, args.output)


if __name__ == "__main__":
    main()
