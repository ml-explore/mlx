#!/usr/bin/env python3
# Copyright © 2024 Apple Inc.

"""
Comprehensive benchmark script for MLX optimized collective communications.

This benchmark tests:
1. All-reduce operations with different algorithms
2. All-gather operations with different algorithms
3. Reduce-scatter operations with different algorithms
4. Pipeline parallelism overhead
5. Communication-computation overlap

Run with:
    mpirun -n <num_processes> python optimized_collectives_bench.py
    # or
    mpirun -n <num_processes> python -m mlx.benchmarks.python.optimized_collectives_bench

Usage examples:
    # Test all-reduce with different algorithms and sizes
    python optimized_collectives_bench.py --op all_reduce --algo default

    # Compare algorithms for all_gather
    python optimized_collectives_bench.py --op all_gather --algo ring,tree

    # Run full benchmark suite
    python optimized_collectives_bench.py --full

State-of-the-art metrics:
- Bandwidth: GB/s
- Latency: ms
- Throughput: operations/sec
- Scalability: speedup vs single process
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mlx.core as mx


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    op: str
    algo: str
    size: int  # in bytes
    num_processes: int
    latency_ms: float
    bandwidth_gbps: float
    operations_per_sec: float
    warmup_ms: float = 0.0
    stderr_ms: float = 0.0


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark runs."""

    results: List[BenchmarkResult] = field(default_factory=list)
    config: Dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "config": self.config,
            "results": [
                {
                    "op": r.op,
                    "algo": r.algo,
                    "size_bytes": r.size,
                    "num_processes": r.num_processes,
                    "latency_ms": r.latency_ms,
                    "bandwidth_gbps": r.bandwidth_gbps,
                    "ops_per_sec": r.operations_per_sec,
                }
                for r in self.results
            ],
        }

    def to_json(self, filename: str = "benchmark_results.json"):
        """Save results to JSON file."""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filename

    def print_summary(self):
        """Print formatted summary."""
        if not self.results:
            print("No benchmark results to display.")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS")
        print("=" * 80)

        # Group by operation
        ops = {}
        for r in self.results:
            if r.op not in ops:
                ops[r.op] = []
            ops[r.op].append(r)

        for op, results in ops.items():
            print(f"\n{op.upper()}")
            print("-" * 80)

            # Print header
            print(
                f"{'Algorithm':<20} {'Size (MB)':>12} {'Latency (ms)':>14} "
                f"{'Bandwidth (GB/s)':>18} {'Ops/sec':>14}"
            )
            print("-" * 80)

            # Sort by size
            results.sort(key=lambda x: x.size)

            for r in results:
                size_mb = r.size / (1024 * 1024)
                print(
                    f"{r.algo:<20} {size_mb:>12.2f} {r.latency_ms:>14.5f} "
                    f"{r.bandwidth_gbps:>18.2f} {r.operations_per_sec:>14.0f}"
                )

        print("\n" + "=" * 80)


def get_group_size():
    """Get the number of processes in the distributed group."""
    try:
        group = mx.distributed.init()
        return group.size()
    except Exception as e:
        print(f"Warning: Could not initialize distributed group: {e}")
        return 1


def time_function(fn, num_warmup=5, num_iters=20, name=None):
    """
    Time a function with warmup and multiple iterations.

    Returns:
        tuple: (mean_time_ms, std_time_ms, min_time_ms, max_time_ms)
    """
    if name:
        msg = f"  Benchmarking {name}..."
    else:
        msg = f"  Benchmarking {fn.__name__}..."

    # Warmup
    warmup_times = []
    for i in range(num_warmup):
        start = time.perf_counter()
        result = fn()
        mx.eval(result)
        end = time.perf_counter()
        warmup_times.append((end - start) * 1000)

    # Benchmark
    times = []
    for i in range(num_iters):
        start = time.perf_counter()
        result = fn()
        mx.eval(result)
        end = time.perf_counter()
        times.append((end - start) * 1000)

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
    num_processes: int = None,
) -> BenchmarkSummary:
    """
    Benchmark all-reduce operations with different algorithms and sizes.

    Args:
        sizes: List of array sizes (in elements)
        algorithms: List of algorithm names to benchmark
        op: Reduction operation ("sum", "max", "min")
        num_processes: Number of processes (if None, detected automatically)

    Returns:
        BenchmarkSummary with results
    """
    if num_processes is None:
        num_processes = get_group_size()

    results = []

    for algo in algorithms:
        print(f"\nBenchmarking all_reduce with algorithm: {algo}")

        for size in sizes:
            # Create array
            x = mx.random.normal(shape=(size,))

            def run_benchmark():
                return mx.distributed.all_reduce_opt(x, op=op, algo=algo)

            try:
                mean_time, std_time, min_time, max_time = time_function(
                    run_benchmark, name=f"all_reduce_{algo}_{size}"
                )

                # Calculate metrics
                size_bytes = size * 4  # float32 = 4 bytes
                bandwidth = (
                    (size_bytes * num_processes) / (mean_time / 1000) / (1024**3)
                )
                ops_per_sec = 1000 / mean_time

                result = BenchmarkResult(
                    op="all_reduce",
                    algo=algo,
                    size=size_bytes,
                    num_processes=num_processes,
                    latency_ms=mean_time,
                    bandwidth_gbps=bandwidth,
                    operations_per_sec=ops_per_sec,
                )
                results.append(result)

                print(
                    f"  Size {size/1024:.2f}K: {mean_time:.3f} ms, "
                    f"{bandwidth:.2f} GB/s, {ops_per_sec:.1f} ops/s"
                )

            except Exception as e:
                print(f"  Error with size {size}: {e}")

    summary = BenchmarkSummary(results=results)
    summary.config = {
        "operation": "all_reduce",
        "reduction_op": op,
        "algorithms_tested": algorithms,
        "sizes_tested": sizes,
        "num_processes": num_processes,
    }

    return summary


def benchmark_all_gather(
    sizes: List[int],
    algorithms: List[str] = ["default", "linear", "ring", "recursive_doubling", "tree"],
    num_processes: int = None,
) -> BenchmarkSummary:
    """
    Benchmark all-gather operations with different algorithms and sizes.

    Args:
        sizes: List of array sizes (in elements)
        algorithms: List of algorithm names to benchmark
        num_processes: Number of processes (if None, detected automatically)

    Returns:
        BenchmarkSummary with results
    """
    if num_processes is None:
        num_processes = get_group_size()

    results = []

    for algo in algorithms:
        print(f"\nBenchmarking all_gather with algorithm: {algo}")

        for size in sizes:
            x = mx.random.normal(shape=(size,))

            def run_benchmark():
                return mx.distributed.all_gather_opt(x, algo=algo)

            try:
                mean_time, std_time, min_time, max_time = time_function(
                    run_benchmark, name=f"all_gather_{algo}_{size}"
                )

                # Calculate metrics
                size_bytes = size * 4  # float32 = 4 bytes
                total_bytes = size_bytes * num_processes
                bandwidth = total_bytes / (mean_time / 1000) / (1024**3)
                ops_per_sec = 1000 / mean_time

                result = BenchmarkResult(
                    op="all_gather",
                    algo=algo,
                    size=size_bytes,
                    num_processes=num_processes,
                    latency_ms=mean_time,
                    bandwidth_gbps=bandwidth,
                    operations_per_sec=ops_per_sec,
                )
                results.append(result)

                print(
                    f"  Size {size/1024:.2f}K: {mean_time:.3f} ms, "
                    f"{bandwidth:.2f} GB/s, {ops_per_sec:.1f} ops/s"
                )

            except Exception as e:
                print(f"  Error with size {size}: {e}")

    summary = BenchmarkSummary(results=results)
    summary.config = {
        "operation": "all_gather",
        "algorithms_tested": algorithms,
        "sizes_tested": sizes,
        "num_processes": num_processes,
    }

    return summary


def benchmark_reduce_scatter(
    sizes: List[int],
    algorithms: List[str] = ["default", "linear", "ring", "recursive_doubling", "tree"],
    op: str = "sum",
    num_processes: int = None,
) -> BenchmarkSummary:
    """
    Benchmark reduce-scatter operations with different algorithms and sizes.

    Args:
        sizes: List of array sizes (in elements) - must be divisible by num_processes
        algorithms: List of algorithm names to benchmark
        op: Reduction operation ("sum", "max", "min")
        num_processes: Number of processes (if None, detected automatically)

    Returns:
        BenchmarkSummary with results
    """
    if num_processes is None:
        num_processes = get_group_size()

    results = []

    for algo in algorithms:
        print(f"\nBenchmarking reduce_scatter with algorithm: {algo}")

        for size in sizes:
            # Ensure size is divisible by num_processes
            if size % num_processes != 0:
                size = ((size // num_processes) + 1) * num_processes

            x = mx.random.normal(shape=(size,))

            def run_benchmark():
                return mx.distributed.reduce_scatter_opt(x, op=op, algo=algo)

            try:
                mean_time, std_time, min_time, max_time = time_function(
                    run_benchmark, name=f"reduce_scatter_{algo}_{size}"
                )

                # Calculate metrics
                size_bytes = size * 4  # float32 = 4 bytes
                # Reduce-scatter sends size/num_processes to each rank
                send_bytes = size_bytes // num_processes
                bandwidth = (
                    (send_bytes * num_processes) / (mean_time / 1000) / (1024**3)
                )
                ops_per_sec = 1000 / mean_time

                result = BenchmarkResult(
                    op="reduce_scatter",
                    algo=algo,
                    size=size_bytes,
                    num_processes=num_processes,
                    latency_ms=mean_time,
                    bandwidth_gbps=bandwidth,
                    operations_per_sec=ops_per_sec,
                )
                results.append(result)

                print(
                    f"  Size {size/1024:.2f}K: {mean_time:.3f} ms, "
                    f"{bandwidth:.2f} GB/s, {ops_per_sec:.1f} ops/s"
                )

            except Exception as e:
                print(f"  Error with size {size}: {e}")

    summary = BenchmarkSummary(results=results)
    summary.config = {
        "operation": "reduce_scatter",
        "reduction_op": op,
        "algorithms_tested": algorithms,
        "sizes_tested": sizes,
        "num_processes": num_processes,
    }

    return summary


def benchmark_pipeline_overhead(
    num_stages: int = 4,
    sizes: List[int] = [1024, 4096, 16384],
    num_iterations: int = 20,
    num_processes: int = None,
) -> BenchmarkSummary:
    """
    Benchmark pipeline parallelism overhead.

    Args:
        num_stages: Number of pipeline stages
        sizes: List of input sizes to test
        num_iterations: Number of iterations for each size
        num_processes: Number of processes (if None, detected automatically)

    Returns:
        BenchmarkSummary with results
    """
    if num_processes is None:
        num_processes = get_group_size()

    results = []

    print(f"\nBenchmarking pipeline parallelism with {num_stages} stages")

    for size in sizes:
        input_array = mx.random.normal(shape=(size,))

        # Define simple pipeline stages
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

        def run_benchmark():
            return mx.distributed.execute_pipeline(stages, input_array)

        try:
            mean_time, std_time, min_time, max_time = time_function(
                run_benchmark,
                num_warmup=3,
                num_iters=num_iterations,
                name=f"pipeline_{num_stages}_stages_{size}",
            )

            # Calculate metrics
            size_bytes = size * 4

            result = BenchmarkResult(
                op="pipeline",
                algo=f"{num_stages}_stages",
                size=size_bytes,
                num_processes=num_processes,
                latency_ms=mean_time,
                bandwidth_gbps=0.0,  # Not applicable for pipeline
                operations_per_sec=1000 / mean_time,
            )
            results.append(result)

            print(
                f"  Size {size/1024:.2f}K ({num_stages} stages): "
                f"{mean_time:.3f} ms, {1000/mean_time:.1f} ops/s"
            )

        except Exception as e:
            print(f"  Error with size {size}: {e}")

    summary = BenchmarkSummary(results=results)
    summary.config = {
        "operation": "pipeline",
        "num_stages": num_stages,
        "sizes_tested": sizes,
        "num_iterations": num_iterations,
        "num_processes": num_processes,
    }

    return summary


def benchmark_bandwidth_scaling(
    sizes: List[int],
    algorithms: List[str] = ["default", "ring", "tree"],
    num_processes: int = None,
) -> BenchmarkSummary:
    """
    Benchmark bandwidth scaling with different group sizes.

    Args:
        sizes: List of array sizes to test
        algorithms: Algorithms to benchmark
        num_processes: Target number of processes (if None, use available)

    Returns:
        BenchmarkSummary with results
    """
    if num_processes is None:
        num_processes = get_group_size()

    results = []

    print(f"\nBenchmarking bandwidth scaling with {num_processes} processes")

    for algo in algorithms:
        print(f"  Algorithm: {algo}")

        for size in sizes:
            x = mx.random.normal(shape=(size,))

            def run_benchmark():
                return mx.distributed.all_reduce_opt(x, algo=algo)

            try:
                mean_time, std_time, min_time, max_time = time_function(
                    run_benchmark,
                    num_warmup=3,
                    num_iters=10,  # Fewer iterations for scaling tests
                    name=f"bandwidth_{algo}_{size}",
                )

                size_bytes = size * 4
                bandwidth = (
                    (size_bytes * num_processes) / (mean_time / 1000) / (1024**3)
                )

                result = BenchmarkResult(
                    op="bandwidth_scaling",
                    algo=algo,
                    size=size_bytes,
                    num_processes=num_processes,
                    latency_ms=mean_time,
                    bandwidth_gbps=bandwidth,
                    operations_per_sec=1000 / mean_time,
                )
                results.append(result)

                print(f"    Size {size/1024:.2f}K: {bandwidth:.2f} GB/s")

            except Exception as e:
                print(f"    Error with size {size}: {e}")

    summary = BenchmarkSummary(results=results)
    summary.config = {
        "operation": "bandwidth_scaling",
        "algorithms_tested": algorithms,
        "sizes_tested": sizes,
        "num_processes": num_processes,
    }

    return summary


def run_full_benchmark(num_processes: int = None) -> BenchmarkSummary:
    """
    Run the full benchmark suite.

    Args:
        num_processes: Number of processes (if None, detected automatically)

    Returns:
        BenchmarkSummary with all results
    """
    if num_processes is None:
        num_processes = get_group_size()

    print("=" * 80)
    print("MLX OPTIMIZED COLLECTIVES BENCHMARK")
    print("=" * 80)
    print(f"Number of processes: {num_processes}")

    # Define test sizes (in elements)
    small_sizes = [1024, 4096, 16384]  # Small: <64KB
    medium_sizes = [65536, 262144, 1048576]  # Medium: 64KB-4MB
    large_sizes = [2097152, 4194304, 8388608]  # Large: 8MB-32MB
    all_sizes = small_sizes + medium_sizes + large_sizes

    algorithms = ["default", "linear", "ring", "recursive_doubling", "tree"]

    # Run benchmarks
    all_results = []

    # 1. All-reduce benchmarks
    print("\n" + "=" * 80)
    print("1. ALL-REDUCE BENCHMARKS")
    print("=" * 80)

    for op in ["sum"]:
        summary = benchmark_all_reduce(
            sizes=all_sizes, algorithms=algorithms, op=op, num_processes=num_processes
        )
        all_results.extend(summary.results)

    # 2. All-gather benchmarks
    print("\n" + "=" * 80)
    print("2. ALL-GATHER BENCHMARKS")
    print("=" * 80)

    summary = benchmark_all_gather(
        sizes=all_sizes, algorithms=algorithms, num_processes=num_processes
    )
    all_results.extend(summary.results)

    # 3. Reduce-scatter benchmarks
    print("\n" + "=" * 80)
    print("3. REDUCE-SCATTER BENCHMARKS")
    print("=" * 80)

    for op in ["sum"]:
        summary = benchmark_reduce_scatter(
            sizes=medium_sizes + large_sizes,  # Skip very small for scatter
            algorithms=algorithms,
            op=op,
            num_processes=num_processes,
        )
        all_results.extend(summary.results)

    # 4. Pipeline benchmarks (smaller sizes only)
    print("\n" + "=" * 80)
    print("4. PIPELINE PARALLELISM BENCHMARKS")
    print("=" * 80)

    summary = benchmark_pipeline_overhead(
        num_stages=4, sizes=small_sizes, num_iterations=10, num_processes=num_processes
    )
    all_results.extend(summary.results)

    # 5. Bandwidth scaling benchmarks
    print("\n" + "=" * 80)
    print("5. BANDWIDTH SCALING BENCHMARKS")
    print("=" * 80)

    summary = benchmark_bandwidth_scaling(
        sizes=medium_sizes + large_sizes,
        algorithms=["ring", "tree"],
        num_processes=num_processes,
    )
    all_results.extend(summary.results)

    # Create final summary
    full_summary = BenchmarkSummary(results=all_results)
    full_summary.config = {
        "benchmark_type": "full",
        "num_processes": num_processes,
        "algorithms_tested": algorithms,
        "sizes_tested_by_category": {
            "small": [str(s) for s in small_sizes],
            "medium": [str(s) for s in medium_sizes],
            "large": [str(s) for s in large_sizes],
        },
    }

    return full_summary


def print_state_of_the_art_metrics():
    """Print state-of-the-art performance metrics reference."""
    print("\n" + "=" * 80)
    print("STATE-OF-THE-ART PERFORMANCE METRICS REFERENCE")
    print("=" * 80)

    metrics = """
1. BANDWIDTH METRICS
   - GB/s (Gigabytes per second): Primary metric for collective operations
   - Effective bandwidth = (data_size * num_processes) / time
   - Ideal: Limited by interconnect bandwidth (NVLink, PCIe, InfiniBand)

2. LATENCY METRICS
   - ms (milliseconds): Time for small messages (<1KB)
   - Latency = T0 + T1 * data_size
   - T0: fixed overhead (protocol, scheduling)
   - T1: per-byte transfer time

3. THROUGHPUT METRICS
   - operations/sec: How many collectives per second
   - Important for training workloads with frequent syncs

4. SCALING METRICS
   - Weak scaling:保持固定工作量 per process
   - Strong scaling: Total work fixed, more processes = less per process
   - Ideal linear scaling: N processes = 1/N time

5. COMPETITIVE BENCHMARKS
   - NCCL: NVIDIA's communication library (reference implementation)
   - MPI implementations (OpenMPI, MVAPICH)
   - PyTorch Distributed
   - JAX multi-slice/multi-host

6. EXPECTED PERFORMANCE (approximate)
   | Size          | Ring (GB/s) | Tree (GB/s) | Recursive Doubling |
   |---------------|-------------|-------------|-------------------|
   | 1KB           |     0.5     |     0.3     |       0.8         |
   | 1MB           |     8-12    |    10-15    |      10-14        |
   | 10MB          |    10-15    |    12-18    |      12-16        |
   | 100MB         |    12-18    |    15-22    |      14-20        |
   """

    print(metrics)


def parse_algorithms(algo_str: str) -> List[str]:
    """Parse comma-separated algorithm string."""
    if algo_str == "all":
        return ["default", "linear", "ring", "recursive_doubling", "tree"]
    return algo_str.split(",")


def main():
    parser = argparse.ArgumentParser(
        description="MLX Optimized Collectives Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark suite
  python optimized_collectives_bench.py --full
  
  # Benchmark all_reduce with specific algorithms
  python optimized_collectives_bench.py --op all_reduce --algo default,ring,tree
  
  # Run single operation benchmark
  python optimized_collectives_bench.py --op all_gather
  
  # Custom sizes
  python optimized_collectives_bench.py --sizes 1024,65536,262144
  
  # Output to JSON
  python optimized_collectives_bench.py --full --output results.json
  
  # Print state-of-the-art metrics reference
  python optimized_collectives_bench.py --metrics
  
Note: For distributed benchmarks, run with mpirun:
  mpirun -n 4 python optimized_collectives_bench.py --full
        """,
    )

    parser.add_argument(
        "--op",
        "-o",
        type=str,
        default=None,
        help="Operation to benchmark: all_reduce, all_gather, reduce_scatter, pipeline, bandwidth, or None for full suite",
    )

    parser.add_argument(
        "--algo",
        "-a",
        type=str,
        default="all",
        help="Algorithms to benchmark (comma-separated): default,linear,ring,recursive_doubling,tree, or 'all'",
    )

    parser.add_argument(
        "--sizes",
        "-s",
        type=str,
        default=None,
        help="Sizes to benchmark (comma-separated in elements), e.g., 1024,65536,262144",
    )

    parser.add_argument(
        "--full", "-f", action="store_true", help="Run full benchmark suite"
    )

    parser.add_argument(
        "--output", "-O", type=str, default=None, help="Output file for JSON results"
    )

    parser.add_argument(
        "--metrics",
        "-m",
        action="store_true",
        help="Print state-of-the-art metrics reference",
    )

    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print results without running benchmarks",
    )

    parser.add_argument(
        "--num-processes",
        "-p",
        type=int,
        default=None,
        help="Number of processes (if not using MPI)",
    )

    args = parser.parse_args()

    # Print metrics if requested
    if args.metrics:
        print_state_of_the_art_metrics()
        return

    # Parse algorithms
    algorithms = parse_algorithms(args.algo)

    # Parse sizes
    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",")]
    else:
        # Default sizes
        sizes = [1024, 65536, 262144, 1048576]

    # Check if we need to print only
    if args.print_only and args.output:
        try:
            with open(args.output, "r") as f:
                import json

                data = json.load(f)
                summary = BenchmarkSummary()
                summary.config = data.get("config", {})
                for r in data.get("results", []):
                    summary.results.append(BenchmarkResult(**r))
                summary.print_summary()
            return
        except FileNotFoundError:
            print(f"Error: Output file {args.output} not found")
            return

    # Run benchmarks
    if args.full or (not args.op and not args.print_only):
        summary = run_full_benchmark(num_processes=args.num_processes)
    else:
        # Single operation benchmark
        if args.op == "all_reduce":
            summary = benchmark_all_reduce(
                sizes=sizes, algorithms=algorithms, num_processes=args.num_processes
            )
        elif args.op == "all_gather":
            summary = benchmark_all_gather(
                sizes=sizes, algorithms=algorithms, num_processes=args.num_processes
            )
        elif args.op == "reduce_scatter":
            summary = benchmark_reduce_scatter(
                sizes=sizes, algorithms=algorithms, num_processes=args.num_processes
            )
        elif args.op == "pipeline":
            summary = benchmark_pipeline_overhead(
                sizes=sizes, num_processes=args.num_processes
            )
        elif args.op == "bandwidth":
            summary = benchmark_bandwidth_scaling(
                sizes=sizes, algorithms=algorithms, num_processes=args.num_processes
            )
        else:
            print(f"Unknown operation: {args.op}")
            parser.print_help()
            return

    # Print summary
    summary.print_summary()

    # Save to JSON if requested
    if args.output:
        filename = summary.to_json(args.output)
        print(f"\nResults saved to: {filename}")
    elif args.full:
        # Default output for full benchmark
        filename = summary.to_json("mlx_benchmark_results.json")
        print(f"\nResults saved to: {filename}")

    # Print comparison with reference
    if args.full:
        print("\n" + "=" * 80)
        print("COMPARISON WITH STATE-OF-THE-ART")
        print("=" * 80)

        # Get best algorithm for each operation
        ops = {}
        for r in summary.results:
            if r.op not in ops:
                ops[r.op] = {}

            # Find best latency for each size
            size_key = r.size
            if size_key not in ops[r.op]:
                ops[r.op][size_key] = r
            else:
                if r.latency_ms < ops[r.op][size_key].latency_ms:
                    ops[r.op][size_key] = r

        print("\nBest algorithms by operation and size:")
        for op, sizes in ops.items():
            print(f"\n{op}:")
            for size, result in sorted(sizes.items(), key=lambda x: x[0]):
                size_mb = size / (1024 * 1024)
                speedup = (
                    result.bandwidth_gbps / 5.0
                    if result.bandwidth_gbps > 0
                    else result.operations_per_sec / 100
                )
                print(
                    f"  {size_mb:.2f}MB: {result.algo} ({result.bandwidth_gbps:.2f} GB/s)"
                )


if __name__ == "__main__":
    main()
