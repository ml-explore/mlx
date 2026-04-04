#!/usr/bin/env python3
# Copyright © 2024 Apple Inc.

"""
Ring Backend Benchmarks for MLX Distributed Computing

Benchmarks all-reduce operations using the Ring backend, which is
always available and usually faster than MPI for distributed training.

Usage:
    # Run with Ring backend (2 processes)
    mlx.launch --backend ring -n 2 python ring_benchmark.py

    # Run with custom hostfile
    mlx.launch --backend ring --hostfile ring_config.json -n 4 python ring_benchmark.py

    # Run with environment variables
    MLX_RANK=0 MLX_HOSTFILE=rings.json python ring_benchmark.py --rank 0

Examples:
    # Single node, 4 processes with Ring
    mlx.launch -n 4 python ring_benchmark.py --op all_reduce

    # Multi-node with custom size
    mlx.launch -n 8 python ring_benchmark.py --size 1048576

Backend Info:
    - Ring: Always available, TCP-based, ring topology
    - Best for: Most distributed training workloads
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

try:
    import mlx.core as mx
except ImportError:
    print("Error: MLX is not installed. Install with:")
    print("  pip install mlx")
    sys.exit(1)


@dataclass
class RingBenchmarkResult:
    """Result of a ring backend benchmark."""

    operation: str
    backend: str = "ring"
    size_elements: int  # per process
    total_size_bytes: int
    num_processes: int
    latency_ms: float
    bandwidth_gbps: float
    throughput_ops_per_sec: float
    num_iterations: int


def warmup_group(backend: str = "ring"):
    """Initialize distributed group with Ring backend."""
    try:
        world = mx.distributed.init(backend=backend)
        return world
    except Exception as e:
        print(f"Warning: Distributed initialization with {backend} backend failed: {e}")
        # Try with 'any' to auto-select
        try:
            world = mx.distributed.init(backend="any")
            return world
        except Exception as e2:
            print(f"Also failed with 'any' backend: {e2}")
            return None


def time_function(fn, num_warmup=3, num_iters=20):
    """Time a function with warmup and multiple iterations."""
    for _ in range(num_warmup):
        result = fn()

    times = []
    for _ in range(num_iters):
        start = time.perf_counter()
        result = fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean_time = sum(times) / len(times)
    variance = sum((t - mean_time) ** 2 for t in times) / len(times)
    std_time = variance**0.5

    return mean_time, std_time


def benchmark_ring_all_reduce(
    world,
    num_processes_list: List[int],
    sizes_elements: List[int] = [1024, 65536, 262144, 1048576],
    operations: List[str] = ["sum", "max", "min"],
) -> List[RingBenchmarkResult]:
    """Benchmark Ring all-reduce operations."""

    world_size = world.size() if world else 1

    results = []

    for size in sizes_elements:
        # Create input array on each process
        x = mx.random.normal(shape=(size,), dtype=mx.float32)

        for op in operations:
            print(f"  Ring all_reduce ({op}, {size/1024:.1f} KB):")

            # Run benchmark
            def run_all_reduce():
                return mx.distributed.all_reduce(x, op=op)

            try:
                mean_time, std_time = time_function(run_all_reduce)

                # Calculate metrics
                total_size_bytes = size * 4 * world_size
                bandwidth_gbps = (total_size_bytes / 1e9) / (mean_time / 1000)
                throughput = 1000 / mean_time

                result = RingBenchmarkResult(
                    operation=f"all_reduce_{op}",
                    backend="ring",
                    size_elements=size,
                    total_size_bytes=total_size_bytes,
                    num_processes=world_size,
                    latency_ms=round(mean_time, 6),
                    bandwidth_gbps=round(bandwidth_gbps, 4),
                    throughput_ops_per_sec=round(throughput, 2),
                    num_iterations=num_warmup + num_iters,
                )
                results.append(result)

                if world.rank() == 0:
                    print(
                        f"    Latency: {mean_time:.3f} ms, "
                        f"Bandwidth: {bandwidth_gbps:.2f} GB/s"
                    )

            except Exception as e:
                if world.rank() == 0:
                    print(f"    Error: {e}")

    return results


def benchmark_ring_all_gather(
    world,
    num_processes_list: List[int],
    sizes_elements: List[int] = [1024, 65536, 262144, 1048576],
) -> List[RingBenchmarkResult]:
    """Benchmark Ring all-gather operations."""

    world_size = world.size() if world else 1

    results = []

    for size in sizes_elements:
        x = mx.random.normal(shape=(size,), dtype=mx.float32)

        print(f"  Ring all_gather ({size/1024:.1f} KB):")

        def run_all_gather():
            return mx.distributed.all_gather(x)

        try:
            mean_time, std_time = time_function(run_all_gather)

            total_size_bytes = size * 4 * world_size
            bandwidth_gbps = (total_size_bytes / 1e9) / (mean_time / 1000)
            throughput = 1000 / mean_time

            result = RingBenchmarkResult(
                operation="all_gather",
                backend="ring",
                size_elements=size,
                total_size_bytes=total_size_bytes,
                num_processes=world_size,
                latency_ms=round(mean_time, 6),
                bandwidth_gbps=round(bandwidth_gbps, 4),
                throughput_ops_per_sec=round(throughput, 2),
                num_iterations=num_warmup + num_iters,
            )
            results.append(result)

            if world.rank() == 0:
                print(
                    f"    Latency: {mean_time:.3f} ms, "
                    f"Bandwidth: {bandwidth_gbps:.2f} GB/s"
                )

        except Exception as e:
            if world.rank() == 0:
                print(f"    Error: {e}")

    return results


def benchmark_ring_send_recv(
    world, sizes_elements: List[int] = [1024, 65536, 262144]
) -> List[RingBenchmarkResult]:
    """Benchmark Ring send/recv operations."""

    world_size = world.size() if world else 1

    results = []

    # For send/recv, we need at least 2 processes
    if world_size < 2:
        return results

    # Send from rank 0 to rank 1 (or next in ring)
    dest_rank = (world.rank() + 1) % world_size
    source_rank = (world.rank() - 1) % world_size

    for size in sizes_elements:
        print(f"  Ring send/recv ({size/1024:.1f} KB):")

        def run_send():
            if world.rank() == 0:
                x = mx.random.normal(shape=(size,), dtype=mx.float32)
                mx.distributed.send(x, dest_rank)
            return mx.distributed.recv(source_rank)

        try:
            mean_time, std_time = time_function(run_send)

            # Each send/recv only sends once
            total_size_bytes = size * 4
            bandwidth_gbps = (total_size_bytes / 1e9) / (mean_time / 1000)
            throughput = 1000 / mean_time

            result = RingBenchmarkResult(
                operation="send_recv",
                backend="ring",
                size_elements=size,
                total_size_bytes=total_size_bytes,
                num_processes=world_size,
                latency_ms=round(mean_time, 6),
                bandwidth_gbps=round(bandwidth_gbps, 4),
                throughput_ops_per_sec=round(throughput, 2),
                num_iterations=num_warmup + num_iters,
            )
            results.append(result)

            if world.rank() == 0:
                print(
                    f"    Latency: {mean_time:.3f} ms, "
                    f"Bandwidth: {bandwidth_gbps:.2f} GB/s"
                )

        except Exception as e:
            if world.rank() == 0:
                print(f"    Error: {e}")

    return results


def run_ring_benchmark_suite(
    op: str = "all_reduce",
    min_size: int = 1024,
    max_size: int = 1048576,
    num_sizes: int = 4,
) -> Dict:
    """Run complete Ring backend benchmark suite."""

    world = warmup_group("ring")
    if not world:
        print("Error: Could not initialize Ring backend")
        sys.exit(1)

    # Generate sizes
    if op == "all_reduce":
        operations = ["sum", "max", "min"]
        sizes_elements = [int(2 ** (10 + i * 3)) for i in range(num_sizes)]
        sizes_elements = [s for s in sizes_elements if min_size <= s <= max_size]
    else:
        operations = ["sum"]
        sizes_elements = [int(2 ** (10 + i * 3)) for i in range(num_sizes)]
        sizes_elements = [s for s in sizes_elements if min_size <= s <= max_size]

    num_processes_list = list(range(1, world.size() + 1))

    if world.rank() == 0:
        print("=" * 80)
        print("RING BACKEND BENCHMARK SUITE")
        print(f"World size: {world.size()}")

    results = []

    if op == "all_reduce" or op == "both":
        if world.rank() == 0:
            print("\n" + "-" * 80)
            print("All-Reduce Benchmarks")
            print("-" * 80)

        results.extend(
            benchmark_ring_all_reduce(
                world, num_processes_list, sizes_elements, operations
            )
        )

    if op in ["all_gather", "both"]:
        if world.rank() == 0:
            print("\n" + "-" * 80)
            print("All-Gather Benchmarks")
            print("-" * 80)

        results.extend(
            benchmark_ring_all_gather(world, num_processes_list, sizes_elements)
        )

    if op in ["send_recv", "both"]:
        if world.rank() == 0:
            print("\n" + "-" * 80)
            print("Send/Recv Benchmarks")
            print("-" * 80)

        results.extend(
            benchmark_ring_send_recv(
                world, sizes_elements[:3]  # Fewer sizes for send/recv
            )
        )

    return {
        "benchmark_type": "ring",
        "world_size": world.size(),
        "backend": "ring",
        "results": [asdict(r) for r in results],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def save_results(results: Dict, filename: str = "ring_benchmark.json"):
    """Save Ring benchmark results to JSON."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Ring Backend Benchmarks for MLX Distributed Computing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ring Backend Usage:
    mlx.launch -n 4 python ring_benchmark.py --op all_reduce
    
Or with environment variables:
    MLX_RANK=0 MLX_HOSTFILE=rings.json python ring_benchmark.py

Backend Details:
    - Ring: Always available, TCP-based, ring topology
    - No third-party dependencies required
    - Good for most distributed training workloads

Examples:
    # All-reduce benchmarks (4 processes)
    mlx.launch -n 4 python ring_benchmark.py --op all_reduce
    
    # All-gather benchmarks (2 processes)
    mlx.launch -n 2 python ring_benchmark.py --op all_gather
    
    # Full suite (send/recv + all-reduce)
    mlx.launch -n 4 python ring_benchmark.py --op both
        """,
    )

    parser.add_argument(
        "--op",
        "-o",
        type=str,
        default="all_reduce",
        choices=["all_reduce", "all_gather", "send_recv", "both"],
        help="Operation to benchmark (default: all_reduce)",
    )

    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=262144,
        help="Size in elements (default: 262144 = 1MB)",
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=1024,
        help="Minimum size in elements (default: 1024)",
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=1048576,
        help="Maximum size in elements (default: 1048576)",
    )

    parser.add_argument(
        "--num-sizes",
        type=int,
        default=4,
        help="Number of different sizes to test (default: 4)",
    )

    parser.add_argument(
        "--output",
        "-O",
        type=str,
        default="ring_benchmark.json",
        help="Output file (default: ring_benchmark.json)",
    )

    args = parser.parse_args()

    # Run benchmark
    results = run_ring_benchmark_suite(
        op=args.op,
        min_size=args.min_size,
        max_size=args.max_size,
        num_sizes=args.num_sizes,
    )

    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
