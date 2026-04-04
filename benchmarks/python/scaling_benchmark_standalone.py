#!/usr/bin/env python3
# Copyright © 2024 Apple Inc.

"""
Standalone scaling benchmark for MLX (no MPI required).

This version simulates distributed performance metrics without requiring
an actual MPI installation. Use this for testing the benchmark infrastructure.

For real distributed benchmarks, install MPI:
    # Ubuntu/Debian
    sudo apt-get install openmpi-bin libopenmpi-dev

    # CentOS/RHEL
    sudo yum install openmpi openmpi-devel

    # macOS (Homebrew)
    brew install open-mpi
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


# Mock distributed module for standalone mode
class MockDistributedGroup:
    """Mock distributed group for standalone testing."""

    def __init__(self, size: int = 1):
        self._size = size

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return 0


# Override mlx.distributed if needed
class MockDistributedModule:
    """Mock distributed module for standalone testing."""

    def init(self) -> MockDistributedGroup:
        return MockDistributedGroup(size=1)


# Try to import real MLX, fall back to mock
try:
    import mlx.core as mx
except ImportError:
    # Create a minimal mock for testing without MLX
    class MockArray:
        def __init__(self, shape, dtype=None):
            self.shape = shape
            self.dtype = dtype

        def __repr__(self):
            return f"MockArray(shape={self.shape})"

    class MockDistributed:
        def init(self):
            return MockDistributedGroup(size=1)

    class MockMx:
        def random(self):
            return self

        def normal(self, shape, dtype=None):
            return MockArray(shape)

        distributed = MockDistributed()

    mx = MockMx()


def warmup_group():
    """Initialize distributed group."""
    try:
        world = mx.distributed.init()
        return world
    except Exception as e:
        # Use mock for standalone mode
        return MockDistributedGroup(size=1)


@dataclass
class ScalingResult:
    """Result of a single scaling benchmark."""

    num_processes: int
    size_elements: int
    total_size_bytes: int
    operation: str
    algorithm: str
    latency_ms: float
    bandwidth_gbps: Optional[float]
    throughput_ops_per_sec: float
    scalability_efficiency: float
    num_iterations: int


@dataclass
class ScalingExperiment:
    """Complete scaling experiment results."""

    name: str
    description: str
    num_processes: List[int]
    size_per_process: int
    total_size: Optional[int]
    operation: str
    algorithm: str
    results: List[ScalingResult]


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


def calculate_scalability_efficiency(
    time_single: float, time_multi: float, num_processes: int
) -> float:
    """Calculate scalability efficiency."""
    if num_processes == 1:
        return 1.0

    ideal_time = time_single / num_processes
    efficiency = ideal_time / time_multi if time_multi > 0 else 0.0

    return min(efficiency, 1.0)


def benchmark_all_reduce_scaling(
    num_processes_list: List[int],
    size_per_process: int,
    algorithms: List[str] = ["default", "ring", "tree"],
    operation: str = "sum",
) -> List[ScalingResult]:
    """Benchmark all-reduce scalability (standalone simulation)."""

    results = []

    for algo in algorithms:
        print(f"  Benchmarking all_reduce with {algo}...")

        for num_procs_test in sorted(set(num_processes_list)):
            # Simulate performance based on algorithm characteristics
            if algo == "ring":
                # Ring: O(n) communication steps
                base_time = 10.0 + (num_procs_test * 2)
            elif algo == "tree":
                # Tree: O(log n) communication steps
                base_time = 8.0 + (num_procs_test * 1.5)
            elif algo == "recursive_doubling":
                # Recursive doubling: O(log n) with better constants
                base_time = 7.0 + (num_procs_test * 1.2)
            else:
                # Default: automatic selection
                base_time = 9.0 + (num_procs_test * 1.8)

            # Add some variability
            mean_time = base_time + (num_procs_test * 0.5)

            # Calculate metrics
            total_size_bytes = (size_per_process * 4) * num_procs_test

            if mean_time > 0:
                bandwidth_gbps = (total_size_bytes / 1e9) / (mean_time / 1000)
            else:
                bandwidth_gbps = None

            throughput = 1000 / mean_time

            # Simulate efficiency
            if num_procs_test == 1:
                time_single = mean_time
                efficiency = 1.0
            else:
                if algo == "tree":
                    # Tree algorithm typically achieves 85-92% efficiency
                    efficiency = max(0.85, min(1.0, 1.0 - (num_procs_test * 0.02)))
                elif algo == "ring":
                    # Ring algorithm typically achieves 75-85% efficiency
                    efficiency = max(0.75, min(1.0, 1.0 - (num_procs_test * 0.03)))
                else:
                    # Default/recursive doubling: ~80-90%
                    efficiency = max(0.80, min(1.0, 1.0 - (num_procs_test * 0.025)))

            result = ScalingResult(
                num_processes=num_procs_test,
                size_elements=size_per_process,
                total_size_bytes=total_size_bytes,
                operation="all_reduce",
                algorithm=algo,
                latency_ms=round(mean_time, 6),
                bandwidth_gbps=round(bandwidth_gbps, 4) if bandwidth_gbps else None,
                throughput_ops_per_sec=round(throughput, 2),
                scalability_efficiency=round(efficiency, 4),
                num_iterations=num_procs_test * 20 + 3,
            )
            results.append(result)

            size_mb = (size_per_process * num_procs_test * 4) / (1024 * 1024)
            print(
                f"    {num_procs_test:3d} processes, {size_mb:8.2f} MB total: "
                f"{mean_time:.3f} ms, {bandwidth_gbps:.2f} GB/s, eff={efficiency:.2%}"
            )

    return results


def benchmark_pipeline_scaling(
    num_processes_list: List[int], size_per_process: int, num_stages: int = 4
) -> List[ScalingResult]:
    """Benchmark pipeline parallelism (standalone simulation)."""

    results = []

    print(f"  Benchmarking pipeline with {num_stages} stages...")

    for num_procs_test in sorted(set(num_processes_list)):
        # Simulate pipeline performance
        if num_procs_test == 1:
            base_time = 50.0
        else:
            # Pipeline has overhead but can overlap compute/comm
            base_time = 40.0 + (num_procs_test * 2)

        mean_time = base_time + (num_procs_test * 3)
        throughput = 1000 / mean_time

        # Pipeline efficiency
        if num_procs_test == 1:
            efficiency = 1.0
        else:
            # Pipeline typically achieves 70-85% efficiency due to overhead
            efficiency = max(0.70, min(1.0, 0.95 - (num_procs_test * 0.02)))

        result = ScalingResult(
            num_processes=num_procs_test,
            size_elements=size_per_process,
            total_size_bytes=size_per_process * 4,
            operation="pipeline",
            algorithm=f"stages_{num_stages}",
            latency_ms=round(mean_time, 6),
            bandwidth_gbps=None,
            throughput_ops_per_sec=round(throughput, 2),
            scalability_efficiency=round(efficiency, 4),
            num_iterations=num_procs_test * 20 + 3,
        )
        results.append(result)

        print(
            f"    {num_procs_test:3d} processes, {size_per_process/1024:.2f} KB/process: "
            f"{mean_time:.3f} ms, {throughput:.0f} ops/s, eff={efficiency:.2%}"
        )

    return results


def run_weak_scaling_benchmark(
    min_gpus: int = 2,
    max_gpus: int = 40,
    size_per_process: int = 1048576,
    algorithms: List[str] = ["default", "ring", "tree"],
) -> Dict:
    """Run weak scaling benchmark."""

    # Generate process counts (power of 2)
    num_processes_list = []
    n = min_gpus
    while n <= max_gpus:
        num_processes_list.append(n)
        if n * 2 > max_gpus:
            break
        n *= 2

    print("=" * 80)
    print("WEAK SCALING BENCHMARK (Standalone Mode)")
    print(f"Fixed size per GPU: {size_per_process * 4 / (1024*1024):.2f} MB")
    print(f"Process range: {min_gpus} to {max_gpus}")

    all_reduce_results = benchmark_all_reduce_scaling(
        num_processes_list=num_processes_list,
        size_per_process=size_per_process,
        algorithms=algorithms,
    )

    pipeline_results = benchmark_pipeline_scaling(
        num_processes_list=num_processes_list,
        size_per_process=size_per_process,
        num_stages=4,
    )

    return {
        "benchmark_type": "weak_scaling",
        "num_processes_range": num_processes_list,
        "size_per_process_elements": size_per_process,
        "algorithms_tested": algorithms,
        "experiments": {
            "all_reduce_weak_scaling": ScalingExperiment(
                name="All-Reduce Weak Scaling",
                description=f"Fixed size per GPU ({size_per_process * 4 / (1024*1024):.2f} MB)",
                num_processes=num_processes_list,
                size_per_process=size_per_process,
                total_size=None,
                operation="all_reduce",
                algorithm="combined",
                results=all_reduce_results,
            ),
            "pipeline_weak_scaling": ScalingExperiment(
                name="Pipeline Parallelism Weak Scaling",
                description=f"Fixed size per GPU with 4 stages",
                num_processes=num_processes_list,
                size_per_process=size_per_process,
                total_size=None,
                operation="pipeline",
                algorithm="4_stages",
                results=pipeline_results,
            ),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_strong_scaling_benchmark(
    min_gpus: int = 2,
    max_gpus: int = 40,
    total_size: int = 4194304,
    algorithms: List[str] = ["default", "ring", "tree"],
) -> Dict:
    """Run strong scaling benchmark."""

    num_processes_list = []
    n = min_gpus
    while n <= max_gpus:
        num_processes_list.append(n)
        if n * 2 > max_gpus:
            break
        n *= 2

    print("=" * 80)
    print("STRONG SCALING BENCHMARK (Standalone Mode)")
    print(f"Fixed total size: {total_size * 4 / (1024*1024):.2f} MB")
    print(f"Process range: {min_gpus} to {max_gpus}")

    all_reduce_results = benchmark_all_reduce_scaling(
        num_processes_list=num_processes_list,
        size_per_process=total_size // max_gpus,
        algorithms=algorithms,
    )

    pipeline_results = benchmark_pipeline_scaling(
        num_processes_list=num_processes_list,
        size_per_process=total_size // max_gpus,
        num_stages=4,
    )

    return {
        "benchmark_type": "strong_scaling",
        "num_processes_range": num_processes_list,
        "total_size_elements": total_size,
        "algorithms_tested": algorithms,
        "experiments": {
            "all_reduce_strong_scaling": ScalingExperiment(
                name="All-Reduce Strong Scaling",
                description=f"Fixed total size ({total_size * 4 / (1024*1024):.2f} MB)",
                num_processes=num_processes_list,
                size_per_process=total_size // max_gpus,
                total_size=total_size,
                operation="all_reduce",
                algorithm="combined",
                results=all_reduce_results,
            ),
            "pipeline_strong_scaling": ScalingExperiment(
                name="Pipeline Parallelism Strong Scaling",
                description=f"Fixed total size with 4 stages",
                num_processes=num_processes_list,
                size_per_process=total_size // max_gpus,
                total_size=total_size,
                operation="pipeline",
                algorithm="4_stages",
                results=pipeline_results,
            ),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_cpu_scaling_benchmark(
    min_cores: int = 1, max_cores: int = 16, size_per_core: int = 262144
) -> Dict:
    """Run CPU scaling benchmark."""

    num_cores_list = list(range(min_cores, min(max_cores + 1, max_cores + 1)))

    print("=" * 80)
    print("CPU SCALING BENCHMARK (Standalone Mode)")
    print(f"Fixed size per core: {size_per_core * 4 / (1024*1024):.2f} MB")
    print(f"Core range: {min_cores} to {max_cores}")

    results = []

    for num_cores in num_cores_list:
        # Simulate CPU performance
        if num_cores == 1:
            base_time = 50.0
        else:
            # CPU scaling typically better than GPU due to less communication overhead
            base_time = 40.0 + (num_cores * 1.5)

        mean_time = base_time
        throughput = (1000 / mean_time) * num_cores

        # CPU efficiency typically higher
        if num_cores == 1:
            efficiency = 1.0
        else:
            efficiency = max(0.85, min(1.0, 1.0 - (num_cores * 0.015)))

        result = ScalingResult(
            num_processes=num_cores,
            size_elements=size_per_core,
            total_size_bytes=size_per_core * 4 * num_cores,
            operation="cpu_compute",
            algorithm=f"{num_cores}_cores",
            latency_ms=round(mean_time, 6),
            bandwidth_gbps=None,
            throughput_ops_per_sec=round(throughput, 2),
            scalability_efficiency=round(efficiency, 4),
            num_iterations=num_cores * 20 + 3,
        )
        results.append(result)

        size_mb = (size_per_core * num_cores * 4) / (1024 * 1024)
        print(
            f"    {num_cores:3d} cores, {size_mb:8.2f} MB total: "
            f"{mean_time:.3f} ms, eff={efficiency:.2%}"
        )

    return {
        "benchmark_type": "cpu_scaling",
        "num_processes_range": num_cores_list,
        "size_per_core_elements": size_per_core,
        "experiments": {
            "cpu_scaling": ScalingExperiment(
                name="CPU Compute Scaling",
                description=f"Fixed size per core ({size_per_core * 4 / (1024*1024):.2f} MB)",
                num_processes=num_cores_list,
                size_per_process=size_per_core,
                total_size=None,
                operation="cpu_compute",
                algorithm="multi_threaded",
                results=results,
            )
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_full_scaling_benchmark(
    min_gpus: int = 2, max_gpus: int = 40, min_cores: int = 1, max_cores: int = 16
) -> Dict:
    """Run complete scaling benchmark suite."""

    results = {}

    print("=" * 80)
    print("FULL SCALING BENCHMARK SUITE (Standalone Mode)")
    print("=" * 80)

    # Weak scaling
    print("\n" + "=" * 80)
    print("PART 1: WEAK SCALING")
    print("=" * 80)

    results["weak_scaling"] = run_weak_scaling_benchmark(
        min_gpus=min_gpus,
        max_gpus=max_gpus,
        size_per_process=1048576,
        algorithms=["default", "ring", "tree"],
    )

    # Strong scaling
    print("\n" + "=" * 80)
    print("PART 2: STRONG SCALING")
    print("=" * 80)

    results["strong_scaling"] = run_strong_scaling_benchmark(
        min_gpus=min_gpus,
        max_gpus=min(max_gpus, 16),
        total_size=4194304,
        algorithms=["default", "ring", "tree"],
    )

    # CPU scaling
    print("\n" + "=" * 80)
    print("PART 3: CPU SCALING")
    print("=" * 80)

    results["cpu_scaling"] = run_cpu_scaling_benchmark(
        min_cores=min_cores, max_cores=max_cores, size_per_core=262144
    )

    return results


class ScalingResultEncoder(json.JSONEncoder):
    """Custom JSON encoder for dataclasses."""

    def default(self, obj):
        if isinstance(obj, ScalingResult):
            return asdict(obj)
        elif isinstance(obj, ScalingExperiment):
            return {
                "name": obj.name,
                "description": obj.description,
                "num_processes": obj.num_processes,
                "size_per_process": obj.size_per_process,
                "total_size": obj.total_size,
                "operation": obj.operation,
                "algorithm": obj.algorithm,
                "results": [asdict(r) for r in obj.results],
            }
        return super().default(obj)


def save_results(results: Dict, filename: str = "scaling_benchmark.json"):
    """Save benchmark results to JSON."""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2, cls=ScalingResultEncoder)
    print(f"\nResults saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="MLX Distributed Scaling Performance Benchmark (Standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This is a standalone version that doesn't require MPI installation.

Examples:
  # Run full benchmark suite
  python scaling_benchmark_standalone.py --full
  
  # Run weak scaling only
  python scaling_benchmark_standalone.py --weak-scaling
  
  # Run strong scaling only  
  python scaling_benchmark_standalone.py --strong-scaling
  
  # Run CPU scaling only
  python scaling_benchmark_standalone.py --cpu-scaling

For distributed benchmarks with real MPI:
    mpirun -n <num> python scaling_benchmark.py
        """,
    )

    parser.add_argument(
        "--weak-scaling", action="store_true", help="Run weak scaling benchmarks"
    )

    parser.add_argument(
        "--strong-scaling", action="store_true", help="Run strong scaling benchmarks"
    )

    parser.add_argument(
        "--cpu-scaling", action="store_true", help="Run CPU scaling benchmarks"
    )

    parser.add_argument("--full", action="store_true", help="Run full benchmark suite")

    parser.add_argument(
        "--min-gpus", type=int, default=2, help="Minimum number of GPUs (default: 2)"
    )

    parser.add_argument(
        "--max-gpus", type=int, default=40, help="Maximum number of GPUs (default: 40)"
    )

    parser.add_argument(
        "--min-cores",
        type=int,
        default=1,
        help="Minimum number of CPU cores (default: 1)",
    )

    parser.add_argument(
        "--max-cores",
        type=int,
        default=16,
        help="Maximum number of CPU cores (default: 16)",
    )

    parser.add_argument(
        "--algo",
        "-a",
        type=str,
        default="default,ring,tree",
        help="Algorithms to benchmark (comma-separated)",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=1048576,
        help="Size in elements (default: 1048576 = 4MB)",
    )

    parser.add_argument(
        "--output",
        "-O",
        type=str,
        default="scaling_benchmark_standalone.json",
        help="Output file for JSON results",
    )

    args = parser.parse_args()

    # Run benchmarks
    if args.full or (
        not any([args.weak_scaling, args.strong_scaling, args.cpu_scaling])
    ):
        results = run_full_scaling_benchmark(
            min_gpus=args.min_gpus,
            max_gpus=args.max_gpus,
            min_cores=args.min_cores,
            max_cores=args.max_cores,
        )
    elif args.weak_scaling:
        algorithms = [a.strip() for a in args.algo.split(",")]
        results = run_weak_scaling_benchmark(
            min_gpus=args.min_gpus,
            max_gpus=args.max_gpus,
            size_per_process=args.size,
            algorithms=algorithms,
        )
    elif args.strong_scaling:
        algorithms = [a.strip() for a in args.algo.split(",")]
        results = run_strong_scaling_benchmark(
            min_gpus=args.min_gpus,
            max_gpus=min(args.max_gpus, 16),
            total_size=args.size,
            algorithms=algorithms,
        )
    elif args.cpu_scaling:
        results = run_cpu_scaling_benchmark(
            min_cores=args.min_cores, max_cores=args.max_cores, size_per_core=args.size
        )

    # Save results
    save_results(results, args.output)

    print("\nStandalone benchmark complete!")
    print("To install MPI for real distributed benchmarks:")
    print("  Ubuntu: sudo apt-get install openmpi-bin libopenmpi-dev")
    print("  CentOS: sudo yum install openmpi openmpi-devel")
    print("  macOS: brew install open-mpi")


if __name__ == "__main__":
    main()
