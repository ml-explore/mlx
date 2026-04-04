#!/usr/bin/env python3
# Copyright © 2024 Apple Inc.

"""
Backend Comparison Benchmark for MLX Distributed Computing

Compares performance across all supported backends:
- RING: Always available, TCP-based
- JACCL: RDMA over Thunderbolt (macOS 26.2+)
- MPI: Full-featured distributed communications

Usage:
    # Run comparison (auto-select best backend)
    python backend_comparison.py --json ring_benchmark.json jaccl_benchmark.json mpi_benchmark.json

    # Generate summary plots
    python backend_comparison.py --json *.json --summary

Backend Selection:
    mlx.distributed.init(backend="ring")      # Always available
    mlx.distributed.init(backend="jaccl")     # RDMA over Thunderbolt
    mlx.distributed.init(backend="mpi")       # Full MPI implementation
    mlx.distributed.init(backend="any")       # Auto-select best available
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required for plotting.")
    print("Install with: pip install matplotlib numpy")
    sys.exit(1)


# Backend colors and styles
BACKEND_STYLES = {
    "ring": {"color": "#3498db", "marker": "o", "label": "Ring (TCP)"},
    "jaccl": {"color": "#e74c3c", "marker": "^", "label": "JACCL (RDMA)"},
    "mpi": {"color": "#2ecc71", "marker": "s", "label": "MPI"},
    "default": {"color": "#95a5a6", "marker": "d", "label": "Default"},
    "auto": {"color": "#9b59b6", "marker": "*", "label": "Auto-select"},
}


def load_benchmark_data(json_files: List[str]) -> Dict:
    """Load benchmark data from JSON files."""
    all_data = {}

    for filepath in json_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)

            backend = data.get("backend", Path(filepath).stem.split("_")[0])

            # Ensure results list
            if "results" in data:
                all_data[backend] = {
                    **data,
                    "results": [
                        r if isinstance(r, dict) else dict(r) for r in data["results"]
                    ],
                }
            else:
                # Try to extract results from experiments
                if "experiments" in data:
                    for exp_name, experiment in data["experiments"].items():
                        if isinstance(experiment, dict) and "results" in experiment:
                            results = [
                                r if isinstance(r, dict) else dict(r)
                                for r in experiment["results"]
                            ]
                            if results:
                                all_data[backend] = {**data, "results": results}
                                break

        except FileNotFoundError:
            print(f"Warning: File not found: {filepath}")
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in {filepath}: {e}")

    return all_data


def extract_metrics(all_data: Dict) -> Dict:
    """Extract performance metrics by operation and size."""
    metrics = {}

    for backend, data in all_data.items():
        if "results" not in data:
            continue

        for result in data["results"]:
            operation = result.get("operation", "unknown")
            size = result.get("size_elements", 0)

            # Create unique key
            key = f"{backend}_{operation}"

            if key not in metrics:
                metrics[key] = {
                    "backend": backend,
                    "operation": operation,
                    "sizes": [],
                    "latencies": [],
                    "bandwidths": [],
                    "throughputs": [],
                }

            metrics[key]["sizes"].append(size)
            metrics[key]["latencies"].append(result.get("latency_ms", 0))
            metrics[key]["bandwidths"].append(result.get("bandwidth_gbps", 0))
            metrics[key]["throughputs"].append(result.get("throughput_ops_per_sec", 0))

    return metrics


def plot_backend_comparison(metrics: Dict, output_dir: str = "backend_comparison"):
    """Create comprehensive backend comparison plots."""

    os.makedirs(output_dir, exist_ok=True)

    # Group by operation
    operations = {}
    for key in metrics.keys():
        op = key.split("_", 1)[1] if "_" in key else key
        if op not in operations:
            operations[op] = []
        operations[op].append(key)

    for operation, keys in operations.items():
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f'Backend Comparison: {operation.replace("_", " ").title()}',
            fontsize=14,
            fontweight="bold",
        )

        # Ensure 2x2 grid
        axes = np.array(axes).flatten()

        # Extract data for this operation
        backends_data = []
        for key in keys:
            if key in metrics:
                backends_data.append(metrics[key])

        # Filter to only have backends present in all data points
        sizes_map = {}
        for data in backends_data:
            backend = data["backend"]
            # Sort by size
            sorted_indices = np.argsort(data["sizes"])
            sizes_map[backend] = {
                "sizes": [data["sizes"][i] for i in sorted_indices],
                "latencies": [data["latencies"][i] for i in sorted_indices],
                "bandwidths": [data["bandwidths"][i] for i in sorted_indices],
                "throughputs": [data["throughputs"][i] for i in sorted_indices],
            }

        # Plot 1: Latency comparison
        ax = axes[0]
        for backend, data in sizes_map.items():
            style = BACKEND_STYLES.get(backend, BACKEND_STYLES["default"])
            ax.plot(
                data["sizes"],
                data["latencies"],
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Message Size (elements)", fontsize=12)
        ax.set_ylabel("Latency (ms)", fontsize=12)
        ax.set_title(
            f'Latency vs Size\n{operation.replace("_", " ").title()}', fontsize=12
        )
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xscale(
            "log" if max(d["sizes"] for d in sizes_map.values()) > 100 else "linear"
        )

        # Plot 2: Bandwidth comparison
        ax = axes[1]
        for backend, data in sizes_map.items():
            style = BACKEND_STYLES.get(backend, BACKEND_STYLES["default"])
            ax.plot(
                data["sizes"],
                data["bandwidths"],
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Message Size (elements)", fontsize=12)
        ax.set_ylabel("Bandwidth (GB/s)", fontsize=12)
        ax.set_title(
            f'Bandwidth vs Size\n{operation.replace("_", " ").title()}', fontsize=12
        )
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)

        # Plot 3: Throughput comparison
        ax = axes[2]
        for backend, data in sizes_map.items():
            style = BACKEND_STYLES.get(backend, BACKEND_STYLES["default"])
            ax.plot(
                data["sizes"],
                data["throughputs"],
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Message Size (elements)", fontsize=12)
        ax.set_ylabel("Throughput (Ops/sec)", fontsize=12)
        ax.set_title(
            f'Throughput vs Size\n{operation.replace("_", " ").title()}', fontsize=12
        )
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)

        # Plot 4: Performance summary bar chart
        ax = axes[3]

        if sizes_map:
            # Find largest size for comparison
            all_sizes = []
            for data in sizes_map.values():
                all_sizes.extend(data["sizes"])

            max_size = max(all_sizes) if all_sizes else 0

            # Extract best performance for each backend at largest size
            backend_stats = {}
            for backend, data in sizes_map.items():
                # Find closest to max size
                idx = min(
                    range(len(data["sizes"])),
                    key=lambda i: abs(data["sizes"][i] - max_size),
                )
                backend_stats[backend] = {
                    "latency": data["latencies"][idx],
                    "bandwidth": data["bandwidths"][idx],
                    "throughput": data["throughputs"][idx],
                }

            x_pos = np.arange(len(backend_stats))
            width = 0.25

            # Latency bar
            latencies = [backend_stats[b]["latency"] for b in backend_stats]
            bars1 = ax.bar(
                x_pos - width, latencies, width, label="Latency (ms)", color="#e74c3c"
            )

            # Bandwidth bar
            bandwidths = [backend_stats[b]["bandwidth"] for b in backend_stats]
            bars2 = ax.bar(
                x_pos, bandwidths, width, label="Bandwidth (GB/s)", color="#3498db"
            )

            # Throughput bar
            throughputs = [backend_stats[b]["throughput"] for b in backend_stats]
            bars3 = ax.bar(
                x_pos + width, throughputs, width, label="Throughput", color="#2ecc71"
            )

            ax.set_xlabel("Backend", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)
            ax.set_title(
                f"Performance Summary at {max_size} elements\n"
                f'{operation.replace("_", " ").title()}',
                fontsize=12,
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [BACKEND_STYLES.get(b, {}).get("label", b) for b in backend_stats],
                rotation=45,
                ha="right",
            )
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        output_path = os.path.join(output_dir, f"backend_comparison_{operation}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Created: {output_path}")

    # Create overall summary plot
    create_overall_summary(sizes_map, operations, output_dir)


def create_overall_summary(sizes_map: Dict, operations: Dict, output_dir: str):
    """Create overall performance summary."""

    fig, ax = plt.subplots(figsize=(14, 8))

    # Collect all backends
    backends = list(sizes_map.keys())

    if not backends:
        return

    # Get all metrics
    latencies = []
    bandwidths = []
    throughputs = []

    for backend in backends:
        if backend in sizes_map:
            data = sizes_map[backend]
            # Find largest size
            max_size_idx = np.argmax(data["sizes"])
            latencies.append(data["latencies"][max_size_idx])
            bandwidths.append(data["bandwidths"][max_size_idx])
            throughputs.append(data["throughputs"][max_size_idx])

    x_pos = np.arange(len(backends))
    width = 0.25

    # Create bars
    bars1 = ax.bar(
        x_pos - width, latencies, width, label="Latency (ms)", color="#e74c3c"
    )
    bars2 = ax.bar(x_pos, bandwidths, width, label="Bandwidth (GB/s)", color="#3498db")
    bars3 = ax.bar(
        x_pos + width, throughputs, width, label="Throughput", color="#2ecc71"
    )

    ax.set_xlabel("Backend", fontsize=14, fontweight="bold")
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(
        "Overall Performance Comparison\n"
        f"{len(operations)} Operations | {len(backends)} Backends",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [BACKEND_STYLES.get(b, {}).get("label", b) for b in backends],
        rotation=45,
        ha="right",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar, val in zip(
            bars,
            (
                bars.get_width()
                if hasattr(bars[0], "get_width")
                else [b.get_height() for b in bars]
            ),
        ):
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    output_path = os.path.join(output_dir, "overall_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Created: {output_path}")


def generate_summary_report(all_data: Dict, metrics: Dict, output_dir: str):
    """Generate comprehensive summary report."""

    os.makedirs(output_dir, exist_ok=True)

    # Create text summary
    summary_path = os.path.join(output_dir, "summary_report.txt")

    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MLX Backend Comparison Summary Report\n")
        f.write("=" * 80 + "\n\n")

        # Overview
        f.write("OVERVIEW\n")
        f.write("-" * 80 + "\n")
        f.write(f"Backends compared: {len(all_data)}\n")

        for backend, data in all_data.items():
            results_count = len(data.get("results", []))
            f.write(f"  - {backend}: {results_count} test results\n")

        f.write("\n")

        # Best backend for each operation
        f.write("BEST BACKEND BY OPERATION\n")
        f.write("-" * 80 + "\n\n")

        operations = {}
        for key in metrics.keys():
            op = key.split("_", 1)[1] if "_" in key else key
            if op not in operations:
                operations[op] = {}

        for key, data in metrics.items():
            if "_" in key:
                op = key.split("_", 1)[1]

                # Find best by bandwidth (higher is better)
                max_bw_idx = np.argmax(data["bandwidths"])

                if op not in operations or data["backend"] not in operations[op]:
                    operations[op][data["backend"]] = {
                        "bandwidth": data["bandwidths"][max_bw_idx],
                        "latency": data["latencies"][max_bw_idx],
                    }

        for op, backends in operations.items():
            if not backends:
                continue

            # Find best backend
            best_backend = max(backends.keys(), key=lambda b: backends[b]["bandwidth"])

            f.write(f"{op.replace('_', ' ').title()}:\n")
            f.write(f"  Best: {best_backend}\n")
            f.write(f"  Bandwidth: {backends[best_backend]['bandwidth']:.2f} GB/s\n")
            f.write(f"  Latency: {backends[best_backend]['latency']:.3f} ms\n")
            f.write("\n")

        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 80 + "\n\n")

        if "jaccl" in all_data:
            f.write("1. For ultra-low latency (macOS with Thunderbolt): Use JACCL\n")

        if "ring" in all_data:
            f.write("2. For universal compatibility: Use Ring (no dependencies)\n")

        if "mpi" in all_data:
            f.write("3. For enterprise environments: Use MPI\n")

        if "auto" in all_data or "default" in all_data:
            f.write("4. For automatic selection: Use backend='auto'\n")

        f.write("\n")
        f.write(
            f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    print(f"  Created: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Backend Comparison for MLX Distributed Computing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Backend Comparison Usage:

    # Compare multiple backends
    python backend_comparison.py --json ring_benchmark.json jaccl_benchmark.json mpi_benchmark.json
    
    # Generate summary plots
    python backend_comparison.py --json *.json --output comparison_plots/
    
    # Generate only summary report
    python backend_comparison.py --json *.json --summary

Backends Supported:
    - ring: Always available, TCP-based
    - jaccl: RDMA over Thunderbolt (macOS 26.2+)
    - mpi: Full MPI implementation
        """,
    )

    parser.add_argument(
        "--json", "-j", type=str, nargs="+", required=True, help="JSON files to compare"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="backend_comparison",
        help="Output directory for plots (default: backend_comparison)",
    )

    parser.add_argument(
        "--summary", action="store_true", help="Generate summary report only"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading benchmark data from {len(args.json)} files...")
    all_data = load_benchmark_data(args.json)

    if not all_data:
        print("Error: No valid benchmark data found!")
        sys.exit(1)

    print(f"Loaded backends: {', '.join(all_data.keys())}")

    # Extract metrics
    print("\nExtracting performance metrics...")
    metrics = extract_metrics(all_data)

    if args.summary:
        # Generate summary report
        generate_summary_report(all_data, metrics, args.output)
    else:
        # Generate all plots
        print(f"\nGenerating comparison plots to: {args.output}")
        plot_backend_comparison(metrics, args.output)

        # Generate summary report
        generate_summary_report(all_data, metrics, args.output)


if __name__ == "__main__":
    main()
