#!/usr/bin/env python3
# Copyright © 2024 Apple Inc.

"""
State-of-the-art plotting for MLX scaling benchmarks.

This module creates publication-quality visualizations of distributed
performance scaling including:

- Scalability curves (linear, strong, weak)
- Efficiency plots
- Bandwidth vs GPU count
- Algorithm comparison charts
- Speedup plots
- Scalability heatmaps

Requirements:
    pip install matplotlib numpy pandas

Usage:
    python plot_scaling.py --json scaling_benchmark.json --output plots/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for server use
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("Error: matplotlib, numpy, and pandas are required.")
    print("Install with: pip install matplotlib numpy pandas")
    sys.exit(1)


# Set up style for publication-quality plots
plt.style.use("seaborn-v0_8-whitegrid")

# Custom color palette for algorithm comparison
ALGO_COLORS = {
    "default": "#2ecc71",  # Green
    "linear": "#3498db",  # Blue
    "ring": "#e74c3c",  # Red
    "recursive_doubling": "#9b59b6",  # Purple
    "tree": "#f1c40f",  # Yellow
}

ALGO_MARKERS = {
    "default": "o",
    "linear": "s",
    "ring": "^",
    "recursive_doubling": "D",
    "tree": "v",
}

ALGO_LINESTYLES = {
    "default": "-",
    "linear": "--",
    "ring": "-.",
    "recursive_doubling": ":",
    "tree": "-",
}


class ScalingPlotter:
    """State-of-the-art plotting for MLX scaling benchmarks."""

    def __init__(self, data: Dict):
        """Initialize plotter with benchmark data."""
        self.data = data
        self.results_df = self._prepare_dataframes()

    def _prepare_dataframes(self) -> Dict[str, pd.DataFrame]:
        """Convert benchmark data to DataFrames for easier plotting."""
        dfs = {}

        for bench_type, bench_data in self.data.items():
            if not isinstance(bench_data, dict):
                continue

            # Handle experiments
            if "experiments" in bench_data:
                for exp_name, experiment in bench_data["experiments"].items():
                    if isinstance(experiment, dict):
                        exp_results = experiment.get("results", [])
                    else:
                        # Try to get from object-like dict
                        exp_results = experiment.get("results", [])

                    if not exp_results:
                        continue

                    df = pd.DataFrame(exp_results)
                    dfs[exp_name] = df

            # Handle results directly
            elif "results" in bench_data:
                results_list = bench_data["results"]
                if isinstance(results_list, list) and len(results_list) > 0:
                    # Check if it's nested
                    if isinstance(results_list[0], dict):
                        df = pd.DataFrame(results_list)
                        dfs[f"{bench_type}_results"] = df

        return dfs

    def _get_metric(self, df: pd.DataFrame, metric: str) -> np.ndarray:
        """Get metric from dataframe."""
        if metric == "bandwidth_gbps":
            return df["bandwidth_gbps"].values
        elif metric == "latency_ms":
            return df["latency_ms"].values
        elif metric == "throughput_ops_per_sec":
            return df["throughput_ops_per_sec"].values
        elif metric == "efficiency":
            return df["scalability_efficiency"].values
        else:
            return df[metric].values

    def _get_num_processes(self, df: pd.DataFrame) -> np.ndarray:
        """Get number of processes from dataframe."""
        return df["num_processes"].values

    def _get_algorithms(self, df: pd.DataFrame) -> List[str]:
        """Get list of algorithms from dataframe."""
        if "algorithm" in df.columns:
            return sorted(df["algorithm"].unique().tolist())
        else:
            # Check for operation + algorithm columns
            if "operation" in df.columns and "algorithm" in df:
                combined = (
                    df["operation"].astype(str) + "_" + df["algorithm"].astype(str)
                )
                return sorted(combined.unique().tolist())
            else:
                return ["default"]

    def create_all_plots(self, output_dir: str = "plots"):
        """Create all available plots."""
        os.makedirs(output_dir, exist_ok=True)

        print(f"Creating plots in: {output_dir}")

        # 1. Scalability Plots
        self.plot_scalability_curves(output_dir)

        # 2. Efficiency Plots
        self.plot_efficiency_scatter(output_dir)

        # 3. Algorithm Comparison
        self.plot_algorithm_comparison(output_dir)

        # 4. Bandwidth vs GPU Count
        self.plot_bandwidth_vs_gpu(output_dir)

        # 5. Speedup Plots
        self.plot_speedup_plots(output_dir)

        # 6. Scalability Heatmaps
        self.plot_scalability_heatmap(output_dir)

        # 7. Throughput vs GPU Count
        self.plot_throughput_vs_gpu(output_dir)

        # 8. Latency vs GPU Count
        self.plot_latency_vs_gpu(output_dir)

        print("\nAll plots created!")

    def plot_scalability_curves(
        self, output_dir: str = "plots", metric: str = "latency_ms"
    ):
        """Plot scalability curves showing performance vs GPU count."""

        for exp_name, df in self.results_df.items():
            if df.empty:
                continue

            fig, ax = plt.subplots(1, 2, figsize=(16, 6))

            # Get metrics
            num_gpus = self._get_num_processes(df)
            metric_vals = self._get_metric(df, metric)

            # Plot raw values
            ax[0].plot(
                num_gpus,
                metric_vals,
                "o-",
                linewidth=2,
                markersize=8,
                color="#3498db",
                label=f'{metric.replace("_", " ").title()}',
            )

            # Add ideal scaling line
            if num_gpus[0] == 1:
                ideal = metric_vals[0] / num_gpus
                ax[0].plot(
                    num_gpus,
                    ideal,
                    "--",
                    linewidth=2,
                    color="red",
                    label="Ideal Scaling",
                )

            ax[0].set_xlabel("Number of GPUs/Processes", fontsize=12)
            ax[0].set_ylabel(metric.replace("_", " ").title(), fontsize=12)
            ax[0].set_title(
                f'Scalability Curve: {exp_name.replace("_", " ").title()}', fontsize=14
            )
            ax[0].legend(fontsize=10)
            ax[0].grid(True, alpha=0.3)

            # Plot log scale
            ax[1].loglog(
                num_gpus,
                metric_vals,
                "o-",
                linewidth=2,
                markersize=8,
                color="#3498db",
                label=f'{metric.replace("_", " ").title()}',
            )

            if num_gpus[0] == 1:
                ideal = metric_vals[0] / num_gpus
                ax[1].loglog(
                    num_gpus,
                    ideal,
                    "--",
                    linewidth=2,
                    color="red",
                    label="Ideal Scaling",
                )

            ax[1].set_xlabel("Number of GPUs/Processes (log scale)", fontsize=12)
            ax[1].set_ylabel(
                f'{metric.replace("_", " ").title()} (log scale)', fontsize=12
            )
            ax[1].set_title("Scalability Curve (Log-Log)", fontsize=14)
            ax[1].legend(fontsize=10)
            ax[1].grid(True, which="both", alpha=0.3)

            plt.tight_layout()
            output_path = os.path.join(output_dir, f"scalability_{exp_name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"  Created: {output_path}")

    def plot_efficiency_scatter(self, output_dir: str = "plots"):
        """Plot efficiency vs GPU count scatter plot."""

        for exp_name, df in self.results_df.items():
            if df.empty or "scalability_efficiency" not in df.columns:
                continue

            fig, ax = plt.subplots(1, 2, figsize=(16, 6))

            num_gpus = self._get_num_processes(df)
            efficiency = df["scalability_efficiency"].values

            # Scatter plot
            scatter = ax[0].scatter(
                num_gpus,
                efficiency,
                c=efficiency,
                cmap="viridis",
                s=100,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
            )

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax[0])
            cbar.set_label("Scalability Efficiency", fontsize=11)

            # Add ideal efficiency line
            ax[0].axhline(
                y=1.0,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Perfect Scalability (100%)",
            )

            # Add reference lines
            for efficiency_line in [0.9, 0.8, 0.7, 0.6]:
                ax[0].axhline(
                    y=efficiency_line,
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                    alpha=0.5,
                )

            ax[0].set_xlabel("Number of GPUs/Processes", fontsize=12)
            ax[0].set_ylabel("Scalability Efficiency", fontsize=12)
            ax[0].set_title(
                f'Scalability Efficiency: {exp_name.replace("_", " ").title()}',
                fontsize=14,
            )
            ax[0].set_ylim(0, 1.1)
            ax[0].legend(fontsize=10)

            # Bar chart
            width = 0.6
            x_pos = np.arange(len(num_gpus))

            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(num_gpus)))
            bars = ax[1].bar(
                x_pos,
                efficiency,
                width=width,
                color=colors,
                edgecolor="black",
                alpha=0.8,
            )

            ax[1].set_xlabel("Number of GPUs/Processes", fontsize=12)
            ax[1].set_ylabel("Scalability Efficiency", fontsize=12)
            ax[1].set_title(
                f'Efficiency by Process Count: {exp_name.replace("_", " ").title()}',
                fontsize=14,
            )
            ax[1].set_xticks(x_pos)
            ax[1].set_xticklabels(num_gpus)
            ax[1].axhline(
                y=1.0,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Perfect Scalability",
            )
            ax[1].legend(fontsize=10)

            # Add value labels on bars
            for bar, eff in zip(bars, efficiency):
                height = bar.get_height()
                ax[1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{eff:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            plt.tight_layout()
            output_path = os.path.join(output_dir, f"efficiency_{exp_name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"  Created: {output_path}")

    def plot_algorithm_comparison(
        self, output_dir: str = "plots", metric: str = "bandwidth_gbps"
    ):
        """Compare performance across different algorithms."""

        # Group by operation
        ops = {}
        for exp_name, df in self.results_df.items():
            if "operation" in df.columns:
                op = df["operation"].iloc[0]
                if op not in ops:
                    ops[op] = []
                ops[op].append(df)

        if not ops:
            # Try to group by algorithm column
            for exp_name, df in self.results_df.items():
                if "algorithm" in df.columns and len(df["algorithm"].unique()) > 1:
                    fig, ax = plt.subplots(figsize=(12, 7))

                    for algo in df["algorithm"].unique():
                        algo_df = df[df["algorithm"] == algo].sort_values(
                            "num_processes"
                        )
                        num_gpus = self._get_num_processes(algo_df)
                        metric_vals = self._get_metric(algo_df, metric)

                        color = ALGO_COLORS.get(algo, "#3498db")
                        marker = ALGO_MARKERS.get(algo, "o")

                        ax.plot(
                            num_gpus,
                            metric_vals,
                            marker=marker,
                            color=color,
                            linewidth=2,
                            markersize=8,
                            label=f'{algo.replace("_", " ").title()}',
                        )

                    ax.set_xlabel("Number of GPUs/Processes", fontsize=12)
                    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
                    ax.set_title(
                        f'Algorithm Comparison: {metric.replace("_", " ").title()}',
                        fontsize=14,
                    )
                    ax.legend(fontsize=10, loc="best")
                    ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    output_path = os.path.join(
                        output_dir, f"algorithm_comparison_{exp_name}.png"
                    )
                    plt.savefig(output_path, dpi=300, bbox_inches="tight")
                    plt.close()

                    print(f"  Created: {output_path}")

        else:
            # Group by operation
            for op, dfs in ops.items():
                fig, ax = plt.subplots(figsize=(12, 7))

                all_algos = set()
                for df in dfs:
                    if "algorithm" in df.columns:
                        all_algos.update(df["algorithm"].unique())

                for algo in sorted(all_algos):
                    # Find data for this algorithm
                    num_gpus_list = []
                    metric_vals_list = []

                    for df in dfs:
                        if "algorithm" in df.columns and algo in df["algorithm"].values:
                            algo_df = df[df["algorithm"] == algo].sort_values(
                                "num_processes"
                            )
                            num_gpus_list.extend(self._get_num_processes(algo_df))
                            metric_vals_list.extend(self._get_metric(algo_df, metric))

                    if num_gpus_list:
                        idx = np.argsort(num_gpus_list)
                        num_gpus_sorted = [num_gpus_list[i] for i in idx]
                        metric_vals_sorted = [metric_vals_list[i] for i in idx]

                        color = ALGO_COLORS.get(algo, "#3498db")
                        marker = ALGO_MARKERS.get(algo, "o")

                        ax.plot(
                            num_gpus_sorted,
                            metric_vals_sorted,
                            marker=marker,
                            color=color,
                            linewidth=2,
                            markersize=8,
                            label=f'{algo.replace("_", " ").title()}',
                        )

                ax.set_xlabel("Number of GPUs/Processes", fontsize=12)
                ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
                ax.set_title(
                    f'Algorithm Comparison: {op.replace("_", " ").title()}', fontsize=14
                )
                ax.legend(fontsize=10, loc="best")
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                output_path = os.path.join(output_dir, f"algorithm_comparison_{op}.png")
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"  Created: {output_path}")

    def plot_bandwidth_vs_gpu(self, output_dir: str = "plots"):
        """Plot bandwidth vs GPU count with theoretical limits."""

        for exp_name, df in self.results_df.items():
            if df.empty or "bandwidth_gbps" not in df.columns:
                continue

            fig, ax = plt.subplots(1, 2, figsize=(16, 7))

            # Group by algorithm
            algorithms = self._get_algorithms(df)

            colors = list(ALGO_COLORS.values())[: len(algorithms)]

            for i, algo in enumerate(algorithms):
                algo_df = df[df["algorithm"] == algo].sort_values("num_processes")
                num_gpus = self._get_num_processes(algo_df)
                bandwidth = self._get_metric(algo_df, "bandwidth_gbps")

                color = colors[i % len(colors)]
                marker = ALGO_MARKERS.get(algo, "o")

                # Plot bandwidth
                ax[0].plot(
                    num_gpus,
                    bandwidth,
                    marker=marker,
                    color=color,
                    linewidth=2.5,
                    markersize=10,
                    label=f'{algo.replace("_", " ").title()}',
                    alpha=0.8,
                )

                # Calculate theoretical maximum (assume 25 GB/s per GPU for NVLink)
                if algo == algorithms[0]:
                    max_bandwidth = 25.0 * num_gpus
                    ax[1].plot(
                        num_gpus,
                        max_bandwidth,
                        "--",
                        color="red",
                        linewidth=2,
                        label="Theoretical Max (NVLink)",
                    )

                # Calculate actual utilization percentage
                if max_bandwidth[0] > 0:
                    utilization = (bandwidth / max_bandwidth) * 100
                    ax[1].plot(
                        num_gpus,
                        utilization,
                        marker=marker,
                        color=color,
                        linewidth=2.5,
                        markersize=10,
                        label=f'{algo.replace("_", " ").title()}',
                    )

            # Raw bandwidth plot
            ax[0].set_xlabel("Number of GPUs", fontsize=12)
            ax[0].set_ylabel("Bandwidth (GB/s)", fontsize=12)
            ax[0].set_title("Bandwidth Scaling vs GPU Count", fontsize=14)
            ax[0].legend(fontsize=10, loc="best")
            ax[0].grid(True, alpha=0.3)

            # Utilization plot
            ax[1].set_xlabel("Number of GPUs", fontsize=12)
            ax[1].set_ylabel("Bandwidth Utilization (%)", fontsize=12)
            ax[1].set_title("Bandwidth Efficiency vs GPU Count", fontsize=14)
            ax[1].legend(fontsize=10, loc="best")
            ax[1].set_ylim(0, 110)
            ax[1].grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = os.path.join(output_dir, f"bandwidth_vs_gpu_{exp_name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"  Created: {output_path}")

    def plot_speedup_plots(self, output_dir: str = "plots"):
        """Plot speedup vs single-process baseline."""

        for exp_name, df in self.results_df.items():
            if df.empty or "latency_ms" not in df.columns:
                continue

            # Get single process baseline
            single_df = df[df["num_processes"] == 1]
            if single_df.empty:
                continue

            baseline_latency = single_df["latency_ms"].iloc[0]

            # Group by algorithm
            algorithms = self._get_algorithms(df)

            fig, ax = plt.subplots(figsize=(12, 7))

            for algo in algorithms:
                algo_df = df[df["algorithm"] == algo].sort_values("num_processes")
                num_gpus = self._get_num_processes(algo_df)
                latencies = self._get_metric(algo_df, "latency_ms")

                # Calculate speedup
                speedup = baseline_latency / latencies

                color = ALGO_COLORS.get(algo, "#3498db")
                marker = ALGO_MARKERS.get(algo, "o")

                ax.plot(
                    num_gpus,
                    speedup,
                    marker=marker,
                    color=color,
                    linewidth=2.5,
                    markersize=10,
                    label=f'{algo.replace("_", " ").title()}',
                )

            # Add ideal speedup line
            num_gpus = self._get_num_processes(df[df["algorithm"] == algorithms[0]])
            ideal_speedup = num_gpus
            ax.plot(
                num_gpus,
                ideal_speedup,
                "--",
                color="red",
                linewidth=2,
                label="Ideal Speedup",
            )

            ax.set_xlabel("Number of GPUs", fontsize=12)
            ax.set_ylabel(
                f"Speedup (vs {baseline_latency:.3f}ms single GPU)", fontsize=12
            )
            ax.set_title(
                f'Speedup Analysis: {exp_name.replace("_", " ").title()}', fontsize=14
            )
            ax.legend(fontsize=10, loc="best")
            ax.grid(True, alpha=0.3)

            # Add efficiency annotation
            for algo in algorithms:
                algo_df = df[df["algorithm"] == algo].sort_values("num_processes")
                num_gpus_algo = self._get_num_processes(algo_df)
                latencies_algo = self._get_metric(algo_df, "latency_ms")
                speedup_algo = baseline_latency / latencies_algo

                if len(num_gpus_algo) > 1:
                    efficiency_at_max = speedup_algo[-1] / num_gpus_algo[-1]
                    ax.annotate(
                        f'{algo.replace("_", " ").title()}: '
                        f"{efficiency_at_max:.1%} eff",
                        xy=(num_gpus_algo[-1], speedup_algo[-1]),
                        textcoords="offset points",
                        xytext=(10, 10),
                        fontsize=9,
                        color=color,
                    )

            plt.tight_layout()
            output_path = os.path.join(output_dir, f"speedup_{exp_name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"  Created: {output_path}")

    def plot_scalability_heatmap(
        self, output_dir: str = "plots", metric: str = "bandwidth_gbps"
    ):
        """Create heatmap of performance across GPUs and algorithms."""

        # Collect all data
        data_matrix = []
        gpu_counts = set()
        algorithms = set()

        for exp_name, df in self.results_df.items():
            if df.empty or metric not in df.columns:
                continue

            for _, row in df.iterrows():
                gpu_counts.add(row["num_processes"])
                if "algorithm" in row:
                    algorithms.add(row["algorithm"])

        if not gpu_counts or not algorithms:
            return

        # Create matrix
        for algo in sorted(algorithms):
            row = []
            for gpu in sorted(gpu_counts):
                # Find matching data
                matches = df[(df["num_processes"] == gpu) & (df["algorithm"] == algo)]
                if not matches.empty:
                    row.append(matches[metric].iloc[0])
                else:
                    row.append(np.nan)
            data_matrix.append(row)

        if not data_matrix:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        data_array = np.array(data_matrix)

        # Handle NaN values for colormap
        masked_data = np.ma.masked_invalid(data_array)

        # Create custom colormap
        if metric == "bandwidth_gbps":
            cmap = plt.cm.viridis
            vmin, vmax = 0, np.nanmax(data_array) * 1.1
        elif metric == "scalability_efficiency":
            cmap = plt.cm.RdYlGn
            vmin, vmax = 0, 1.2
        else:
            cmap = plt.cm.viridis
            vmin, vmax = None, None

        im = ax.imshow(masked_data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        # Set ticks
        ax.set_xticks(range(len(sorted(gpu_counts))))
        ax.set_yticks(range(len(sorted(algorithms))))

        ax.set_xticklabels([f"{gpu}" for gpu in sorted(gpu_counts)])
        ax.set_yticklabels(
            [algo.replace("_", " ").title() for algo in sorted(algorithms)]
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        metric_name = metric.replace("_", " ").title()
        cbar.set_label(f"{metric_name}", fontsize=12)

        # Add annotations
        for i, algo in enumerate(sorted(algorithms)):
            for j, gpu in enumerate(sorted(gpu_counts)):
                value = data_array[i, j]
                if not np.isnan(value):
                    color = "white" if value > (vmax or 1) / 2 else "black"
                    ax.text(
                        j,
                        i,
                        (
                            f"{value:.2f}"
                            if metric != "scalability_efficiency"
                            else f"{value:.1%}"
                        ),
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=9,
                    )

        ax.set_xlabel("Number of GPUs/Processes", fontsize=12)
        ax.set_ylabel("Algorithm", fontsize=12)
        ax.set_title(f"Performance Heatmap: {metric_name}", fontsize=14)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"heatmap_{metric}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Created: {output_path}")

    def plot_throughput_vs_gpu(self, output_dir: str = "plots"):
        """Plot throughput vs GPU count."""

        for exp_name, df in self.results_df.items():
            if df.empty or "throughput_ops_per_sec" not in df.columns:
                continue

            algorithms = self._get_algorithms(df)

            fig, ax = plt.subplots(figsize=(12, 7))

            colors = list(ALGO_COLORS.values())[: len(algorithms)]

            for i, algo in enumerate(algorithms):
                algo_df = df[df["algorithm"] == algo].sort_values("num_processes")
                num_gpus = self._get_num_processes(algo_df)
                throughput = self._get_metric(algo_df, "throughput_ops_per_sec")

                color = colors[i % len(colors)]
                marker = ALGO_MARKERS.get(algo, "o")

                ax.plot(
                    num_gpus,
                    throughput,
                    marker=marker,
                    color=color,
                    linewidth=2.5,
                    markersize=10,
                    label=f'{algo.replace("_", " ").title()}',
                )

            ax.set_xlabel("Number of GPUs/Processes", fontsize=12)
            ax.set_ylabel(f"Throughput (Ops/sec)", fontsize=12)
            ax.set_title(
                f'Throughput Scaling: {exp_name.replace("_", " ").title()}', fontsize=14
            )
            ax.legend(fontsize=10, loc="best")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = os.path.join(output_dir, f"throughput_vs_gpu_{exp_name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"  Created: {output_path}")

    def plot_latency_vs_gpu(self, output_dir: str = "plots"):
        """Plot latency vs GPU count with error bars."""

        for exp_name, df in self.results_df.items():
            if df.empty or "latency_ms" not in df.columns:
                continue

            algorithms = self._get_algorithms(df)

            fig, ax = plt.subplots(figsize=(12, 7))

            colors = list(ALGO_COLORS.values())[: len(algorithms)]

            for i, algo in enumerate(algorithms):
                algo_df = df[df["algorithm"] == algo].sort_values("num_processes")
                num_gpus = self._get_num_processes(algo_df)
                latencies = self._get_metric(algo_df, "latency_ms")

                color = colors[i % len(colors)]
                marker = ALGO_MARKERS.get(algo, "o")

                # Plot mean
                ax.plot(
                    num_gpus,
                    latencies,
                    marker=marker,
                    color=color,
                    linewidth=2.5,
                    markersize=10,
                    label=f'{algo.replace("_", " ").title()}',
                )

            ax.set_xlabel("Number of GPUs/Processes", fontsize=12)
            ax.set_ylabel(f"Latency (ms)", fontsize=12)
            ax.set_title(
                f'Latency Scaling: {exp_name.replace("_", " ").title()}', fontsize=14
            )
            ax.legend(fontsize=10, loc="best")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            output_path = os.path.join(output_dir, f"latency_vs_gpu_{exp_name}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"  Created: {output_path}")

    def create_summary_report(self, output_dir: str = "plots"):
        """Create comprehensive summary report."""

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis("off")

        # Title
        ax.text(
            0.5,
            0.95,
            "MLX Distributed Scaling Benchmark Summary",
            ha="center",
            va="top",
            fontsize=20,
            fontweight="bold",
        )

        # Add key metrics
        y_pos = 0.85

        for exp_name, df in list(self.results_df.items())[:5]:  # Top 5 experiments
            if df.empty:
                continue

            ax.text(
                0.1,
                y_pos,
                f'{exp_name.replace("_", " ").title()}',
                fontsize=12,
                fontweight="bold",
            )

            # Find best algorithm
            if "bandwidth_gbps" in df.columns:
                best_idx = df["bandwidth_gbps"].idxmax()
            elif "throughput_ops_per_sec" in df.columns:
                best_idx = df["throughput_ops_per_sec"].idxmax()
            else:
                best_idx = df["latency_ms"].idxmin()

            best_row = df.loc[best_idx]

            metrics_text = (
                f"  • Best algorithm: {best_row['algorithm'].replace('_', ' ').title()}\n"
                f"  • Processes: {int(best_row['num_processes'])}\n"
                f"  • Best bandwidth: {best_row.get('bandwidth_gbps', 'N/A') or 'N/A'} GB/s\n"
                f"  • Scalability efficiency: {best_row.get('scalability_efficiency', 'N/A') or 'N/A'}"
            )

            ax.text(0.1, y_pos - 0.05, metrics_text, fontsize=10)
            y_pos -= 0.2

        # Add scaling characteristics
        ax.text(
            0.5,
            y_pos - 0.1,
            "Scaling Characteristics",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )

        y_pos -= 0.15

        for exp_name, df in list(self.results_df.items())[:3]:
            if df.empty or "scalability_efficiency" not in df.columns:
                continue

            eff = df["scalability_efficiency"].values
            valid_eff = eff[~np.isnan(eff)]

            if len(valid_eff) > 0:
                ax.text(0.1, y_pos, f"{exp_name}:", fontsize=10)
                ax.text(
                    0.25,
                    y_pos - 0.03,
                    f"Min: {valid_eff.min():.1%} | Max: {valid_eff.max():.1%} | "
                    f"Mean: {valid_eff.mean():.1%}",
                    fontsize=9,
                )
                y_pos -= 0.08

        # Footer
        ax.text(
            0.5,
            0.02,
            f'Generated: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            ha="center",
            va="bottom",
            fontsize=8,
            color="gray",
        )

        plt.tight_layout()
        output_path = os.path.join(output_dir, "summary_report.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"  Created: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="State-of-the-art plotting for MLX scaling benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all plots
    python plot_scaling.py --json scaling_benchmark.json
    
    # Specify output directory
    python plot_scaling.py --json scaling_benchmark.json --output plots/
    
    # Generate specific plot types
    python plot_scaling.py --json scaling_benchmark.json --type efficiency
    
    # Create summary report only
    python plot_scaling.py --json scaling_benchmark.json --summary
    
Note:
    Requires matplotlib, numpy, and pandas
        """,
    )

    parser.add_argument(
        "--json",
        "-j",
        type=str,
        required=True,
        help="Input JSON file with benchmark results",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots/)",
    )

    parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="all",
        choices=[
            "all",
            "scalability",
            "efficiency",
            "algorithm",
            "bandwidth",
            "speedup",
            "heatmap",
            "summary",
        ],
        help="Type of plot to generate (default: all)",
    )

    parser.add_argument(
        "--summary", action="store_true", help="Generate summary report only"
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="bandwidth_gbps",
        choices=[
            "bandwidth_gbps",
            "latency_ms",
            "throughput_ops_per_sec",
            "scalability_efficiency",
        ],
        help="Metric for plots (default: bandwidth_gbps)",
    )

    args = parser.parse_args()

    # Load data
    try:
        with open(args.json, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found: {args.json}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)

    # Create plotter
    plotter = ScalingPlotter(data)

    # Generate plots
    if args.summary:
        plotter.create_summary_report(args.output)
    elif args.type == "all":
        plotter.create_all_plots(args.output)
    else:
        # Generate specific plot type
        if args.type == "scalability":
            plotter.plot_scalability_curves(args.output, args.metric)
        elif args.type == "efficiency":
            plotter.plot_efficiency_scatter(args.output)
        elif args.type == "algorithm":
            plotter.plot_algorithm_comparison(args.output, args.metric)
        elif args.type == "bandwidth":
            plotter.plot_bandwidth_vs_gpu(args.output)
        elif args.type == "speedup":
            plotter.plot_speedup_plots(args.output)
        elif args.type == "heatmap":
            plotter.plot_scalability_heatmap(args.output, args.metric)
        else:
            # Throughput and latency
            if "throughput" in args.type:
                plotter.plot_throughput_vs_gpu(args.output)
            elif "latency" in args.type:
                plotter.plot_latency_vs_gpu(args.output)


if __name__ == "__main__":
    main()
