# Copyright © 2026 Apple Inc.
"""Turn cpu_ops_bench.py JSON results into a markdown table and a chart.

Usage:
    python benchmarks/python/cpu_ops_report.py BEFORE.json [BEFORE2.json ...] \
        --after AFTER.json [--one-thread AFTER_1T.json] [--chart out.png]

When several BEFORE files are given (repeated baseline runs), each op uses
the run with the lowest median. That is the conservative choice: run-to-run
noise can only make the reported speedup smaller, never larger.
"""

import argparse
import json


def load(path):
    with open(path) as f:
        return json.load(f)


def best_of(runs):
    """For each op, keep the entry with the lowest median across runs."""
    out = {}
    for run in runs:
        for name, r in run.items():
            if name not in out or r["median_ms"] < out[name]["median_ms"]:
                out[name] = r
    return out


def fmt(r):
    return f"{r['median_ms']:.3f} [{r['min_ms']:.3f}..{r['max_ms']:.3f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("before", nargs="+")
    p.add_argument("--after", required=True)
    p.add_argument("--one-thread", default="")
    p.add_argument("--chart", default="")
    args = p.parse_args()
    before = best_of([load(f) for f in args.before])
    after = load(args.after)
    one = load(args.one_thread) if args.one_thread else None

    hdr = "| Op | Before (ms) | After (ms) | Speedup |"
    sep = "|---|---:|---:|---:|"
    if one:
        hdr += " After, 1 thread (ms) | Speedup, 1 thread |"
        sep += "---:|---:|"
    print(hdr)
    print(sep)
    names, speedups = [], []
    for name, b in before.items():
        if name not in after:
            continue
        a = after[name]
        s = b["median_ms"] / a["median_ms"]
        row = f"| {name} | {fmt(b)} | {fmt(a)} | {s:.2f}x |"
        if one:
            o = one[name]
            row += f" {o['median_ms']:.3f} | {b['median_ms'] / o['median_ms']:.2f}x |"
        print(row)
        names.append(name)
        speedups.append(s)

    if args.chart:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        order = sorted(range(len(names)), key=lambda i: speedups[i])
        fig, ax = plt.subplots(figsize=(9, 0.32 * len(names) + 1.2))
        ys = range(len(order))
        vals = [speedups[i] for i in order]
        colors = ["#2f7ed8" if v >= 1.1 else "#999999" for v in vals]
        ax.barh(list(ys), vals, color=colors)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([names[i] for i in order], fontsize=8)
        ax.set_xscale("log")
        ax.axvline(1.0, color="black", linewidth=0.8)
        for y, v in zip(ys, vals):
            ax.text(v * 1.05, y, f"{v:.1f}x", va="center", fontsize=7)
        ax.set_xlabel("Speedup vs upstream 96fbf67 (median time per call, log scale)")
        ax.set_title(
            "MLX CPU backend: perf/cpu vs upstream (8 vCPU Xeon VM, Linux, no GPU)"
        )
        ax.set_xlim(0.5, max(vals) * 2)
        fig.tight_layout()
        fig.savefig(args.chart, dpi=150)


if __name__ == "__main__":
    main()
