# Copyright © 2026 Apple Inc.

"""Run sequential, alternating A/B measurements or summarize saved samples."""

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path


def summarize(data):
    keys = sorted(data["parent"], key=lambda key: tuple(map(int, key.split("x"))))
    n = data["reps"]
    if n < 2 or set(keys) != set(data["pr"]):
        raise ValueError(
            "At least two paired repetitions with matching shapes required"
        )
    for arm in ("parent", "pr"):
        for key in keys:
            samples = data[arm][key]
            if len(samples) != n or any(
                not math.isfinite(x) or x <= 0 for x in samples
            ):
                raise ValueError(
                    "Expected positive finite timing samples for every pair"
                )

    def estimates(indices):
        return [
            statistics.median(data["parent"][key][i] for i in indices)
            / statistics.median(data["pr"][key][i] for i in indices)
            for key in keys
        ]

    point = estimates(range(n))
    rng = random.Random(0)
    boot = [estimates(rng.choices(range(n), k=n)) for _ in range(5000)]

    def interval(values):
        values = sorted(values)
        return values[int(0.025 * len(values))], values[int(0.975 * len(values))]

    rows = []
    for j, key in enumerate(keys):
        p, r = data["parent"][key], data["pr"][key]
        rows.append(
            {
                "shape": key,
                "parent_us": statistics.median(p),
                "pr_us": statistics.median(r),
                "speedup": point[j],
                "bootstrap_95": interval(row[j] for row in boot),
                "pr_faster_pairs": sum(a > b for a, b in zip(p, r)),
            }
        )
    return {
        "dtype": data["dtype"],
        "reps": n,
        "rows": rows,
        "geomean": statistics.geometric_mean(point),
        "geomean_bootstrap_95": interval(
            statistics.geometric_mean(row) for row in boot
        ),
        "method": (
            "Ratio of per-arm medians; 5000 paired bootstrap resamples, seed 0. "
            "Resample whole repetitions jointly across shapes. All samples retained. "
            "Exploratory percentile intervals assume exchangeable repetitions; "
            "they do not account for temporal drift or multiple comparisons."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, help="Summarize an existing raw JSON file"
    )
    parser.add_argument("--parent-pythonpath")
    parser.add_argument("--pr-pythonpath")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--out", type=Path, help="Save raw samples when running A/B")
    args = parser.parse_args()
    if args.input:
        data = json.loads(args.input.read_text())
    else:
        if not args.parent_pythonpath or not args.pr_pythonpath or not args.out:
            parser.error("A/B requires both --*-pythonpath arguments and --out")
        if args.reps < 2:
            parser.error("--reps must be at least 2")
        bench = Path(__file__).with_name("gemv_m1_bench.py")
        data = {
            "dtype": args.dtype,
            "reps": args.reps,
            "parent": {},
            "pr": {},
            "invocations": [],
        }
        for rep in range(args.reps):
            order = ("parent", "pr") if rep % 2 == 0 else ("pr", "parent")
            for arm in order:
                env = dict(os.environ, PYTHONPATH=getattr(args, f"{arm}_pythonpath"))
                command = [
                    args.python,
                    str(bench),
                    "--dtype",
                    args.dtype,
                    "--rounds",
                    "9",
                    "--iters",
                    "64",
                    "--warmup",
                    "6",
                ]
                run = subprocess.run(
                    command, env=env, text=True, capture_output=True, check=True
                )
                result = json.loads(run.stdout)
                data["invocations"].append({"rep": rep, "arm": arm, "result": result})
                for row in result["results"]:
                    key = f"{row['N']}x{row['K']}"
                    data[arm].setdefault(key, []).append(row["us"])
                args.out.write_text(json.dumps(data, indent=2) + "\n")
                print(f"pair {rep + 1}/{args.reps}: {arm} complete", file=sys.stderr)
    print(json.dumps(summarize(data), indent=2))


if __name__ == "__main__":
    main()
