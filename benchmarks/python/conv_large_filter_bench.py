"""Compare large-filter convolution forward and input-gradient latency.

Select the MLX build with PYTHONPATH and rotate baseline/patched processes.
Example: python conv_large_filter_bench.py --output trial.json --compiled
Use --eval-batch 8 to amortize evaluation/synchronization over distinct weights.
Reported times are per convolution, including graph construction.
"""

import argparse
import hashlib
import json
import resource
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

CASES = (
    ("crepe_64", (2, 1024, 1), (16, 64, 1)),
    ("crepe_32", (2, 512, 16), (16, 32, 16)),
    ("batch_64", (32, 1024, 16), (32, 64, 16)),
    ("height_17", (2, 32, 48, 16), (16, 17, 3, 16)),
    ("width_17", (2, 32, 48, 16), (16, 3, 17, 16)),
    ("both_17", (2, 32, 32, 16), (16, 17, 17, 16)),
    ("control_16", (2, 512, 16), (16, 16, 16)),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=101)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--eval-batch", type=int, default=1)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or args.eval_batch < 1:
        parser.error("warmup, repeats and eval-batch must be positive")

    rng = np.random.default_rng(20261022)
    report = {
        "mlx": mx.__version__,
        "module": mx.__file__,
        "compiled": args.compiled,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "eval_batch": args.eval_batch,
        "cases": {},
    }
    for name, input_shape, weight_shape in CASES:
        x = mx.array(rng.standard_normal(input_shape).astype(np.float32))
        w = mx.array((rng.standard_normal(weight_shape) * 0.01).astype(np.float32))
        padding = tuple(k // 2 for k in weight_shape[1:-1])

        def forward(a, b):
            return mx.conv_general(a, b, padding=padding)

        y = forward(x, w)
        cotangent = mx.array((rng.standard_normal(y.shape) * 0.1).astype(np.float32))
        mx.eval(x, w, cotangent, y)
        forward_digest = hashlib.sha256(np.array(y).tobytes()).hexdigest()
        inputs = [(x, w)] + [
            (x, mx.contiguous(w + i * 1e-6)) for i in range(1, args.eval_batch)
        ]
        mx.eval(inputs)

        def input_gradient(a, b):
            _, (dx,) = mx.vjp(lambda v: forward(v, b), (a,), (cotangent,))
            return dx

        case = {
            "input_shape": input_shape,
            "weight_shape": weight_shape,
            "padding": padding,
            "forward_sha256": forward_digest,
            "stages": {},
        }
        for stage, fn in (("forward", forward), ("input_gradient", input_gradient)):
            if args.compiled:
                fn = mx.compile(fn)
            for _ in range(args.warmup):
                mx.eval([fn(a, b) for a, b in inputs])
            mx.synchronize()
            live = mx.get_active_memory()
            mx.reset_peak_memory()
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter_ns()
                values = [fn(a, b) for a, b in inputs]
                mx.eval(values)
                mx.synchronize()
                samples.append(
                    (time.perf_counter_ns() - start) / (1e6 * args.eval_batch)
                )
            case["stages"][stage] = {
                "median_ms": statistics.median(samples),
                "p10_ms": float(np.quantile(samples, 0.1)),
                "p90_ms": float(np.quantile(samples, 0.9)),
                "samples_ms": samples,
                "mlx_live_bytes": live,
                "mlx_peak_bytes": mx.get_peak_memory(),
                "process_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
                "nonfinite": sum(int(mx.sum(~mx.isfinite(v)).item()) for v in values),
            }
            if stage == "forward":
                case["stages"][stage]["sha256"] = hashlib.sha256(
                    np.array(values[0]).tobytes()
                ).hexdigest()
        report["cases"][name] = case
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    stage: values["median_ms"]
                    for stage, values in case["stages"].items()
                }
                for name, case in report["cases"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
