# Copyright © 2026 Apple Inc.

import argparse
import json
import statistics
import time

import mlx.core as mx


def benchmark(fn, args, warmup, iterations):
    for _ in range(warmup):
        mx.eval(fn(*args))
    samples = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        mx.eval(fn(*args))
        samples.append((time.perf_counter_ns() - start) / 1000)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float32"
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 262144])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()
    results = []
    with mx.stream(getattr(mx, args.device)):
        for size in args.sizes:
            a = mx.linspace(-8.0, 8.0, size).astype(getattr(mx, args.dtype))
            b = -a
            mx.eval(a, b)
            for compiled in (False, True):
                functions = {
                    "first_vjp": (mx.grad(lambda a: mx.logaddexp(a, b).sum()), (a,)),
                    "second_vjp": (mx.grad(lambda b: mx.logaddexp(a, b).sum()), (b,)),
                    "both_vjp": (
                        mx.grad(lambda a, b: mx.logaddexp(a, b).sum(), argnums=(0, 1)),
                        (a, b),
                    ),
                    "both_jvp": (
                        lambda a, b: mx.jvp(
                            mx.logaddexp, [a, b], [mx.ones_like(a), mx.ones_like(b)]
                        )[1],
                        (a, b),
                    ),
                }
                for operation, (fn, inputs) in functions.items():
                    if compiled:
                        fn = mx.compile(fn)
                    elapsed = benchmark(fn, inputs, args.warmup, args.iterations)
                    row = dict(
                        device=args.device,
                        dtype=args.dtype,
                        size=size,
                        compiled=compiled,
                        operation=operation,
                        median_us=elapsed,
                    )
                    results.append(row)
                    print(json.dumps(row), flush=True)
    if args.output:
        with open(args.output, "w") as output:
            json.dump(dict(version=mx.__version__, results=results), output, indent=2)


if __name__ == "__main__":
    main()
