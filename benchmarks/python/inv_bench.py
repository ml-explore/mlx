# Copyright © 2026 Apple Inc.

import argparse
import platform
import time

import mlx.core as mx
import numpy as np


def make_input(order, batch_size, seed):
    rng = np.random.default_rng(seed)
    shape = (order, order) if batch_size == 1 else (batch_size, order, order)
    matrix = rng.standard_normal(shape, dtype=np.float32)
    return matrix + order * np.eye(order, dtype=np.float32)


def array_on_device(matrix, device):
    previous_device = mx.default_device()
    mx.set_default_device(device)
    try:
        result = mx.array(matrix)
        mx.eval(result)
        mx.synchronize(device)
    finally:
        mx.set_default_device(previous_device)
    return result


def benchmark(a, device, warmup, iters):
    for _ in range(warmup):
        output = mx.linalg.inv(a, stream=device)
        mx.eval(output)
        mx.synchronize(device)

    samples = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        output = mx.linalg.inv(a, stream=device)
        mx.eval(output)
        mx.synchronize(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return float(np.median(samples))


def print_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row):
        return (
            "| "
            + " | ".join(f"{cell:<{widths[index]}}" for index, cell in enumerate(row))
            + " |"
        )

    print(format_row(headers))
    print("|-" + "-|-".join("-" * width for width in widths) + "-|")
    for row in rows:
        print(format_row(row))


def main():
    parser = argparse.ArgumentParser(
        description="Compare mx.linalg.inv on the CPU and Metal backends."
    )
    parser.add_argument("--sizes", default="1,3,16,64,256,1024")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not mx.metal.is_available():
        raise RuntimeError("This benchmark requires a Metal device.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")

    orders = [int(value) for value in args.sizes.split(",")]
    if any(order < 1 for order in orders):
        raise ValueError("--sizes must contain positive matrix orders.")

    print(
        f"machine={platform.machine()} system={platform.platform()} "
        f"dtype=float32 batch_size={args.batch_size} warmup={args.warmup} "
        f"iters={args.iters}"
    )

    rows = []
    for index, order in enumerate(orders):
        matrix = make_input(order, args.batch_size, args.seed + index)
        cpu_time = benchmark(
            array_on_device(matrix, mx.cpu), mx.cpu, args.warmup, args.iters
        )
        metal_time = benchmark(
            array_on_device(matrix, mx.gpu), mx.gpu, args.warmup, args.iters
        )
        shape = f"{order}x{order}"
        if args.batch_size > 1:
            shape = f"{args.batch_size}x{shape}"
        rows.append(
            [
                shape,
                f"{cpu_time:.3f}",
                f"{metal_time:.3f}",
                f"{cpu_time / metal_time:.2f}x",
            ]
        )

    print_table(["Shape", "CPU ms", "Metal ms", "CPU / Metal"], rows)


if __name__ == "__main__":
    main()
