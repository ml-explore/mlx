# Copyright © 2026 Apple Inc.

"""Benchmark the quantized matmul row-dispatch boundary."""

import argparse
import random
import statistics
import time

import mlx.core as mx


def qmm(x, weight, scales, biases):
    return mx.quantized_matmul(
        x,
        weight,
        scales,
        biases,
        transpose=True,
        group_size=64,
        bits=4,
        mode="affine",
    )


def chunked_qmm(x, weight, scales, biases, chunk_size):
    return mx.concatenate(
        [
            qmm(x[start : start + chunk_size], weight, scales, biases)
            for start in range(0, x.shape[0], chunk_size)
        ],
        axis=0,
    )


def time_call(operation):
    start = time.perf_counter_ns()
    mx.eval(operation())
    return time.perf_counter_ns() - start


def benchmark(rows, weight, scales, biases, chunk_size, warmups, trials, seed):
    x = mx.random.normal((rows, weight.shape[1] * 8)).astype(mx.bfloat16)
    mx.eval(x)
    operations = {
        "direct": lambda: qmm(x, weight, scales, biases),
        "chunked": lambda: chunked_qmm(x, weight, scales, biases, chunk_size),
    }
    for _ in range(warmups):
        for operation in operations.values():
            mx.eval(operation())

    direct = operations["direct"]()
    chunked = operations["chunked"]()
    mx.eval(direct, chunked)
    if not bool(mx.allclose(direct, chunked, rtol=0.01, atol=0.06).item()):
        raise RuntimeError(f"row {rows} direct/chunked comparison failed")

    generator = random.Random(seed + rows)
    timings = []
    for _ in range(trials):
        order = ["direct", "chunked"]
        generator.shuffle(order)
        sample = {}
        for name in order:
            sample[name] = time_call(operations[name])
        timings.append(sample)

    direct_ns = statistics.median(sample["direct"] for sample in timings)
    chunked_ns = statistics.median(sample["chunked"] for sample in timings)
    ratios = [sample["chunked"] / sample["direct"] for sample in timings]
    return direct_ns / 1e6, chunked_ns / 1e6, statistics.median(ratios)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[5, 6, 8, 12])
    parser.add_argument("--input-dims", type=int, default=5376)
    parser.add_argument("--output-dims", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    mx.random.seed(args.seed)
    dense_weight = mx.random.normal((args.output_dims, args.input_dims)).astype(
        mx.float32
    )
    weight, scales, biases = mx.quantize(
        dense_weight, group_size=64, bits=4, mode="affine"
    )
    mx.eval(weight, scales, biases)
    del dense_weight

    print("rows  direct_ms  chunked_ms  chunked/direct")
    for rows in args.rows:
        direct_ms, chunked_ms, ratio = benchmark(
            rows,
            weight,
            scales,
            biases,
            args.chunk_size,
            args.warmups,
            args.trials,
            args.seed,
        )
        print(f"{rows:4d}  {direct_ms:9.4f}  {chunked_ms:10.4f}  {ratio:14.4f}")


if __name__ == "__main__":
    main()
