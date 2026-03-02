# Copyright © 2023-2024 Apple Inc.

import argparse
import csv
import json
import math
import time
from itertools import product
from pathlib import Path

import mlx.core as mx

DTYPE_MAP = {
    "float16": mx.float16,
    "float32": mx.float32,
    "bfloat16": mx.bfloat16,
}


def parse_int_list(value):
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def parse_dtype_list(value):
    values = [v.strip() for v in value.split(",") if v.strip()]
    invalid = [v for v in values if v not in DTYPE_MAP]
    if invalid:
        allowed = ", ".join(sorted(DTYPE_MAP))
        raise ValueError(f"Unsupported dtype(s): {invalid}. Allowed: {allowed}")
    return tuple(values)


def benchmark_runtime(fn, warmup, iters):
    for _ in range(warmup):
        mx.eval(fn())

    tic = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    toc = time.perf_counter()
    return (toc - tic) * 1000.0 / iters


def make_tanh_rnn_workload(x, w_in, w_rec, bias):
    def run():
        batch_size = x.shape[0]
        hidden_size = w_rec.shape[0]
        hidden = mx.zeros((batch_size, hidden_size), dtype=x.dtype)
        outputs = []
        for t in range(x.shape[1]):
            hidden = mx.tanh(x[:, t, :] @ w_in + hidden @ w_rec + bias)
            outputs.append(hidden)
        return mx.stack(outputs, axis=1)

    return run


def make_lif_workload(x, w_in, w_rec, leak, threshold):
    leak_value = mx.array(leak, dtype=x.dtype)
    threshold_value = mx.array(threshold, dtype=x.dtype)
    one = mx.array(1.0, dtype=x.dtype)

    def run():
        batch_size = x.shape[0]
        hidden_size = w_rec.shape[0]
        membrane = mx.zeros((batch_size, hidden_size), dtype=x.dtype)
        spikes = mx.zeros((batch_size, hidden_size), dtype=x.dtype)
        outputs = []
        for t in range(x.shape[1]):
            current = x[:, t, :] @ w_in + spikes @ w_rec
            membrane = leak_value * membrane + current
            spikes = (membrane >= threshold_value).astype(x.dtype)
            membrane = membrane * (one - spikes)
            outputs.append(spikes)
        return mx.stack(outputs, axis=1)

    return run


def run_case(batch_size, sequence_length, input_size, hidden_size, dtype_name, args):
    dtype = DTYPE_MAP[dtype_name]
    scale = 1.0 / math.sqrt(hidden_size)

    x = mx.random.normal((batch_size, sequence_length, input_size)).astype(dtype)
    w_in = mx.random.uniform(
        low=-scale, high=scale, shape=(input_size, hidden_size)
    ).astype(dtype)
    w_rec = mx.random.uniform(
        low=-scale, high=scale, shape=(hidden_size, hidden_size)
    ).astype(dtype)
    bias = mx.random.uniform(low=-scale, high=scale, shape=(hidden_size,)).astype(dtype)
    mx.eval(x, w_in, w_rec, bias)

    workloads = [
        ("rnn_tanh_unrolled", make_tanh_rnn_workload(x, w_in, w_rec, bias)),
        (
            "lif_hard_reset_unrolled",
            make_lif_workload(x, w_in, w_rec, args.leak, args.threshold),
        ),
    ]

    rows = []
    for workload_name, workload_fn in workloads:
        runtime_ms = benchmark_runtime(workload_fn, args.warmup, args.iters)
        effective_steps_per_second = (batch_size * sequence_length) / (
            runtime_ms / 1000.0
        )
        rows.append(
            {
                "workload": workload_name,
                "batch_size": batch_size,
                "sequence_length": sequence_length,
                "input_size": input_size,
                "hidden_size": hidden_size,
                "dtype": dtype_name,
                "runtime_ms": runtime_ms,
                "effective_steps_per_second": effective_steps_per_second,
            }
        )

    return rows


def write_json(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def write_csv(path, rows):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [
        "workload",
        "batch_size",
        "sequence_length",
        "input_size",
        "hidden_size",
        "dtype",
        "runtime_ms",
        "effective_steps_per_second",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    print(
        "workload,dtype,batch_size,sequence_length,input_size,hidden_size,runtime_ms,steps_per_second"
    )
    for row in rows:
        print(
            f"{row['workload']},{row['dtype']},{row['batch_size']},{row['sequence_length']},{row['input_size']},{row['hidden_size']},{row['runtime_ms']:.6f},{row['effective_steps_per_second']:.2f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark unrolled recurrent and LIF-style workloads in MLX."
    )
    parser.add_argument("--batch-sizes", default=(1, 8, 32), type=parse_int_list)
    parser.add_argument(
        "--sequence-lengths", default=(32, 128, 256), type=parse_int_list
    )
    parser.add_argument("--hidden-sizes", default=(64, 256, 512), type=parse_int_list)
    parser.add_argument("--input-size", default=40, type=int)
    parser.add_argument(
        "--dtypes",
        default=("float16", "float32"),
        type=parse_dtype_list,
    )
    parser.add_argument("--warmup", default=10, type=int)
    parser.add_argument("--iters", default=100, type=int)
    parser.add_argument("--leak", default=0.95, type=float)
    parser.add_argument("--threshold", default=1.0, type=float)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--csv-output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []

    for batch_size, sequence_length, hidden_size, dtype_name in product(
        args.batch_sizes,
        args.sequence_lengths,
        args.hidden_sizes,
        args.dtypes,
    ):
        rows.extend(
            run_case(
                batch_size=batch_size,
                sequence_length=sequence_length,
                input_size=args.input_size,
                hidden_size=hidden_size,
                dtype_name=dtype_name,
                args=args,
            )
        )

    print_summary(rows)

    if args.json_output:
        write_json(args.json_output, rows)
    if args.csv_output:
        write_csv(args.csv_output, rows)


if __name__ == "__main__":
    main()
