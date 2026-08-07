import argparse
import csv
import itertools
import os
import time
from datetime import datetime
from typing import Optional, Tuple

import mlx.core as mx
import numpy as np

RED_BOLD = "\033[1;31m"
GREEN = "\033[0;32m"
RESET = "\033[0m"


N_warmup = 8
N_iter_bench = 80
N_iter_func = 5


# similar to ./blas/bench_gemm.py
def bench(f, *args):
    for _ in range(N_warmup):
        f(*args)
    mx.synchronize()

    s = time.perf_counter_ns()
    for _ in range(N_iter_bench):
        f(*args)
    mx.synchronize()
    e = time.perf_counter_ns()
    return (e - s) * 1e-9  # total seconds for N_iter_bench * N_iter_func calls


def do_kernel_bench(f, *args):
    ys = []
    for _ in range(N_iter_func):
        out, hf = f(*args)
        ys.append(out)
        ys.append(hf)
    mx.eval(ys)
    return ys


def make_grad_fn():
    def f(q, k, v, g, b, h0):
        out, state = mx.fast.gated_delta_update(q, k, v, g, b, h0)
        return out.sum() + state.sum()

    return mx.grad(f, argnums=(0, 1, 2, 3, 4, 5))


def do_grad_bench(f, *args):
    ys = []
    for _ in range(N_iter_func):
        ys.extend(f(*args))
    mx.eval(ys)
    return ys


def benchmark_shape(B, T, Hk, Hv, Dk, Dv, chunk_sizes, do_backward):
    mx.random.seed(42)
    q = mx.random.normal(shape=(B, T, Hk, Dk))
    k = mx.random.normal(shape=(B, T, Hk, Dk))
    k = k / (mx.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    v = mx.random.normal(shape=(B, T, Hv, Dv))
    g = mx.random.normal(shape=(B, T, Hv)) * 0.1 - 1.0
    b = mx.sigmoid(mx.random.normal(shape=(B, T, Hv)))

    shape_str = f"B={B} T={T} Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv}"
    denom = N_iter_bench * N_iter_func

    if do_backward:

        def f(q, k, v, g, b, h0):
            out, state = mx.fast.gated_delta_update(q, k, v, g, b, h0)
            return out.sum() + state.sum()

        fn = mx.grad(f, argnums=(0, 1, 2, 3, 4, 5))
        runner = do_grad_bench

    else:
        fn = mx.fast.gated_delta_update
        runner = do_kernel_bench

    def time_one(C):
        os.environ["GATED_DELTA_CHUNK"] = "0"
        h0 = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        mx.eval(*fn(q, k, v, g, b, h0))
        return bench(runner, fn, q, k, v, g, b, h0) / denom * 1e3

    ms_seq = time_one(0)

    speedups = []
    for C in (c for c in chunk_sizes if c != 0):
        try:
            ms_c = time_one(C)
            speedups.append(ms_seq / ms_c if ms_c > 0 else float("nan"))
        except Exception as ex:
            print(f"  chunk {C} failed: {ex}")
            speedups.append(float("nan"))

    return shape_str, f"{ms_seq:.3f}", speedups, ms_seq


def run_benchmark(
    run_full, to_csv=False, csv_path="benchmark_results.csv", do_backward=False
):
    if run_full:
        Bs = [1, 4, 8, 16]
        Ts = [8, 64, 256, 512, 1024, 2048, 4096]
        Hks = [16]
        Hvs = [32]
        Dks = [128]
        Dvs = [128]
    else:
        Bs = [1, 8, 16]
        Ts = [8, 512, 1024, 2048]
        Hks = [16]
        Hvs = [32]
        Dks = [128]
        Dvs = [128]

    chunk_sizes = [0, 8, 16]
    non_zero_Cs = [C for C in chunk_sizes if C != 0]

    headers = ["B", "T", "Hk", "Hv", "Dk", "Dv", "time_seq (ms)"] + [
        f"C={C} (speedup)" for C in non_zero_Cs
    ]

    col_widths = [6, 6, 6, 6, 6, 6, 15] + [25] * (len(non_zero_Cs))
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)

    rows = []

    print(fmt.format(*headers))
    print("-" * (sum(col_widths)))

    for B, T, Hk, Hv, Dk, Dv in itertools.product(Bs, Ts, Hks, Hvs, Dks, Dvs):
        shapes_s, base_time_s, speedups, base_time = benchmark_shape(
            B, T, Hk, Hv, Dk, Dv, chunk_sizes, do_backward=do_backward
        )
        row = [f"{B}", f"{T}", f"{Hk}", f"{Hv}", f"{Dk}", f"{Dv}", base_time_s]
        for speed in speedups:
            row.append(f"{(base_time / speed):<8.2f} ({speed:<5.2f}x)")

        print(fmt.format(*row), end="")
        print(f"{RESET}")

        rows.append(row)

    if to_csv:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        print(f"\nResults also written to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gated delta benchmark")
    parser.add_argument("--full", "-f", action="store_true")
    parser.add_argument("--csv", "-c", action="store_true")
    parser.add_argument("--csv_out", "-co", default="benchmark_results.csv")
    parser.add_argument("--backward", "-bw", action="store_true")
    args = parser.parse_args()

    run_benchmark(
        args.full, to_csv=args.csv, csv_path=args.csv_out, do_backward=args.backward
    )
