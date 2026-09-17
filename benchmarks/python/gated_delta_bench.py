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

# Skip any configuration whose checkpoint buffer alone would exceed this.
# The sequential VJP stores one state per timestep, so the buffer is
# B * Hv * T * Dv * Dk * 4 bytes and grows fast enough to get the process
# OOM-killed rather than raising.
MEM_BUDGET = 128 << 30  # bytes


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


def state_cache_bytes(B, T, Hv, Dk, Dv, chunked):
    """Size of the forward checkpoint buffer.

    The chunked path stores one state per chunk of 16, the sequential path one
    per timestep, so they differ by 16x. This is the dominant allocation and the
    reason large shapes need skipping.
    """
    n_states = (T + 15) // 16 if chunked else T
    return B * Hv * n_states * Dv * Dk * 4


def clear_env():
    """Reset every switch the dispatch reads, and release MLX's buffer pool.

    The pool is not what makes the big shapes fail (those requests are genuinely
    too large), but freeing it between configurations stops fragmentation from
    turning a feasible allocation into a failure. Resetting every variable on
    every configuration also prevents a stale value from silently measuring the
    same variant twice.
    """
    os.environ["GATED_DELTA_VJP_2PASS"] = "0"
    os.environ["GATED_DELTA_VJP_FALLBACK"] = "0"
    os.environ["GATED_DELTA_CHUNK"] = "0"
    os.environ["GATED_DELTA_VJP_CHUNK"] = "0"
    os.environ.pop("GATED_DELTA_DEBUG_SLOT", None)
    mx.clear_cache()


def run_many(
    n_runs=10,
    do_backward=False,
    B=1,
    T=1024,
    Hk=16,
    Hv=32,
    Dk=128,
    Dv=128,
    C=16,
    dtype=mx.float32,
):
    """Run the kernel n_runs times on fresh random inputs, without timing.

    Intended for external observers: powermetrics, Activity Monitor, Xcode GPU
    counters, or a sampling profiler that needs the process to stay busy long
    enough to collect samples. New inputs each iteration so nothing can be
    cached or constant-folded away.
    """
    clear_env()
    if do_backward:
        # Keep the forward fixed so only the VJP varies.
        os.environ["GATED_DELTA_CHUNK"] = "0"
        os.environ["GATED_DELTA_VJP_CHUNK"] = str(C)
    else:
        os.environ["GATED_DELTA_CHUNK"] = str(C)

    if do_backward:

        def f(q, k, v, g, b, h0):
            out, state = mx.fast.gated_delta_update(q, k, v, g, b, h0)
            return out.sum() + state.sum()

        fn = mx.grad(f, argnums=(0, 1, 2, 3, 4, 5))
    else:
        fn = mx.fast.gated_delta_update

    mode = "BACKWARD" if do_backward else "FORWARD"
    print(f"\n=== run_many: {mode} x{n_runs} ===")
    print(f"B={B} T={T} Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv} C={C} {dtype}", flush=True)

    for i in range(n_runs):
        q = mx.random.normal(shape=(B, T, Hk, Dk)).astype(dtype)
        k = mx.random.normal(shape=(B, T, Hk, Dk)).astype(dtype)
        k = k / (mx.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
        v = mx.random.normal(shape=(B, T, Hv, Dv)).astype(dtype)
        g = mx.sigmoid(mx.random.normal(shape=(B, T, Hv))).astype(dtype)
        b = mx.sigmoid(mx.random.normal(shape=(B, T, Hv))).astype(dtype)
        # state stays fp32
        h0 = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        mx.eval(q, k, v, g, b, h0)

        mx.eval(*fn(q, k, v, g, b, h0))
        mx.synchronize()
        print(f"  run {i + 1:>3}/{n_runs}", flush=True)
        mx.clear_cache()

    print("  done\n", flush=True)


def benchmark_shape(B, T, Hk, Hv, Dk, Dv, chunk_sizes, do_backward, do_autograd):
    mx.random.seed(42)
    q = mx.random.normal(shape=(B, T, Hk, Dk))
    k = mx.random.normal(shape=(B, T, Hk, Dk))
    k = k / (mx.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)
    v = mx.random.normal(shape=(B, T, Hv, Dv))
    g = mx.random.normal(shape=(B, T, Hv)) * 0.1 - 1.0
    b = mx.sigmoid(mx.random.normal(shape=(B, T, Hv)))
    mx.eval(q, k, v, g, b)

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

    # C = -1 selects the autodiff fallback: the primitive is decomposed into
    # elementary ops and differentiated by MLX, so no hand-written VJP runs.
    def time_one(C):
        clear_env()
        # Atomic accumulation only; the two-pass reduction stays off.
        os.environ["GATED_DELTA_VJP_FALLBACK"] = "1" if C < 0 else "0"
        if do_backward:
            # Keep the forward fixed so only the VJP varies with C.
            os.environ["GATED_DELTA_CHUNK"] = "0"
            os.environ["GATED_DELTA_VJP_CHUNK"] = str(max(C, 0))
        else:
            os.environ["GATED_DELTA_CHUNK"] = str(max(C, 0))
        h0 = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
        mx.eval(*fn(q, k, v, g, b, h0))  # prime: build/compile this variant
        return bench(runner, fn, q, k, v, g, b, h0) / denom * 1e3

    # Baseline is the autodiff fallback when asked for, otherwise sequential.
    ms_base = time_one(-1 if do_autograd else 0)

    speedups = []
    for C in chunk_sizes:
        if C == 0 and not do_autograd:
            continue  # already the baseline
        try:
            ms_c = time_one(C)
            speedups.append(ms_base / ms_c if ms_c > 0 else float("nan"))
        except Exception as ex:
            print(f"  chunk {C} failed: {ex}")
            speedups.append(float("nan"))

    return shape_str, f"{ms_base:.3f}", speedups, ms_base


def run_benchmark(
    run_full,
    to_csv=False,
    csv_path="benchmark_results.csv",
    do_backward=False,
    do_autograd=False,
):
    if run_full:
        Bs = [8, 16]
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

    if do_backward:
        chunk_sizes = [0, 16]
    else:
        chunk_sizes = [0, 8, 16]

    # With an autodiff baseline the sequential kernel becomes a variant column,
    # since it is no longer what everything is measured against.
    if do_autograd:
        variant_Cs = chunk_sizes
        base_col = "autodiff (ms)"
    else:
        variant_Cs = [C for C in chunk_sizes if C != 0]
        base_col = "time_seq (ms)"

    def col_name(C):
        return "seq (speedup)" if C == 0 else f"C={C} (speedup)"

    headers = ["B", "T", "Hk", "Hv", "Dk", "Dv", base_col] + [
        col_name(C) for C in variant_Cs
    ]

    col_widths = [6, 6, 6, 6, 6, 6, 15] + [25] * len(variant_Cs)
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)

    rows = []

    mode = "BACKWARD (vjp, atomic)" if do_backward else "FORWARD"
    if do_autograd:
        mode += " vs autodiff"
    print(f"\n=== {mode} ===")
    print(fmt.format(*headers))
    print("-" * (sum(col_widths)))

    for B, T, Hk, Hv, Dk, Dv in itertools.product(Bs, Ts, Hks, Hvs, Dks, Dvs):
        # The sequential checkpoint buffer dominates memory. Skip rather than
        # let the OS kill the process mid-sweep.
        need = state_cache_bytes(B, T, Hv, Dk, Dv, chunked=not do_backward)
        if need > MEM_BUDGET:
            print(
                f"  skip B={B} T={T}: checkpoints need "
                f"{need / 2**30:.1f} GiB > {MEM_BUDGET / 2**30:.0f} GiB",
                flush=True,
            )
            continue

        # The autodiff fallback unrolls the recurrence into T primitive ops and
        # keeps every intermediate alive, so it holds O(T) copies of the state.
        # It is a correctness reference, not a baseline worth measuring long.
        if do_autograd and T > 512:
            print(f"  skip B={B} T={T}: autodiff fallback unrolls {T} steps",
                  flush=True)
            continue

        try:
            shapes_s, base_time_s, speedups, base_time = benchmark_shape(
                B,
                T,
                Hk,
                Hv,
                Dk,
                Dv,
                chunk_sizes,
                do_backward=do_backward,
                do_autograd=do_autograd,
            )
        except Exception as ex:
            print(f"  B={B} T={T} failed: {ex}", flush=True)
            mx.clear_cache()
            continue

        row = [f"{B}", f"{T}", f"{Hk}", f"{Hv}", f"{Dk}", f"{Dv}", base_time_s]
        for speed in speedups:
            row.append(f"{(base_time / speed):<8.2f} ({speed:<5.2f}x)")

        print(fmt.format(*row), end="")
        print(f"{RESET}")

        rows.append(row)

        # Release everything this shape allocated before sizing up.
        mx.clear_cache()

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
    parser.add_argument(
        "--autograd",
        "-ag",
        action="store_true",
        help="use the autodiff fallback as the baseline instead of the "
        "sequential kernel",
    )
    parser.add_argument(
        "--many",
        "-m",
        action="store_true",
        help="run the kernel repeatedly on fresh random inputs, no timing, so "
        "external profilers have something to sample",
    )
    parser.add_argument("--runs", "-n", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="run --many in bfloat16 instead of float32",
    )
    parser.add_argument(
        "--mem_gb",
        type=float,
        default=24.0,
        help="skip shapes whose checkpoint buffer exceeds this many GiB",
    )
    parser.add_argument(
        "--mem_limit_gb",
        type=float,
        default=0.0,
        help="if set, cap MLX allocations so an OOM raises instead of the "
        "process being killed",
    )
    args = parser.parse_args()

    MEM_BUDGET = int(args.mem_gb * (1 << 30))
    if args.mem_limit_gb > 0:
        mx.set_memory_limit(int(args.mem_limit_gb * (1 << 30)))

    if args.many:
        run_many(
            n_runs=args.runs,
            do_backward=args.backward,
            B=args.batch,
            T=args.seqlen,
            C=args.chunk,
            dtype=mx.bfloat16 if args.bf16 else mx.float32,
        )
        exit()

    run_benchmark(
        args.full,
        to_csv=args.csv,
        csv_path=args.csv_out,
        do_backward=args.backward,
        do_autograd=args.autograd,
    )
