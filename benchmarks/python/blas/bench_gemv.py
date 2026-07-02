# Copyright © 2023 Apple Inc.

import argparse
import os
import subprocess
import time

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import torch

results_dir = "./results"

if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

try:
    device_name = (
        subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL
        )
        .decode("utf-8")
        .strip()
    )
except (subprocess.CalledProcessError, FileNotFoundError):
    device_name = "unknown"

if torch.backends.mps.is_available():
    torch_device = "mps"
    torch_sync = torch.mps.synchronize
elif torch.cuda.is_available():
    torch_device = "cuda"
    torch_sync = torch.cuda.synchronize
else:
    torch_device = "cpu"
    torch_sync = lambda: None

FULL_WARMUP = 5
FULL_ITER_BENCH = 50
FULL_ITER_FUNC = 20

QUICK_WARMUP = 2
QUICK_ITER_BENCH = 10
QUICK_ITER_FUNC = 5

FULL_OUT_VEC_SIZES = [128, 512, 2048, 4096]
FULL_IN_VEC_SIZES = [128, 512, 2048, 4096]
FULL_BENCHMARK_VECTOR_LENS = sorted(
    [(i + 1) * 4096 for i in range(8)][::2]
    + [(i + 1) * 4095 for i in range(8)][::2]
    + [(i + 1) * 4097 for i in range(8)][::2]
    + [64, 128, 512, 1024, 2048, 11008, 32000]
)

QUICK_OUT_VEC_SIZES = [512, 2048]
QUICK_IN_VEC_SIZES = [512, 2048]
QUICK_BENCHMARK_VECTOR_LENS = sorted([128, 1024, 4096, 11008])

N_warmup = FULL_WARMUP
N_iter_bench = FULL_ITER_BENCH
N_iter_func = FULL_ITER_FUNC


def bench(f, m, v):
    for i in range(N_warmup):
        f(m, v)
    torch_sync()

    s = time.perf_counter_ns()
    for i in range(N_iter_bench):
        f(m, v)
    e = time.perf_counter_ns()
    return (e - s) * 1e-9


def gemv_mlx(m, v):
    ys = []
    for i in range(N_iter_func):
        y = m @ v
        ys.append(y)
    mx.eval(ys)
    return ys


def gemv_t_mlx(m, v):
    ys = []
    for i in range(N_iter_func):
        y = v @ m
        ys.append(y)
    mx.eval(ys)
    return ys


@torch.no_grad()
def gemv_torch(m, v):
    ys = []
    for i in range(N_iter_func):
        y = m @ v
        ys.append(y)
    torch_sync()
    return ys


@torch.no_grad()
def gemv_t_torch(m, v):
    ys = []
    for i in range(N_iter_func):
        y = v @ m
        ys.append(y)
    torch_sync()
    return ys


def bench_lens(
    in_vec_len, out_vec_len, np_dtype, transpose=False, max_torch_elements=None
):
    shape_mat = (in_vec_len, out_vec_len) if transpose else (out_vec_len, in_vec_len)
    shape_vec = (1, in_vec_len) if transpose else (in_vec_len, 1)

    mat_npy = np.random.normal(0.0, 2.0 / in_vec_len, shape_mat).astype(np_dtype)
    vec_npy = np.random.normal(0.0, 2.0 / in_vec_len, shape_vec).astype(np_dtype)
    mat_mlx = mx.array(mat_npy)
    vec_mlx = mx.array(vec_npy)
    mat_trc = torch.from_numpy(mat_npy).to(torch_device)
    vec_trc = torch.from_numpy(vec_npy).to(torch_device)

    torch_sync()

    matrix_elements = in_vec_len * out_vec_len
    time_torch = None
    if max_torch_elements is None or matrix_elements <= max_torch_elements:
        time_torch = (
            bench(gemv_t_torch, mat_trc, vec_trc)
            if transpose
            else bench(gemv_torch, mat_trc, vec_trc)
        )
    time_mlx = (
        bench(gemv_t_mlx, mat_mlx, vec_mlx)
        if transpose
        else bench(gemv_mlx, mat_mlx, vec_mlx)
    )

    c_mlx = (
        np.asarray(vec_mlx @ mat_mlx) if transpose else np.asarray(mat_mlx @ vec_mlx)
    )
    c_npy = (vec_npy @ mat_npy) if transpose else (mat_npy @ vec_npy)

    if not np.allclose(c_mlx, c_npy, atol=2e-5):
        print(
            f"Failed at {shape_mat} [transpose = {transpose}] with max(|a - b|) = {np.max(np.abs(c_npy - c_mlx))}"
        )

    return time_mlx, time_torch


def get_gflop_count(in_vec_len, out_vec_len):
    return float(2.0 * N_iter_bench * N_iter_func * in_vec_len * out_vec_len) / float(
        1024**3
    )


def get_gbyte_size(in_vec_len, out_vec_len, np_dtype):
    n_elem = in_vec_len * out_vec_len + in_vec_len + out_vec_len
    item_size = np.dtype(np_dtype).itemsize
    return float(N_iter_bench * N_iter_func * n_elem * item_size) / float(1024**3)


def bench_with_in_len(
    ax, in_vec_len, out_vector_lens, dtype, transpose, max_torch_elements, verbose=False
):
    np_dtype = getattr(np, dtype)
    mlx_gb_s = []
    mlx_gflops = []
    pyt_gb_s = []
    pyt_gflops = []

    if verbose:
        print(f"  {'in':>5}, {'out':>5},   mlx_GB/s,  trc_GB/s,    diff")

    for out_vec_len in out_vector_lens:
        gflop_count = get_gflop_count(in_vec_len, out_vec_len)
        gbyte_size = get_gbyte_size(in_vec_len, out_vec_len, np_dtype)

        time_mlx, time_torch = bench_lens(
            in_vec_len,
            out_vec_len,
            np_dtype,
            transpose,
            max_torch_elements,
        )

        mlx_gb_s.append(gbyte_size / time_mlx)
        pyt_gb_s.append(np.nan if time_torch is None else gbyte_size / time_torch)

        mlx_gflops.append(gflop_count / time_mlx)
        pyt_gflops.append(np.nan if time_torch is None else gflop_count / time_torch)

        mlx_gb_s_value = gbyte_size / time_mlx
        if verbose:
            if time_torch is None:
                print(
                    f"  in={in_vec_len:5d}, out={out_vec_len:5d}, "
                    f"mlx={mlx_gb_s_value:7.2f} GB/s, torch=skipped"
                )
            else:
                pyt_gb_s_value = gbyte_size / time_torch
                print(
                    f"  in={in_vec_len:5d}, out={out_vec_len:5d}, "
                    f"mlx={mlx_gb_s_value:7.2f} GB/s, "
                    f"torch={pyt_gb_s_value:7.2f} GB/s, "
                    f"diff={mlx_gb_s_value/pyt_gb_s_value - 1:+.1%}"
                )

    if transpose:
        title = f"gemv_t ([1, {in_vec_len}] [{in_vec_len}, out_vec_len]) | {dtype}"
    else:
        title = f"gemv ([out_vec_len, {in_vec_len}] X [{in_vec_len}, 1] ) | {dtype}"

    ax.plot(out_vector_lens, mlx_gb_s, "tab:blue", label="MLX")
    ax.plot(out_vector_lens, pyt_gb_s, "tab:red", label="Torch")
    ax.set_title(title)
    ax.set(xlabel="out_vector_len", ylabel="Performance (GB/s)")
    ax.legend()


def bench_with_out_len(
    ax, out_vec_len, in_vector_lens, dtype, transpose, max_torch_elements, verbose=False
):
    np_dtype = getattr(np, dtype)
    mlx_gb_s = []
    mlx_gflops = []
    pyt_gb_s = []
    pyt_gflops = []

    if verbose:
        print(f"  {'in':>5}, {'out':>5},   mlx_GB/s,  trc_GB/s,    diff")

    for in_vec_len in in_vector_lens:
        gflop_count = get_gflop_count(in_vec_len, out_vec_len)
        gbyte_size = get_gbyte_size(in_vec_len, out_vec_len, np_dtype)

        time_mlx, time_torch = bench_lens(
            in_vec_len,
            out_vec_len,
            np_dtype,
            transpose,
            max_torch_elements,
        )

        mlx_gb_s.append(gbyte_size / time_mlx)
        pyt_gb_s.append(np.nan if time_torch is None else gbyte_size / time_torch)

        mlx_gflops.append(gflop_count / time_mlx)
        pyt_gflops.append(np.nan if time_torch is None else gflop_count / time_torch)

        mlx_gb_s_value = gbyte_size / time_mlx
        if verbose:
            if time_torch is None:
                print(
                    f"  in={in_vec_len:5d}, out={out_vec_len:5d}, "
                    f"mlx={mlx_gb_s_value:7.2f} GB/s, torch=skipped"
                )
            else:
                pyt_gb_s_value = gbyte_size / time_torch
                print(
                    f"  in={in_vec_len:5d}, out={out_vec_len:5d}, "
                    f"mlx={mlx_gb_s_value:7.2f} GB/s, "
                    f"torch={pyt_gb_s_value:7.2f} GB/s, "
                    f"diff={mlx_gb_s_value/pyt_gb_s_value - 1:+.1%}"
                )

    if transpose:
        title = f"([1, in_vec_len] [in_vec_len, {out_vec_len}])"
    else:
        title = f"([{out_vec_len}, in_vec_len] X [in_vec_len, 1] )"

    ax.plot(in_vector_lens, mlx_gb_s, "tab:blue", label="MLX")
    ax.plot(in_vector_lens, pyt_gb_s, "tab:red", label="Torch")
    ax.set_title(title)
    ax.set(xlabel="in_vector_len", ylabel="Performance (GB/s)")
    ax.legend()


def main():
    parser = argparse.ArgumentParser(description="Run gemv benchmarks")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fewer iterations and a reduced vector-length set.",
    )
    parser.add_argument(
        "--max-torch-elements",
        type=int,
        default=None,
        help="Skip PyTorch timing for cases where in_vec_len*out_vec_len exceeds this value.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-shape timing results.",
    )
    parser.add_argument(
        "--single-threaded",
        action="store_true",
        help="Set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 for single-threaded PyTorch/NumPy comparison.",
    )
    args = parser.parse_args()

    if args.single_threaded:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

    global N_warmup, N_iter_bench, N_iter_func
    if args.quick:
        N_warmup = QUICK_WARMUP
        N_iter_bench = QUICK_ITER_BENCH
        N_iter_func = QUICK_ITER_FUNC
        out_vec_sizes = QUICK_OUT_VEC_SIZES
        in_vec_sizes = QUICK_IN_VEC_SIZES
        benchmark_vector_lens = QUICK_BENCHMARK_VECTOR_LENS
    else:
        N_warmup = FULL_WARMUP
        N_iter_bench = FULL_ITER_BENCH
        N_iter_func = FULL_ITER_FUNC
        out_vec_sizes = FULL_OUT_VEC_SIZES
        in_vec_sizes = FULL_IN_VEC_SIZES
        benchmark_vector_lens = FULL_BENCHMARK_VECTOR_LENS

    for transpose in (False, True):
        for dtype in ("float32", "float16", "complex64"):
            op_name = "gemv_t" if transpose else "gemv"
            print(f"\n{'='*60}")
            print(f"{op_name} | {dtype} | device: {torch_device}")
            print(f"{'='*60}")

            fig, axs = plt.subplots(
                len(in_vec_sizes), 2, figsize=(8.5, 11), layout="constrained"
            )

            print(f"--- sweep out_vec_len (fixed in_vec_len) ---")
            for i, in_vec_len in enumerate(in_vec_sizes):
                bench_with_in_len(
                    axs[i][0],
                    in_vec_len,
                    benchmark_vector_lens,
                    dtype,
                    transpose,
                    args.max_torch_elements,
                    args.verbose,
                )

            print(f"--- sweep in_vec_len (fixed out_vec_len) ---")
            for i, out_vec_len in enumerate(out_vec_sizes):
                bench_with_out_len(
                    axs[i][1],
                    out_vec_len,
                    benchmark_vector_lens,
                    dtype,
                    transpose,
                    args.max_torch_elements,
                    args.verbose,
                )

            fig.suptitle(f"{device_name}: {dtype} {op_name}")
            fig.savefig(
                os.path.join(
                    results_dir,
                    f"{device_name.replace(' ', '_')}_{dtype}_{op_name}.pdf",
                )
            )
            plt.close(fig)


if __name__ == "__main__":
    main()
