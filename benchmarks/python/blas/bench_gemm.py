# Copyright © 2023 Apple Inc.

import argparse
import math
import os
import subprocess
import time

import mlx.core as mx
import numpy as np
import torch

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

FULL_WARMUP = 8
FULL_ITER_BENCH = 80
FULL_ITER_FUNC = 5

QUICK_WARMUP = 2
QUICK_ITER_BENCH = 10
QUICK_ITER_FUNC = 5

N_warmup = FULL_WARMUP
N_iter_bench = FULL_ITER_BENCH
N_iter_func = FULL_ITER_FUNC


def bench(f, a, b):
    for i in range(N_warmup):
        f(a, b)
    torch_sync()

    s = time.perf_counter_ns()
    for i in range(N_iter_bench):
        f(a, b)
    e = time.perf_counter_ns()
    return (e - s) * 1e-9


def gemm_nn_mlx(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a @ b
        ys.append(y)
    mx.eval(ys)
    return ys


def gemm_nt_mlx(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a @ b.transpose((0, 2, 1))
        ys.append(y)
    mx.eval(ys)
    return ys


def gemm_tn_mlx(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a.transpose((0, 2, 1)) @ b
        ys.append(y)
    mx.eval(ys)
    return ys


def gemm_tt_mlx(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a.transpose((0, 2, 1)) @ b.transpose((0, 2, 1))
        ys.append(y)
    mx.eval(ys)
    return ys


@torch.no_grad()
def gemm_nn_torch(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a @ b
        ys.append(y)
    torch_sync()
    return ys


@torch.no_grad()
def gemm_nt_torch(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a @ b.transpose(-1, -2)
        ys.append(y)
    torch_sync()
    return ys


@torch.no_grad()
def gemm_tn_torch(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a.transpose(-1, -2) @ b
        ys.append(y)
    torch_sync()
    return ys


@torch.no_grad()
def gemm_tt_torch(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a.transpose(-1, -2) @ b.transpose(-1, -2)
        ys.append(y)
    torch_sync()
    return ys


def bench_shape(B, M, N, K, np_dtype, transpose="nn", max_torch_ops=None):
    shape_a = (B, M, K) if transpose[0] == "n" else (B, K, M)
    shape_b = (B, K, N) if transpose[1] == "n" else (B, N, K)

    a_np = np.random.normal(0.0, 1.0 / math.sqrt(M + K), shape_a).astype(np_dtype)
    b_np = np.random.normal(0.0, 1.0 / math.sqrt(N + K), shape_b).astype(np_dtype)

    a_mx = mx.array(a_np)
    b_mx = mx.array(b_np)

    a_pt = torch.from_numpy(a_np).to(torch_device)
    b_pt = torch.from_numpy(b_np).to(torch_device)

    torch_sync()

    f_mx = {
        "nn": gemm_nn_mlx,
        "nt": gemm_nt_mlx,
        "tn": gemm_tn_mlx,
        "tt": gemm_tt_mlx,
    }[transpose]

    f_pt = {
        "nn": gemm_nn_torch,
        "nt": gemm_nt_torch,
        "tn": gemm_tn_torch,
        "tt": gemm_tt_torch,
    }[transpose]

    gemm_ops = B * M * N * K
    time_torch = None
    if max_torch_ops is None or gemm_ops <= max_torch_ops:
        time_torch = bench(f_pt, a_pt, b_pt)

    time_mlx = bench(f_mx, a_mx, b_mx)

    t_a = (0, 1, 2) if transpose[0] == "n" else (0, 2, 1)
    t_b = (0, 1, 2) if transpose[1] == "n" else (0, 2, 1)

    c_mlx = a_mx.transpose(t_a) @ b_mx.transpose(t_b)
    c_npy = a_np.transpose(t_a).astype(np_dtype) @ b_np.transpose(t_b).astype(np_dtype)

    atol = 1e-5 if np_dtype == np.float32 else 1e-4

    if not np.allclose(c_mlx, c_npy.astype(np_dtype), atol=atol):
        print(
            f"Failed at {(B, M, N, K)} [transpose = {transpose}] with max(|a - b|) = {np.max(np.abs(c_npy - c_mlx))}"
        )

    return time_mlx, time_torch


def get_gflop_count(B, M, N, K):
    return float(2.0 * N_iter_bench * N_iter_func * B * M * N * K) / float(1024.0**3)


def main():
    global N_warmup, N_iter_bench, N_iter_func

    parser = argparse.ArgumentParser(description="Run gemm benchmarks")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run fewer iterations and a reduced shape set.",
    )
    parser.add_argument(
        "--max-torch-ops",
        type=int,
        default=None,
        help="Skip PyTorch timing for cases where B*M*N*K exceeds this value.",
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

    if args.quick:
        N_warmup = QUICK_WARMUP
        N_iter_bench = QUICK_ITER_BENCH
        N_iter_func = QUICK_ITER_FUNC
    else:
        N_warmup = FULL_WARMUP
        N_iter_bench = FULL_ITER_BENCH
        N_iter_func = FULL_ITER_FUNC

    dtypes = ("float32", "float16", "complex64")
    transposes = ("nn", "nt", "tn")
    if args.quick:
        shapes = (
            (16, 234, 768, 3072),
            (1, 1024, 1024, 2048),
        )
    else:
        shapes = (
            (16, 234, 768, 3072),
            (1, 64, 64, 25344),
            (16, 1024, 1024, 1024),
            (1, 1024, 1024, 2048),
            (4, 1024, 1024, 4096),
            (4, 1024, 4096, 1024),
            (1, 4096, 4096, 4096),
        )

    if args.verbose:
        print(
            f"{'B':>3}, {'M':>4}, {'N':>4}, {'K':>4}, {'dtype':<9}, {'t':<2},  torch_gf,   mlx_gf,     diff"
        )
        print("-" * 66)

    for dtype in dtypes:
        for transpose in transposes:
            for B, M, N, K in shapes:
                np_dtype = getattr(np, dtype)
                time_mlx, time_torch = bench_shape(
                    B,
                    M,
                    N,
                    K,
                    np_dtype,
                    transpose,
                    args.max_torch_ops,
                )

                gflop_count = get_gflop_count(B, M, N, K)
                gflops_mx = gflop_count / (time_mlx)
                if args.verbose:
                    if time_torch is None:
                        print(
                            f"{B:3d}, {M:4d}, {N:4d}, {K:4d}, {dtype}, {transpose}, skipped, {gflops_mx:05.3f}, n/a"
                        )
                    else:
                        gflops_pt = gflop_count / (time_torch)
                        diff = gflops_mx / gflops_pt - 1.0
                        print(
                            f"{B:3d}, {M:4d}, {N:4d}, {K:4d}, {dtype}, {transpose}, {gflops_pt:05.3f}, {gflops_mx:05.3f}, {100.0 * diff:+5.2f}%"
                        )
                        if gflops_pt >= 2.0 * gflops_mx:
                            print("ATTENTION ^^^^^^^")


if __name__ == "__main__":
    main()
