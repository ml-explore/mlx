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

device_name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
device_name = device_name.decode("utf-8").strip("\n")

N_warmup = 5
N_iter_bench = 50
N_iter_func = 20

out_vec_sizes = [128, 512, 2048, 4096]
in_vec_sizes = [128, 512, 2048, 4096]

benchmark_vector_lens = []
benchmark_vector_lens += [(i + 1) * 4096 for i in range(8)][::2]
benchmark_vector_lens += [(i + 1) * 4095 for i in range(8)][::2]
benchmark_vector_lens += [(i + 1) * 4097 for i in range(8)][::2]
benchmark_vector_lens += [64, 128, 512, 1024, 2048, 11008, 32000]

benchmark_vector_lens.sort()


def bench(f, m, v):
    for i in range(N_warmup):
        f(m, v)
    torch.mps.synchronize()

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
    torch.mps.synchronize()
    return ys


@torch.no_grad()
def gemv_t_torch(m, v):
    ys = []
    for i in range(N_iter_func):
        y = v @ m
        ys.append(y)
    torch.mps.synchronize()
    return ys


def bench_lens(in_vec_len, out_vec_len, np_dtype, transpose=False):
    shape_mat = (in_vec_len, out_vec_len) if transpose else (out_vec_len, in_vec_len)
    shape_vec = (1, in_vec_len) if transpose else (in_vec_len, 1)

    mat_npy = np.random.normal(0.0, 2.0 / in_vec_len, shape_mat).astype(np_dtype)
    vec_npy = np.random.normal(0.0, 2.0 / in_vec_len, shape_vec).astype(np_dtype)
    mat_mlx = mx.array(mat_npy)
    vec_mlx = mx.array(vec_npy)
    mat_trc = torch.from_numpy(mat_npy).to("mps")
    vec_trc = torch.from_numpy(vec_npy).to("mps")

    torch.mps.synchronize()

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
    item_size = 4 if np_dtype == np.float32 else 2
    return float(N_iter_bench * N_iter_func * n_elem * item_size) / float(1024**3)


def bench_with_in_len(ax, in_vec_len, out_vector_lens, dtype, tranpose):
    np_dtype = getattr(np, dtype)
    mlx_gb_s = []
    mlx_gflops = []
    pyt_gb_s = []
    pyt_gflops = []

    for out_vec_len in out_vector_lens:
        gflop_count = get_gflop_count(in_vec_len, out_vec_len)
        gbyte_size = get_gbyte_size(in_vec_len, out_vec_len, np_dtype)

        time_mlx, time_torch = bench_lens(in_vec_len, out_vec_len, np_dtype, transpose)

        mlx_gb_s.append(gbyte_size / time_mlx)
        pyt_gb_s.append(gbyte_size / time_torch)

        mlx_gflops.append(gflop_count / time_mlx)
        pyt_gflops.append(gflop_count / time_torch)

    if transpose:
        title = f"gemv_t ([1, {in_vec_len}] [{in_vec_len}, out_vec_len]) | {dtype}"
    else:
        title = f"gemv ([out_vec_len, {in_vec_len}] X [{in_vec_len}, 1] ) | {dtype}"

    ax.plot(out_vector_lens, mlx_gb_s, "tab:blue", label="MLX")
    ax.plot(out_vector_lens, pyt_gb_s, "tab:red", label="Torch")
    ax.set_title(title)
    ax.set(xlabel="out_vector_len", ylabel="Performance (GB/s)")
    ax.legend()


def bench_with_out_len(ax, out_vec_len, in_vector_lens, dtype, tranpose):
    np_dtype = getattr(np, dtype)
    mlx_gb_s = []
    mlx_gflops = []
    pyt_gb_s = []
    pyt_gflops = []

    for in_vec_len in in_vector_lens:
        gflop_count = get_gflop_count(in_vec_len, out_vec_len)
        gbyte_size = get_gbyte_size(in_vec_len, out_vec_len, np_dtype)

        time_mlx, time_torch = bench_lens(in_vec_len, out_vec_len, np_dtype, transpose)

        mlx_gb_s.append(gbyte_size / time_mlx)
        pyt_gb_s.append(gbyte_size / time_torch)

        mlx_gflops.append(gflop_count / time_mlx)
        pyt_gflops.append(gflop_count / time_torch)

    if transpose:
        title = f"([1, in_vec_len] [in_vec_len, {out_vec_len}])"
    else:
        title = f"([{out_vec_len}, in_vec_len] X [in_vec_len, 1] )"

    ax.plot(in_vector_lens, mlx_gb_s, "tab:blue", label="MLX")
    ax.plot(in_vector_lens, pyt_gb_s, "tab:red", label="Torch")
    ax.set_title(title)
    ax.set(xlabel="in_vector_len", ylabel="Performance (GB/s)")
    ax.legend()


for transpose in (False, True):
    for dtype in ("float32", "float16"):
        fig, axs = plt.subplots(
            len(in_vec_sizes), 2, figsize=(8.5, 11), layout="constrained"
        )

        for i, in_vec_len in enumerate(in_vec_sizes):
            bench_with_in_len(
                axs[i][0], in_vec_len, benchmark_vector_lens, dtype, transpose
            )

        for i, out_vec_len in enumerate(out_vec_sizes):
            bench_with_out_len(
                axs[i][1], out_vec_len, benchmark_vector_lens, dtype, transpose
            )

        op_name = "gemv_t" if transpose else "gemv"
        fig.suptitle(f"{device_name}: {dtype} {op_name}")
        fig.savefig(
            os.path.join(
                results_dir, f'{device_name.replace(" ", "_")}_{dtype}_{op_name}.pdf'
            )
        )
        plt.close(fig)
