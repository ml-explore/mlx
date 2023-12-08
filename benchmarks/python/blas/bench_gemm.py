# Copyright © 2023 Apple Inc.

import argparse
import math
import os
import subprocess
import time

import mlx.core as mx
import numpy as np
import torch

device_name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
device_name = device_name.decode("utf-8").strip("\n")

N_warmup = 8
N_iter_bench = 80
N_iter_func = 5


def bench(f, a, b):
    for i in range(N_warmup):
        f(a, b)
    torch.mps.synchronize()

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
    torch.mps.synchronize()
    return ys


@torch.no_grad()
def gemm_nt_torch(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a @ b.transpose(-1, -2)
        ys.append(y)
    torch.mps.synchronize()
    return ys


@torch.no_grad()
def gemm_tn_torch(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a.transpose(-1, -2) @ b
        ys.append(y)
    torch.mps.synchronize()
    return ys


@torch.no_grad()
def gemm_tt_torch(a, b):
    ys = []
    for i in range(N_iter_func):
        y = a.transpose(-1, -2) @ b.transpose(-1, -2)
        ys.append(y)
    torch.mps.synchronize()
    return ys


def bench_shape(B, M, N, K, np_dtype, transpose="nn"):
    shape_a = (B, M, K) if transpose[0] == "n" else (B, K, M)
    shape_b = (B, K, N) if transpose[1] == "n" else (B, N, K)

    a_np = np.random.normal(0.0, 1.0 / math.sqrt(M + K), shape_a).astype(np_dtype)
    b_np = np.random.normal(0.0, 1.0 / math.sqrt(N + K), shape_b).astype(np_dtype)

    a_mx = mx.array(a_np)
    b_mx = mx.array(b_np)

    a_pt = torch.from_numpy(a_np).to("mps")
    b_pt = torch.from_numpy(b_np).to("mps")

    torch.mps.synchronize()

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

    time_torch = bench(f_pt, a_pt, b_pt)
    time_mlx = bench(f_mx, a_mx, b_mx)

    t_a = (0, 1, 2) if transpose[0] == "n" else (0, 2, 1)
    t_b = (0, 1, 2) if transpose[1] == "n" else (0, 2, 1)

    c_mlx = a_mx.transpose(t_a) @ b_mx.transpose(t_b)
    c_npy = a_np.transpose(t_a).astype(np.float32) @ b_np.transpose(t_b).astype(
        np.float32
    )

    atol = 1e-5 if np_dtype == np.float32 else 1e-4

    if not np.allclose(c_mlx, c_npy.astype(np_dtype), atol=atol):
        print(
            f"Failed at {(B, M, N, K)} [transpose = {transpose}] with max(|a - b|) = {np.max(np.abs(c_npy - c_mlx))}"
        )

    return time_mlx, time_torch


def get_gflop_count(B, M, N, K):
    return float(2.0 * N_iter_bench * N_iter_func * B * M * N * K) / float(1024.0**3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gemm benchmarks")

    dtypes = ("float32", "float16")
    transposes = ("nn", "nt", "tn")
    shapes = (
        (16, 1024, 1024, 1024),
        (1, 1024, 1024, 2048),
        (4, 1024, 1024, 4096),
        (4, 1024, 4096, 1024),
        (1, 4096, 4096, 4096),
        (15, 1023, 1023, 1023),
        (17, 1025, 1025, 1025),
    )

    for dtype in dtypes:
        for transpose in transposes:
            for B, M, N, K in shapes:
                np_dtype = getattr(np, dtype)
                time_mlx, time_torch = bench_shape(B, M, N, K, np_dtype, transpose)

                gflop_count = get_gflop_count(B, M, N, K)
                gflops_mx = gflop_count / (time_mlx)
                gflops_pt = gflop_count / (time_torch)
                diff = gflops_mx / gflops_pt - 1.0

                print(
                    f"{B:3d}, {M:4d}, {N:4d}, {K:4d}, {dtype}, {transpose}, {gflops_pt:05.3f}, {gflops_mx:05.3f}, {100. * diff:+5.2f}%"
                )
                if gflops_pt >= 2.0 * gflops_mx:
                    print("ATTENTION ^^^^^^^")
