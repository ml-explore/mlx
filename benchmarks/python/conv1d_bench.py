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

N_warmup = 10
N_iter_bench = 100
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


def make_mx_conv_1D(strides=1, padding=0, groups=1):
    def mx_conv_1D(a, b):
        ys = []
        for _ in range(N_iter_func):
            y = mx.conv1d(a, b, stride=strides, padding=padding, groups=groups)
            ys.append(y)
        mx.eval(ys)
        return ys

    return mx_conv_1D


def make_pt_conv_1D(strides=1, padding=0, groups=1):
    @torch.no_grad()
    def pt_conv_1D(a, b):
        ys = []
        for _ in range(N_iter_func):
            y = torch.conv1d(a, b, stride=strides, padding=padding, groups=groups)
            ys.append(y)
        torch.mps.synchronize()
        return ys

    return pt_conv_1D


def bench_shape(N, iH, C, wH, O, strides, padding, np_dtype, groups):
    scale = 1.0 / math.sqrt(wH * C)
    a_np = np.random.uniform(0, 0.5, (N, iH, C)).astype(np_dtype)
    b_np = np.random.uniform(-scale, scale, (O, wH, int(C / groups))).astype(np_dtype)

    a_mx = mx.array(a_np)
    b_mx = mx.array(b_np)

    a_pt = torch.from_numpy(a_np.transpose((0, 2, 1))).to("mps")
    b_pt = torch.from_numpy(b_np.transpose((0, 2, 1))).to("mps")

    torch.mps.synchronize()

    f_mx = make_mx_conv_1D(strides, padding, groups)
    f_pt = make_pt_conv_1D(strides, padding, groups)

    time_torch = bench(f_pt, a_pt, b_pt)
    time_mlx = bench(f_mx, a_mx, b_mx)

    out_mx = mx.conv1d(a_mx, b_mx, stride=strides, padding=padding, groups=groups)
    out_pt = torch.conv1d(
        a_pt.to("cpu"), b_pt.to("cpu"), stride=strides, padding=padding, groups=groups
    )
    out_pt = torch.permute(out_pt, (0, 2, 1))
    out_pt = out_pt.numpy(force=True)

    atol = 2e-5 if np_dtype == np.float32 else 1e-4

    if not np.allclose(out_pt, out_mx, atol=atol):
        print(
            f"Failed at {(N, iH, C)}, {(O, wH, C)} [strides = {strides}, padding = {padding}, groups = {groups}] with max(|a - b|) = {np.max(np.abs(out_pt - out_mx))}"
        )

    return time_mlx, time_torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run conv benchmarks")

    dtypes = ("float32",)
    shapes = (
        (4, 32, 32, 5, 32, 1, 2, 1),
        (4, 32, 32, 5, 32, 1, 2, 2),
        (4, 32, 32, 5, 32, 1, 2, 4),
        (4, 32, 32, 5, 32, 1, 2, 8),
        (4, 32, 32, 5, 32, 1, 2, 8),
        (4, 32, 32, 5, 32, 1, 2, 16),
        (4, 32, 32, 5, 32, 1, 2, 32),
        (4, 32, 256, 5, 512, 1, 2, 2),
        (4, 32, 256, 5, 512, 1, 2, 128),
        (4, 32, 256, 5, 512, 1, 2, 256),
    )

    for dtype in dtypes:
        print("(N,  iH,  C),  (O,  wH,  C),   dtype,  stride, pads, groups, diff%")
        for N, iH, C, wH, O, strides, padding, groups in shapes:
            np_dtype = getattr(np, dtype)
            time_mlx, time_torch = bench_shape(
                N, iH, C, wH, O, strides, padding, np_dtype, groups
            )
            diff = time_torch / time_mlx - 1.0

            print(
                f"({N}, {iH:3d}, {C:3d}), ({O:3d}, {wH:2d}, {C:3d}), {dtype}, {strides:5d}, {padding:4d}, {groups:6d}, {100. * diff:+5.2f}%"
            )

            if time_mlx >= 2.0 * time_torch:
                print("ATTENTION ^^^^^^^")
