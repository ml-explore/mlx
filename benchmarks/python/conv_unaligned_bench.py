import math
import time

import mlx.core as mx
import numpy as np
import torch

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


def make_mx_conv_2D(strides=(1, 1), padding=(0, 0), groups=1):
    def mx_conv_2D(a, b):
        ys = []
        for i in range(N_iter_func):
            y = mx.conv2d(a, b, stride=strides, padding=padding, groups=groups)
            ys.append(y)
        mx.eval(ys)
        return ys

    return mx_conv_2D


def make_pt_conv_2D(strides=(1, 1), padding=(0, 0), groups=1):
    @torch.no_grad()
    def pt_conv_2D(a, b):
        ys = []
        for i in range(N_iter_func):
            y = torch.conv2d(a, b, stride=strides, padding=padding, groups=groups)
            ys.append(y)
        torch.mps.synchronize()
        return ys

    return pt_conv_2D


def bench_shape(N, H, W, C, kH, kW, O, strides, padding, groups, np_dtype):
    scale = 1.0 / math.sqrt(kH * kH * C)
    a_np = np.random.uniform(0, 0.5, (N, H, W, C)).astype(np_dtype)
    b_np = np.random.uniform(-scale, scale, (O, kH, kW, int(C / groups))).astype(
        np_dtype
    )

    a_mx = mx.array(a_np)
    b_mx = mx.array(b_np)

    a_pt = torch.from_numpy(a_np.transpose((0, 3, 1, 2))).to("mps")
    b_pt = torch.from_numpy(b_np.transpose((0, 3, 1, 2))).to("mps")

    torch.mps.synchronize()

    f_mx = make_mx_conv_2D(strides, padding, groups)
    f_pt = make_pt_conv_2D(strides, padding, groups)

    time_torch = bench(f_pt, a_pt, b_pt)
    time_mlx = bench(f_mx, a_mx, b_mx)

    out_mx = mx.conv2d(a_mx, b_mx, stride=strides, padding=padding, groups=groups)
    out_pt = torch.conv2d(
        a_pt.to("cpu"), b_pt.to("cpu"), stride=strides, padding=padding, groups=groups
    )
    out_pt = torch.permute(out_pt, (0, 2, 3, 1))
    out_pt = out_pt.numpy(force=True)

    atol = 2e-5 if np_dtype == np.float32 else 1e-4

    if not np.allclose(out_pt, out_mx, atol=atol):
        print(
            f"Failed at {(N, H, W, C)}, {(O, kH, kW, C)} [strides = {strides}, padding = {padding}, groups = {groups}] with max(|a - b|) = {np.max(np.abs(out_pt - out_mx))}"
        )

    return time_mlx, time_torch


if __name__ == "__main__":
    dtype = "float32"
    shapes = (
        (4, 32, 32, 21, 3, 3, 128),
        (4, 32, 32, 21, 3, 3, 37),
        (4, 32, 32, 370, 3, 3, 370),
        (4, 32, 32, 370, 7, 7, 128),
        (2, 320, 640, 21, 7, 7, 21),
    )
    for N, H, W, C, kh, kw, O in shapes:
        time_mlx, time_torch = bench_shape(
            N, H, W, C, kh, kw, O, (1, 1), (0, 0), 1, dtype
        )
        diff = time_torch / time_mlx - 1.0

        print(
            f"({N}, {H:3d}, {W:3d}, {C:3d}), ({O:3d}, {kh:2d}, {kw:2d}, {C:3d}), {dtype}, {100. * diff:+5.2f}%"
        )
        if time_mlx >= 2.0 * time_torch:
            print("ATTENTION ^^^^^^^")
