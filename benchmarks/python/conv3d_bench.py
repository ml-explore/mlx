import math
import time

import mlx.core as mx
import numpy as np
import torch

N_warmup = 2
N_iter_bench = 10
N_iter_func = 10


def bench(f, a, b, b_prime):
    for i in range(N_warmup):
        f(a, b, b_prime)
    torch.mps.synchronize()

    s = time.perf_counter_ns()
    for i in range(N_iter_bench):
        f(a, b, b_prime)
    e = time.perf_counter_ns()
    return (e - s) * 1e-9


def make_mx_conv_3D(strides=(1, 1, 1), padding=(0, 0, 0), groups=1):
    def mx_conv_3D(a, b, b_prime):
        y = a
        for i in range(N_iter_func):
            y = mx.conv3d(y, b, stride=strides, padding=padding, groups=groups)
            y = mx.conv3d(y, b_prime, stride=strides, padding=padding, groups=groups)
        mx.eval(y)
        return y

    return mx_conv_3D


def make_pt_conv_3D(strides=(1, 1, 1), padding=(0, 0, 0), groups=1):
    @torch.no_grad()
    def pt_conv_3D(a, b, b_prime):
        y = a
        for i in range(N_iter_func):
            y = torch.conv3d(y, b, stride=strides, padding=padding, groups=groups)
            y = torch.conv3d(y, b_prime, stride=strides, padding=padding, groups=groups)
        torch.mps.synchronize()
        return y

    return pt_conv_3D


def bench_shape(N, D, H, W, C, kD, kH, kW, O, strides, padding, groups, np_dtype):
    scale = 1.0 / math.sqrt(kD * kH * kW * C)
    a_np = np.random.uniform(0, 0.5, (N, D, H, W, C))
    b_np = np.random.uniform(-scale, scale, (O, kD, kH, kW, int(C / groups)))
    b_prime_np = np.random.uniform(-scale, scale, (C, kD, kH, kW, int(O / groups)))

    a_np, b_np, b_prime_np = map(lambda x: x.astype(np_dtype), (a_np, b_np, b_prime_np))
    a_mx, b_mx, b_prime_mx = map(lambda x: mx.array(x), (a_np, b_np, b_prime_np))
    a_pt, b_pt, b_prime_pt = map(
        lambda x: torch.from_numpy(x.transpose(0, 4, 1, 2, 3)).to("mps"),
        (a_np, b_np, b_prime_np),
    )

    torch.mps.synchronize()

    f_mx = make_mx_conv_3D(strides, padding, groups)
    f_pt = make_pt_conv_3D(strides, padding, groups)

    time_torch = bench(f_pt, a_pt, b_pt, b_prime_pt)
    time_mlx = bench(f_mx, a_mx, b_mx, b_prime_mx)

    # Measure MLX memory
    mx.clear_cache()
    mx.reset_peak_memory()
    y = mx.conv3d(a_mx, b_mx, stride=strides, padding=padding, groups=groups)
    mx.eval(y)
    mlx_peak_mb = mx.get_peak_memory() / 1024**2
    mlx_active_mb = mx.get_active_memory() / 1024**2
    del y

    # Measure PyTorch MPS memory
    torch.mps.synchronize()
    torch.mps.empty_cache()
    y = torch.conv3d(a_pt, b_pt, stride=strides, padding=padding, groups=groups)
    torch.mps.synchronize()
    pt_current_mb = torch.mps.current_allocated_memory() / 1024**2
    pt_driver_mb = torch.mps.driver_allocated_memory() / 1024**2
    del y

    out_mx = mx.conv3d(a_mx, b_mx, stride=strides, padding=padding, groups=groups)
    out_pt = torch.conv3d(
        a_pt.to("cpu"), b_pt.to("cpu"), stride=strides, padding=padding, groups=groups
    )
    out_pt = torch.permute(out_pt, (0, 2, 3, 4, 1))
    out_pt = out_pt.numpy(force=True)

    atol = 2e-5 if np_dtype == np.float32 else 5e-4

    if not np.allclose(out_pt, out_mx, atol=atol):
        print(
            f"Failed at {(N, D, H, W, C)}, {(O, kD, kH, kW, C)} "
            f"[strides = {strides}, padding = {padding}, groups = {groups}] "
            f"with max(|a - b|) = {np.max(np.abs(out_pt - out_mx))}"
        )

    return time_mlx, time_torch, mlx_peak_mb, mlx_active_mb, pt_current_mb, pt_driver_mb


if __name__ == "__main__":
    dtypes = ("float16", "float32")
    shapes = (
        # (C % 16 == 0)
        (4, 16, 16, 16, 32, 3, 3, 3, 32, (1, 1, 1), (1, 1, 1), 1),
        (4, 16, 16, 16, 64, 3, 3, 3, 64, (1, 1, 1), (1, 1, 1), 1),
        (4, 16, 16, 16, 128, 3, 3, 3, 128, (1, 1, 1), (1, 1, 1), 1),
        (4, 32, 32, 32, 64, 3, 3, 3, 64, (1, 1, 1), (1, 1, 1), 1),
        (4, 32, 32, 32, 128, 3, 3, 3, 128, (1, 1, 1), (1, 1, 1), 1),
        # Larger spatial dims
        (2, 64, 64, 64, 32, 3, 3, 3, 64, (1, 1, 1), (1, 1, 1), 1),
        (1, 64, 64, 64, 64, 3, 3, 3, 128, (1, 1, 1), (1, 1, 1), 1),
        # Strided
        (4, 32, 32, 32, 64, 3, 3, 3, 128, (2, 2, 2), (1, 1, 1), 1),
        # Asymmetric kernels
        (4, 32, 32, 32, 64, 3, 1, 1, 128, (1, 1, 1), (1, 0, 0), 1),
        (4, 32, 32, 32, 64, 1, 3, 3, 128, (1, 1, 1), (0, 1, 1), 1),
        # (C % 16 != 0)
        (4, 16, 16, 16, 21, 3, 3, 3, 21, (1, 1, 1), (1, 1, 1), 1),
        (4, 16, 16, 16, 55, 3, 3, 3, 55, (1, 1, 1), (1, 1, 1), 1),
        (4, 32, 32, 32, 55, 3, 3, 3, 55, (1, 1, 1), (1, 1, 1), 1),
        (4, 16, 16, 16, 3, 3, 3, 3, 32, (1, 1, 1), (1, 1, 1), 1),
    )

    for dtype in dtypes:
        print(f"\n{'=' * 120}" f"\n  dtype: {dtype}" f"\n{'=' * 120}")
        print(
            f"{'(N,   D,   H,   W,   C)':<26s} {'(  O, kD, kH, kW,   C)':<24s} "
            f"{'stride':<12s} {'pads':<12s} {'groups':>6s} "
            f"{'diff%':>7s}  "
            f"{'MLX peak':>9s} {'MLX act':>8s} {'PT cur':>8s} {'PT drv':>8s}"
        )
        for N, D, H, W, C, kD, kH, kW, O, strides, padding, groups in shapes:
            np_dtype = getattr(np, dtype)
            time_mlx, time_torch, mlx_peak, mlx_act, pt_cur, pt_drv = bench_shape(
                N, D, H, W, C, kD, kH, kW, O, strides, padding, groups, np_dtype
            )
            diff = time_torch / time_mlx - 1.0

            print(
                f"({N}, {D:3d}, {H:3d}, {W:3d}, {C:3d}), ({O:3d}, {kD:2d}, {kH:2d}, {kW:2d}, {C:3d}), "
                f"{strides}, {padding}, {groups:6d}, "
                f"{100. * diff:+6.1f}%  "
                f"{mlx_peak:8.1f}  {mlx_act:7.1f}  {pt_cur:7.1f}  {pt_drv:7.1f}"
            )
