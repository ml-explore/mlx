# Copyright © 2024 Apple Inc.

import argparse
import math
import os
import subprocess
import time

import mlx.core as mx
import numpy as np

device_name = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
device_name = device_name.decode("utf-8").strip("\n")

N_warmup = 5
N_iter_bench = 40
N_iter_func = 8


def bench(f, *args):
    for i in range(N_warmup):
        f(*args)

    s = time.perf_counter_ns()
    for i in range(N_iter_bench):
        f(*args)
    e = time.perf_counter_ns()
    return (e - s) * 1e-9


def mlx_sdpa_fused_inner(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)


def mlx_sdpa_unfused_inner(q, k, v, scale, f32softmax=False):
    q_dtype = q.dtype
    q = q * mx.array(scale, q_dtype)
    n_q_heads = q.shape[-3]
    n_kv_heads = k.shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    B = q.shape[0]
    L = q.shape[2]

    if n_repeats > 1:
        q = mx.reshape(q, [B, n_kv_heads, n_repeats, L, -1])
        k = mx.expand_dims(k, 2)
        v = mx.expand_dims(v, 2)

    scores = q @ mx.swapaxes(k, -1, -2)
    if f32softmax:
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q_dtype)
    else:
        scores = mx.softmax(scores, axis=-1)

    out = scores @ v
    if n_repeats > 1:
        out = mx.reshape(out, [B, n_q_heads, L, -1])

    return out


def mlx_spda_unfused(q, k, v, scale, transpose):
    q_out = q
    if transpose:
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

    for i in range(N_iter_func):
        if transpose:
            q_out = mx.transpose(q_out, (0, 2, 1, 3))
        q_out = mlx_sdpa_unfused_inner(q_out, k, v, scale)
        if transpose:
            q_out = mx.transpose(q_out, (0, 2, 1, 3))

    mx.eval(q_out)
    return q_out


def mlx_spda_fused(q, k, v, scale, transpose):
    q_out = q
    if transpose:
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

    for i in range(N_iter_func):
        if transpose:
            q_out = mx.transpose(q_out, (0, 2, 1, 3))
        q_out = mlx_sdpa_fused_inner(q_out, k, v, scale)
        if transpose:
            q_out = mx.transpose(q_out, (0, 2, 1, 3))

    mx.eval(q_out)
    return q_out


def bench_shape(B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, np_dtype, transpose=True):
    shape_q = (
        (B, qsl, n_q_heads, head_dim) if transpose else (B, n_q_heads, qsl, head_dim)
    )
    shape_kv = (
        (B, ksl, n_kv_heads, head_dim) if transpose else (B, n_kv_heads, ksl, head_dim)
    )

    q_np = np.random.normal(0.0, 1.0 / math.sqrt(head_dim), shape_q).astype(np_dtype)
    k_np = np.random.normal(0.0, 1.0 / math.sqrt(head_dim), shape_kv).astype(np_dtype)
    v_np = np.random.normal(0.0, 1.0 / math.sqrt(head_dim), shape_kv).astype(np_dtype)

    scale = math.sqrt(1.0 / head_dim)

    q_mx = mx.array(q_np)
    k_mx = mx.array(k_np)
    v_mx = mx.array(v_np)

    time_mlx_unfused = bench(mlx_spda_unfused, q_mx, k_mx, v_mx, scale, transpose)
    time_mlx_fused = bench(mlx_spda_fused, q_mx, k_mx, v_mx, scale, transpose)

    if transpose:
        q_mx = mx.transpose(q_mx, (0, 2, 1, 3))
        k_mx = mx.transpose(k_mx, (0, 2, 1, 3))
        v_mx = mx.transpose(v_mx, (0, 2, 1, 3))

    o_mlx_fused = mlx_sdpa_fused_inner(q_mx, k_mx, v_mx, scale)
    o_mlx_unfused = mlx_sdpa_unfused_inner(q_mx, k_mx, v_mx, scale, f32softmax=True)

    atol = 1e-5 if np_dtype == np.float32 else 1e-4

    if not mx.allclose(o_mlx_fused, o_mlx_unfused, atol=atol):
        print(
            f"Failed at (B: {B}, qsl: {qsl}, ksl: {ksl}, head_dim: {head_dim}, n_qh: {n_q_heads}, n_kvh: {n_kv_heads}) [tpose = {transpose}] with max(|a - b|) = {mx.max(mx.abs(o_mlx_unfused - o_mlx_fused)):3.2e}"
        )

    return time_mlx_fused, time_mlx_unfused


def get_gflop_count(B, M, N, K):
    return float(2.0 * N_iter_bench * N_iter_func * B * M * N * K) / float(1024.0**3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gemm benchmarks")

    dtypes = ("float16", "float32")[:1]
    transposes = (False,)

    # fmt: off
    shapes_64 = (
        # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
          (  1,    32,    32,       64,   32,    32),
          (  1,    64,    64,       64,   32,    32),
          (  1,   128,   128,       64,   32,    32),
          (  1,   256,   256,       64,   32,    32),
          (  1,   512,   512,       64,   32,    32),
          (  1,  1024,  1024,       64,   32,    32),
          (  1,  2048,  2048,       64,   32,    32),
          (  1,  4096,  4096,       64,   32,    32),
    )

    shapes_80 = (
        # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
          (  1,  1024,  1024,       80,   32,    32),
          (  1,  2048,  2048,       80,   32,    32),
          (  1,  4096,  4096,       80,   32,    32),
    )

    shapes_128 = (
        # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
          (  1,  1024,  1024,      128,   32,    32),
          (  1,  2048,  2048,      128,   32,    32),
          (  1,  4096,  4096,      128,   32,    32),
    )
    # fmt: on

    shapes = shapes_64 + shapes_80 + shapes_128

    print("  B,   qsl,   ksl, hdim, n_qh, n_kvh, tpose,   dtype, t_unfs, t_fuse, diff%")

    for dtype in dtypes:
        for transpose in transposes:
            for B, qsl, ksl, head_dim, n_q_heads, n_kv_heads in shapes:
                np_dtype = getattr(np, dtype)
                time_mlx_fused, time_mlx_unfused = bench_shape(
                    B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, np_dtype, transpose
                )
                diff = time_mlx_unfused / time_mlx_fused - 1.0
                t_str = 1 if transpose else 0
                print(
                    f"{B:3d}, {qsl:5d}, {ksl:5d}, {head_dim:4d}, {n_q_heads:4d}, {n_kv_heads:5d}, {t_str:5d}, {dtype}, {time_mlx_unfused: 2.3f}, {time_mlx_fused: 2.3f}, {100. * diff:+5.2f}%"
                )
