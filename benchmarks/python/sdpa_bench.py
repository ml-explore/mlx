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


def prepare_inputs(B, qL, kL, D, qH, kH, mask, transpose, dtype):
    np_dtype = getattr(np, dtype)

    shape_q = (B, qL, qH, D) if transpose else (B, qH, qL, D)
    shape_kv = (B, kL, kH, D) if transpose else (B, kH, kL, D)

    scale = 1.0 / math.sqrt(D)

    q_np = np.random.normal(0.0, 1.0, shape_q).astype(np_dtype)
    k_np = np.random.normal(0.0, scale, shape_kv).astype(np_dtype)
    v_np = np.random.normal(0.0, scale, shape_kv).astype(np_dtype)

    q_mx = mx.array(q_np)
    k_mx = mx.array(k_np)
    v_mx = mx.array(v_np)

    if mask is not None:
        if mask == "additive":
            mask_np = np.random.normal(0.0, 1.0, (B, qH, qL, kL)).astype(np_dtype)
            mask = mx.array(mask_np)
        elif mask == "bool":
            mask_np = np.random.uniform(0.0, 1.0, (B, qH, qL, kL)) < 0.5
            mask = mx.array(mask_np)

    return q_mx, k_mx, v_mx, scale, mask


def mlx_ref_attn(q, k, v, scale=1.0, mask=None):
    q_dtype = q.dtype
    q = q * mx.array(scale, q_dtype)
    n_q_heads = q.shape[-3]
    n_kv_heads = k.shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    B = q.shape[0]
    L = q.shape[2]
    kL = k.shape[2]

    if n_repeats > 1:
        q = mx.reshape(q, [B, n_kv_heads, n_repeats, L, -1])
        k = mx.expand_dims(k, 2)
        v = mx.expand_dims(v, 2)

    scores = q @ mx.swapaxes(k, -1, -2)

    if mask is not None:

        if mask == "causal":
            q_offset = max(0, kL - L)
            q_indices = mx.arange(q_offset, q_offset + L)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]

        if n_repeats > 1 and mask.ndim >= 3:
            if mask.shape[-3] == 1:
                mask = mx.expand_dims(mask, -3)
            else:
                mask = mx.unflatten(mask, -3, (n_kv_heads, n_repeats))

        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, -np.float32(np.inf))
        else:
            scores += mask

    scores = mx.softmax(scores, axis=-1, precise=True)

    out = scores @ v
    if n_repeats > 1:
        out = mx.reshape(out, [B, n_q_heads, L, -1])

    return out


def mlx_fused_attn(q, k, v, scale, mask):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def do_attention(f, q, k, v, scale, mask=None, transpose=False):
    if transpose:
        q_t = mx.transpose(q, (0, 2, 1, 3))
        k_t = mx.transpose(k, (0, 2, 1, 3))
        v_t = mx.transpose(v, (0, 2, 1, 3))
        o_t = f(q_t, k_t, v_t, scale=scale, mask=mask)
        return mx.transpose(o_t, (0, 2, 1, 3))
    else:
        return f(q, k, v, scale=scale, mask=mask)


def do_attention_bench(f, q, k, v, scale, mask=None, transpose=False):
    q_out = q

    for i in range(N_iter_func):
        q_out = do_attention(f, q_out, k, v, scale, mask=mask, transpose=transpose)

    mx.eval(q_out)
    return q_out


def bench_shape(
    B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, dtype, transpose=True, mask_in=None
):
    q_mx, k_mx, v_mx, scale, mask = prepare_inputs(
        B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, mask_in, transpose, dtype
    )

    time_mlx_unfused = bench(
        do_attention_bench, mlx_ref_attn, q_mx, k_mx, v_mx, scale, mask, transpose
    )
    time_mlx_fused = bench(
        do_attention_bench, mlx_fused_attn, q_mx, k_mx, v_mx, scale, mask, transpose
    )

    o_mlx_fused = do_attention(mlx_ref_attn, q_mx, k_mx, v_mx, scale, mask, transpose)
    o_mlx_unfused = do_attention(
        mlx_fused_attn, q_mx, k_mx, v_mx, scale, mask, transpose
    )

    atol = 1e-5 if dtype == "float32" else 2e-4

    if not mx.allclose(o_mlx_fused, o_mlx_unfused, atol=atol, rtol=atol):
        print(
            f"Failed at (B: {B}, qsl: {qsl}, ksl: {ksl}, head_dim: {head_dim}, n_qh: {n_q_heads}, n_kvh: {n_kv_heads}, mask: {mask_in}) [tpose = {transpose}] with max(|a - b|) = {mx.max(mx.abs(o_mlx_unfused - o_mlx_fused)):3.2e}"
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
          (  1,  1024,  1024,       64,   32,     8),
          (  1,  2048,  2048,       64,   32,     8),
          (  1,  4096,  4096,       64,   32,     8),
    )

    shapes_80 = (
        # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
          (  1,  1024,  1024,       80,   32,     8),
          (  1,  2048,  2048,       80,   32,     8),
          (  1,  4096,  4096,       80,   32,     8),
    )

    shapes_128 = (
        # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
          (  1,  1024,  1024,      128,   32,     8),
          (  1,  2048,  2048,      128,   32,     8),
          (  1,  4096,  4096,      128,   32,     8),
    )
    # fmt: on

    shapes = shapes_64 + shapes_80 + shapes_128

    masks = [None, "bool", "causal"]

    print(
        "  B,   qsl,   ksl, hdim, n_qh, n_kvh, t,   dtype,     mask, t_unfs, t_fuse, diff%"
    )

    for dtype in dtypes:
        for transpose in transposes:
            for B, qsl, ksl, head_dim, n_q_heads, n_kv_heads in shapes:
                for mask_in in masks:
                    time_mlx_fused, time_mlx_unfused = bench_shape(
                        B,
                        qsl,
                        ksl,
                        head_dim,
                        n_q_heads,
                        n_kv_heads,
                        dtype,
                        transpose,
                        mask_in,
                    )
                    diff = time_mlx_unfused / time_mlx_fused - 1.0
                    t_str = 1 if transpose else 0
                    print(
                        f"{B:3d}, {qsl:5d}, {ksl:5d}, {head_dim:4d}, {n_q_heads:4d}, {n_kv_heads:5d}, {t_str:1d}, {dtype}, {str(mask_in):>8}, {time_mlx_unfused: 2.3f}, {time_mlx_fused: 2.3f}, {100. * diff:+5.2f}%"
                    )
