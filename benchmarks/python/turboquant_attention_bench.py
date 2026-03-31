"""
Benchmark mx.fast.turboquant_attention vs standard SDPA.

Compares the fused TurboQuant kernel (attention directly from 2-bit
compressed KV cache) against standard scaled_dot_product_attention
on full-precision keys/values.

Usage:
    python benchmarks/python/turboquant_attention_bench.py
"""

import math

import mlx.core as mx
import numpy as np
from time_utils import time_fn


def make_random_orthogonal(D, seed=42):
    np.random.seed(seed)
    G = np.random.randn(D, D).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    return mx.array(Q)


def make_compressed_kv(B, H_kv, N, D, group_size=32):
    """Create synthetic compressed KV data matching turboquant_attention input format."""
    packed_d = D // 4  # 2-bit: 4 values per byte
    packed_d_signs = D // 8  # 1-bit: 8 values per byte
    n_groups = D // group_size
    packed_v = n_groups * (group_size // 4)

    k_packed = mx.random.randint(0, 256, (B, H_kv, N, packed_d)).astype(mx.uint8)
    k_signs = mx.random.randint(0, 256, (B, H_kv, N, packed_d_signs)).astype(mx.uint8)
    k_norms = mx.abs(mx.random.normal((B, H_kv, N))) + 0.1
    k_res_norms = mx.abs(mx.random.normal((B, H_kv, N))) * 0.1
    centroids = mx.array([-0.75, -0.25, 0.25, 0.75])

    v_packed = mx.random.randint(0, 256, (B, H_kv, N, packed_v)).astype(mx.uint8)
    v_scales = mx.abs(mx.random.normal((B, H_kv, N, n_groups))) + 0.01
    v_zeros = mx.random.normal((B, H_kv, N, n_groups))

    return (
        k_packed,
        k_signs,
        k_norms,
        k_res_norms,
        centroids,
        v_packed,
        v_scales,
        v_zeros,
    )


def turboquant_attn(
    q,
    k_packed,
    k_signs,
    k_norms,
    k_res_norms,
    centroids,
    v_packed,
    v_scales,
    v_zeros,
    rotation,
    sketch,
    scale,
    qjl_scale,
    loops=10,
):
    for _ in range(loops):
        acc, m, l = mx.fast.turboquant_attention(
            q,
            k_packed,
            k_signs,
            k_norms,
            k_res_norms,
            centroids,
            v_packed,
            v_scales,
            v_zeros,
            rotation,
            sketch,
            scale=scale,
            qjl_scale=qjl_scale,
        )
    return acc


def standard_sdpa(q, k, v, scale, loops=10):
    for _ in range(loops):
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    return out


def bench_config(B, H_q, H_kv, N, D):
    print(f"\n  B={B}, H_q={H_q}, H_kv={H_kv}, N={N}, D={D}")

    scale = 1.0 / math.sqrt(D)
    qjl_scale = 1.0 / math.sqrt(D)
    dtype = mx.float16

    q = mx.random.normal((B, H_q, 1, D)).astype(dtype)
    k_fp = mx.random.normal((B, H_kv, N, D)).astype(dtype)
    v_fp = mx.random.normal((B, H_kv, N, D)).astype(dtype)

    rotation = make_random_orthogonal(D, seed=42).astype(dtype)
    sketch = make_random_orthogonal(D, seed=99).astype(dtype)

    k_packed, k_signs, k_norms, k_res_norms, centroids, v_packed, v_scales, v_zeros = (
        make_compressed_kv(B, H_kv, N, D)
    )

    mx.eval(
        q,
        k_fp,
        v_fp,
        rotation,
        sketch,
        k_packed,
        k_signs,
        k_norms,
        k_res_norms,
        centroids,
        v_packed,
        v_scales,
        v_zeros,
    )

    # Benchmark standard SDPA
    time_fn(standard_sdpa, q, k_fp, v_fp, scale, msg="standard SDPA")

    # Benchmark TurboQuant
    time_fn(
        turboquant_attn,
        q,
        k_packed,
        k_signs,
        k_norms,
        k_res_norms,
        centroids,
        v_packed,
        v_scales,
        v_zeros,
        rotation,
        sketch,
        scale,
        qjl_scale,
        msg="turboquant_attention",
    )

    # Memory comparison
    fp_bytes = k_fp.nbytes + v_fp.nbytes
    tq_bytes = (
        k_packed.nbytes
        + k_signs.nbytes
        + k_norms.nbytes
        + k_res_norms.nbytes
        + v_packed.nbytes
        + v_scales.nbytes
        + v_zeros.nbytes
    )
    ratio = fp_bytes / tq_bytes
    print(
        f"  Memory: FP16 KV = {fp_bytes / 1e6:.1f} MB, "
        f"Compressed = {tq_bytes / 1e6:.1f} MB ({ratio:.1f}x smaller)"
    )


if __name__ == "__main__":
    mx.random.seed(42)
    print("=" * 60)
    print("TurboQuant Attention Benchmark")
    print("=" * 60)

    # Typical model configs
    configs = [
        # (B, H_q, H_kv, N, D) — matching real model architectures
        (1, 16, 4, 1024, 128),  # 3B model, 1K context
        (1, 16, 4, 4096, 128),  # 3B model, 4K context
        (1, 16, 4, 16384, 128),  # 3B model, 16K context
        (1, 40, 8, 4096, 128),  # 32B model, 4K context
        (1, 40, 8, 16384, 128),  # 32B model, 16K context
    ]

    for config in configs:
        bench_config(*config)

    print("\n" + "=" * 60)
    print("Done.")
