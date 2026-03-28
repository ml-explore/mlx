#!/usr/bin/env python3
"""Test mx.fast.turboquant_sdpa with the native Metal kernel."""

import sys
import os

# Use our built MLX
build_path = os.path.join(os.path.dirname(__file__), "build", "python")
sys.path.insert(0, build_path)

import mlx.core as mx
# mx.fast is accessed via mx.core.fast
import math
import time


def test_basic():
    """Basic turboquant_sdpa call."""
    B, H_q, H_kv, T, D = 1, 4, 4, 32, 128
    bits = 3
    vpw = 10
    packed_dim = (D + vpw - 1) // vpw  # 13

    q = mx.random.normal(shape=(B, H_q, 1, D)).astype(mx.float16)
    k_packed = mx.random.randint(0, 8, shape=(B, H_kv, T, packed_dim)).astype(mx.uint32)
    v = mx.random.normal(shape=(B, H_kv, T, D)).astype(mx.float16)
    k_norms = mx.random.normal(shape=(B, H_kv, T)).astype(mx.float32)
    codebook = mx.array([-2.15, -1.34, -0.756, -0.245, 0.245, 0.756, 1.34, 2.15], dtype=mx.float32)
    scale = 1.0 / math.sqrt(D)

    print("Calling mx.fast.turboquant_sdpa...")
    out = mx.fast.turboquant_sdpa(
        q, k_packed, v, k_norms, codebook,
        scale=scale, bits=bits,
    )
    mx.eval(out)
    print(f"  Output shape: {out.shape}")
    print(f"  Output dtype: {out.dtype}")
    print(f"  Output sample: {out[0, 0, 0, :5].tolist()}")
    print(f"  Has NaN: {mx.any(mx.isnan(out)).item()}")
    print("  PASSED!")


def test_speed():
    """Benchmark native turbo SDPA vs standard SDPA."""
    B, H_q, H_kv, D = 1, 28, 4, 128
    bits = 3
    vpw = 10
    packed_dim = (D + vpw - 1) // vpw
    scale = 1.0 / math.sqrt(D)

    for T in [256, 1024, 4096]:
        q = mx.random.normal(shape=(B, H_q, 1, D)).astype(mx.float16)
        v = mx.random.normal(shape=(B, H_kv, T, D)).astype(mx.float16)

        # Standard SDPA with float K
        k_float = mx.random.normal(shape=(B, H_kv, T, D)).astype(mx.float16)

        # TurboQuant
        k_packed = mx.random.randint(0, 8, shape=(B, H_kv, T, packed_dim)).astype(mx.uint32)
        k_norms = mx.abs(mx.random.normal(shape=(B, H_kv, T))).astype(mx.float32)
        codebook = mx.array([-2.15, -1.34, -0.756, -0.245, 0.245, 0.756, 1.34, 2.15], dtype=mx.float32)

        mx.eval(q, k_float, v, k_packed, k_norms)

        # Warmup
        for _ in range(5):
            mx.eval(mx.fast.scaled_dot_product_attention(q, k_float, v, scale=scale))
            mx.eval(mx.fast.turboquant_sdpa(q, k_packed, v, k_norms, codebook, scale=scale, bits=bits))

        # Standard SDPA
        t0 = time.perf_counter()
        for _ in range(50):
            mx.eval(mx.fast.scaled_dot_product_attention(q, k_float, v, scale=scale))
        std_ms = (time.perf_counter() - t0) / 50 * 1000

        # TurboQuant SDPA
        t0 = time.perf_counter()
        for _ in range(50):
            mx.eval(mx.fast.turboquant_sdpa(q, k_packed, v, k_norms, codebook, scale=scale, bits=bits))
        tq_ms = (time.perf_counter() - t0) / 50 * 1000

        print(f"  T={T:>5}: std={std_ms:.3f}ms turbo={tq_ms:.3f}ms ratio={tq_ms/std_ms:.2f}x")


if __name__ == "__main__":
    print("=" * 50)
    print("TurboQuant Native Metal SDPA Test")
    print("=" * 50)

    print("\n[Basic test]")
    try:
        test_basic()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n[Speed benchmark]")
    try:
        test_speed()
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
