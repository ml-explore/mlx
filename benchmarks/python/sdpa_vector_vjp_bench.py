# Copyright © 2024 Apple Inc.

import math
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

    times = []
    for i in range(N_iter_bench):
        s = time.perf_counter_ns()
        f(*args)
        e = time.perf_counter_ns()
        times.append((e - s) * 1e-9)
    times.sort()
    p50 = times[len(times) // 2]
    p90 = times[int(len(times) * 0.9)]
    return p50, p90


def prepare_inputs(B, qL, kL, D, qH, kH, dtype):
    np_dtype = getattr(np, dtype)
    scale = 1.0 / math.sqrt(D)

    q_np = np.random.normal(0.0, 1.0, (B, qH, qL, D)).astype(np_dtype)
    k_np = np.random.normal(0.0, scale, (B, kH, kL, D)).astype(np_dtype)
    v_np = np.random.normal(0.0, scale, (B, kH, kL, D)).astype(np_dtype)

    return mx.array(q_np), mx.array(k_np), mx.array(v_np), scale


def mlx_ref_vjp(q, k, v, scale):
    n_q_heads = q.shape[1]
    n_kv_heads = k.shape[1]
    n_repeats = n_q_heads // n_kv_heads
    B, _, L, D = q.shape

    q_s = q * mx.array(scale, q.dtype)
    if n_repeats > 1:
        q_s = mx.reshape(q_s, [B, n_kv_heads, n_repeats, L, -1])
        k_e = mx.expand_dims(k, 2)
        v_e = mx.expand_dims(v, 2)
    else:
        k_e = k
        v_e = v

    scores = q_s @ mx.swapaxes(k_e, -1, -2)
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = scores @ v_e

    if n_repeats > 1:
        out = mx.reshape(out, [B, n_q_heads, L, -1])
    return out


def mlx_fused_attn(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)


def do_vjp_bench(f, q, k, v, scale):
    """Chain VJP calls by accumulating gradients to force all iterations to compute."""

    def loss_fn(q, k, v):
        return f(q, k, v, scale).sum()

    grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2))
    dq, dk, dv = grad_fn(q, k, v)
    for i in range(N_iter_func - 1):
        dq_i, dk_i, dv_i = grad_fn(q, k, v)
        dq = dq + dq_i
        dk = dk + dk_i
        dv = dv + dv_i

    mx.eval(dq, dk, dv)


def bench_shape(B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, dtype):
    q_mx, k_mx, v_mx, scale = prepare_inputs(
        B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, dtype
    )

    # Warmup both paths
    for _ in range(N_warmup):
        do_vjp_bench(mlx_ref_attn, q_mx, k_mx, v_mx, scale)
        do_vjp_bench(mlx_fused_attn, q_mx, k_mx, v_mx, scale)

    # Interleaved measurement for thermal fairness
    times_unfused = []
    times_fused = []
    for _ in range(N_iter_bench):
        s = time.perf_counter_ns()
        do_vjp_bench(mlx_ref_attn, q_mx, k_mx, v_mx, scale)
        e = time.perf_counter_ns()
        times_unfused.append((e - s) * 1e-9)

        s = time.perf_counter_ns()
        do_vjp_bench(mlx_fused_attn, q_mx, k_mx, v_mx, scale)
        e = time.perf_counter_ns()
        times_fused.append((e - s) * 1e-9)

    times_unfused.sort()
    times_fused.sort()

    def stats(t):
        p50 = t[len(t) // 2]
        p90 = t[int(len(t) * 0.9)]
        return p50, p90

    fused_p50, fused_p90 = stats(times_fused)
    unfused_p50, unfused_p90 = stats(times_unfused)

    # Correctness check
    def loss_ref(q, k, v):
        return mlx_ref_attn(q, k, v, scale).sum()

    def loss_fused(q, k, v):
        return mlx_fused_attn(q, k, v, scale).sum()

    grads_ref = mx.grad(loss_ref, argnums=(0, 1, 2))(q_mx, k_mx, v_mx)
    grads_fused = mx.grad(loss_fused, argnums=(0, 1, 2))(q_mx, k_mx, v_mx)
    mx.eval(grads_ref, grads_fused)

    atol = 1e-5 if dtype == "float32" else 1e-2
    for i, name in enumerate(["dQ", "dK", "dV"]):
        if not mx.allclose(grads_ref[i], grads_fused[i], atol=atol, rtol=atol):
            max_diff = mx.max(mx.abs(grads_ref[i] - grads_fused[i]))
            print(
                f"  {name} MISMATCH (B={B}, qsl={qsl}, ksl={ksl}, D={head_dim}, "
                f"qH={n_q_heads}, kvH={n_kv_heads}) max|diff|={max_diff:3.2e}"
            )

    return (fused_p50, fused_p90), (unfused_p50, unfused_p90)


mlx_ref_attn = mlx_ref_vjp  # used as attention function (not grad)


if __name__ == "__main__":
    print(f"Device: {device_name}")
    print("Note: Measurements use interleaved fused/unfused runs for thermal fairness.")
    print("      P50 (median) used for speedup; P90 also reported.")
    print("      Early 'cold GPU' runs can be misleading for compute-heavy kernels.")
    print()

    # Print dispatch environment
    import os
    vjp_mode = os.environ.get("MLX_SDPA_VJP_MODE", "auto")
    vjp_l_thresh = os.environ.get("MLX_SDPA_VJP_LONG_L_THRESHOLD", "8192")
    vjp_bytes_thresh = os.environ.get("MLX_SDPA_VJP_ATTENTION_BYTES_THRESHOLD", "1073741824")
    print(f"Dispatch: MLX_SDPA_VJP_MODE={vjp_mode}, L_threshold={vjp_l_thresh}, "
          f"bytes_threshold={int(vjp_bytes_thresh)/1e9:.1f}GB")
    print()

    dtypes = ("float16", "float32")

    # --- Section 1: Vector VJP (qsl <= 8) ---
    print("=" * 85)
    print("  VECTOR VJP (query seq len <= 8)")
    print("=" * 85)
    # fmt: off
    vector_shapes = (
        # (  B,  qsl,   ksl, head_dim, n_qh, n_kvh)
          (  1,    1,   512,      128,   32,    32),
          (  1,    1,  2048,      128,   32,    32),
          (  1,    1,  4096,      128,   32,    32),
          (  1,    1,  8192,      128,   32,    32),
          (  1,    1, 16384,      128,   32,    32),
          (  1,    1,  2048,       64,   32,    32),
          (  1,    1,  2048,       96,   32,    32),
          (  1,    4,  2048,      128,   32,    32),
          (  1,    8,  2048,      128,   32,    32),
          # D=256
          (  1,    1,  2048,      256,   32,    32),
          (  1,    4,  2048,      256,   32,    32),
          # GQA configurations
          (  1,    1,  2048,      128,   32,     8),
          (  1,    4,  2048,      128,   32,     8),
    )
    # fmt: on

    print("  B,  qsl,   ksl, hdim, n_qh, n_kvh,   dtype, unf_p50, fus_p50, speedup, unf_p90, fus_p90")

    for dtype in dtypes:
        for B, qsl, ksl, head_dim, n_q_heads, n_kv_heads in vector_shapes:
            (fused_p50, fused_p90), (unfused_p50, unfused_p90) = bench_shape(
                B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, dtype
            )
            speedup = unfused_p50 / fused_p50
            print(
                f"{B:3d}, {qsl:4d}, {ksl:5d}, {head_dim:4d}, {n_q_heads:4d}, "
                f"{n_kv_heads:5d}, {dtype:>8s}, {unfused_p50: 2.3f}, "
                f"{fused_p50: 2.3f}, {speedup:5.2f}x, {unfused_p90: 2.3f}, {fused_p90: 2.3f}"
            )

    # --- Section 2: STEEL VJP (qsl > 8) ---
    print()
    print("=" * 85)
    print("  STEEL VJP (query seq len > 8)")
    print("=" * 85)
    # fmt: off
    steel_shapes = (
        # (  B,  qsl,   ksl, head_dim, n_qh, n_kvh)
        # D=64 — fused VJP happy path
          (  1,   32,    32,       64,   32,    32),
          (  1,  128,   128,       64,   32,    32),
          (  1,  512,   512,       64,   32,    32),
          (  1, 1024,  1024,       64,   32,    32),
          (  1, 2048,  2048,       64,   32,    32),
        # Unaligned (D=64)
          (  1,  100,   100,       64,   32,    32),
        # GQA with D=64
          (  1,  512,   512,       64,   32,     8),
          (  1, 1024,  1024,       64,   32,     8),
    )
    # fmt: on

    print("  B,  qsl,   ksl, hdim, n_qh, n_kvh,   dtype, unf_p50, fus_p50, speedup, unf_p90, fus_p90")

    for dtype in dtypes:
        for B, qsl, ksl, head_dim, n_q_heads, n_kv_heads in steel_shapes:
            (fused_p50, fused_p90), (unfused_p50, unfused_p90) = bench_shape(
                B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, dtype
            )
            speedup = unfused_p50 / fused_p50
            print(
                f"{B:3d}, {qsl:4d}, {ksl:5d}, {head_dim:4d}, {n_q_heads:4d}, "
                f"{n_kv_heads:5d}, {dtype:>8s}, {unfused_p50: 2.3f}, "
                f"{fused_p50: 2.3f}, {speedup:5.2f}x, {unfused_p90: 2.3f}, {fused_p90: 2.3f}"
            )

    # --- Section 2b: Reference unfused backward ---
    print()
    print("=" * 85)
    print("  Reference: Unfused Backward (D=96/128, fused VJP disabled)")
    print("=" * 85)
    # fmt: off
    unfused_ref_shapes = (
        # These shapes use unfused backward for both paths (fused VJP only supports D=64)
        (  1,  512,   512,       96,   32,    32),
        (  1, 1024,  1024,       96,   32,    32),
        (  1,  512,   512,      128,   32,    32),
        (  1, 1024,  1024,      128,   32,    32),
        (  1, 2048,  2048,      128,   32,    32),
    )
    # fmt: on

    print("  B,  qsl,   ksl, hdim, n_qh, n_kvh,   dtype, unf_p50, fus_p50, speedup, unf_p90, fus_p90")
    print("  (Note: Both columns use unfused backward for D=96/128)")

    for dtype in dtypes:
        for B, qsl, ksl, head_dim, n_q_heads, n_kv_heads in unfused_ref_shapes:
            (fused_p50, fused_p90), (unfused_p50, unfused_p90) = bench_shape(
                B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, dtype
            )
            speedup = unfused_p50 / fused_p50
            print(
                f"{B:3d}, {qsl:4d}, {ksl:5d}, {head_dim:4d}, {n_q_heads:4d}, "
                f"{n_kv_heads:5d}, {dtype:>8s}, {unfused_p50: 2.3f}, "
                f"{fused_p50: 2.3f}, {speedup:5.2f}x, {unfused_p90: 2.3f}, {fused_p90: 2.3f}"
            )

    # --- Section 3: Memory usage ---
    print()
    print("=" * 85)
    print("  MEMORY USAGE (peak bytes during VJP, float16, B=1, H=32)")
    print("=" * 85)
    print(f"{'Config':>30s}, {'Unfused':>12s}, {'Fused':>12s}, {'Savings':>8s}")

    mem_configs = [
        (1, 512, 512, 64, 32, 32),
        (1, 1024, 1024, 64, 32, 32),
        (1, 2048, 2048, 64, 32, 32),
        (1, 4096, 4096, 64, 32, 32),
        (1, 1, 2048, 64, 32, 32),  # vector path for reference
    ]

    # Theoretical memory analysis: unfused stores full attention matrix [B, H, qL, kL]
    print(f"\n{'Theoretical attention matrix sizes (unfused must store, fused avoids):':}")
    for B, qsl, ksl, head_dim, n_q_heads, n_kv_heads in mem_configs:
        attn_bytes = B * n_q_heads * qsl * ksl * 2  # float16
        lse_bytes = B * n_q_heads * qsl * 4  # float32 LSE
        label = f"D={head_dim},qL={qsl},kL={ksl}"
        print(f"  {label:>30s}: attn_matrix={attn_bytes/1e6:.1f}MB, LSE={lse_bytes/1e6:.3f}MB")

    print(f"\n{'Note: Both fused and unfused paths may avoid full L×L materialization':}")
    print(f"{'due to MLX lazy evaluation and op fusion. Memory savings from fused VJP':}")
    print(f"{'come from not storing the attention matrix as an intermediate for backward.':}")

    print(f"\n{'Measured peak memory (allocator-reported, may be affected by caching/pooling):':}")
    print(f"{'Config':>30s}, {'Unfused':>12s}, {'Fused':>12s}, {'Savings':>8s}, {'Attn Matrix':>12s}")

    for B, qsl, ksl, head_dim, n_q_heads, n_kv_heads in mem_configs:
        _scale = 1.0 / math.sqrt(head_dim)
        attn_bytes = B * n_q_heads * qsl * ksl * 2  # float16

        # Measure unfused
        mx.clear_cache()
        q, k, v, scale = prepare_inputs(B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, "float16")
        mx.eval(q, k, v)

        def loss_ref(q, k, v):
            return mlx_ref_attn(q, k, v, _scale).sum()

        grad_ref = mx.grad(loss_ref, argnums=(0, 1, 2))
        mx.reset_peak_memory()
        mx.eval(grad_ref(q, k, v))
        mem_unfused = mx.get_peak_memory()

        # Measure fused
        mx.clear_cache()
        q, k, v, scale = prepare_inputs(B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, "float16")
        mx.eval(q, k, v)

        def loss_fused(q, k, v):
            return mlx_fused_attn(q, k, v, _scale).sum()

        grad_fused = mx.grad(loss_fused, argnums=(0, 1, 2))
        mx.reset_peak_memory()
        mx.eval(grad_fused(q, k, v))
        mem_fused = mx.get_peak_memory()

        savings = 1.0 - mem_fused / mem_unfused if mem_unfused > 0 else 0.0
        label = f"D={head_dim},qL={qsl},kL={ksl}"
        print(
            f"{label:>30s}, {mem_unfused / 1e6:>10.1f}MB, "
            f"{mem_fused / 1e6:>10.1f}MB, {100*savings:>6.1f}%, "
            f"{attn_bytes / 1e6:>10.1f}MB"
        )

    print()
    print("Note: Measured peak memory is allocator-reported and may be affected by")
    print("caching and memory pooling. For authoritative resource allocation data,")
    print("use Xcode Metal GPU capture (see profile_sdpa_vjp.py --capture).")
    print("Refs: developer.apple.com/documentation/xcode/analyzing-the-memory-usage-of-your-metal-app")
    print("      developer.apple.com/documentation/xcode/analyzing-your-metal-workload")
