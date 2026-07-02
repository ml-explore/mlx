# Copyright © 2026 Apple Inc.
#
# Benchmark for SDPA backward (VJP) kernels.
#
# Tests three VJP paths:
#   1. Vector VJP   — query seq len <= 8 (decode-time attention)
#   2. Steel VJP    — fused two-kernel backward for D={64,96,128}, float16/bfloat16
#   3. Unfused VJP  — materialized attention matrix backward (reference)
#
# The benchmark explicitly forces fused vs unfused modes via MLX_SDPA_VJP_MODE
# to isolate each path's performance. Auto-dispatch threshold validation is
# tested separately.
#
# Usage:
#   python sdpa_vector_vjp_bench.py [--section vector|steel|memory|auto|all]
#
# Environment:
#   MLX_SDPA_VJP_LONG_L_THRESHOLD       — auto-dispatch L threshold (default 1024)
#   MLX_SDPA_VJP_ATTENTION_BYTES_THRESHOLD — auto-dispatch bytes threshold (default 128MB)

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


def prepare_inputs(B, qL, kL, D, qH, kH, dtype):
    np_dtype = getattr(np, dtype)
    scale = 1.0 / math.sqrt(D)
    q_np = np.random.normal(0.0, 1.0, (B, qH, qL, D)).astype(np_dtype)
    k_np = np.random.normal(0.0, scale, (B, kH, kL, D)).astype(np_dtype)
    v_np = np.random.normal(0.0, scale, (B, kH, kL, D)).astype(np_dtype)
    return mx.array(q_np), mx.array(k_np), mx.array(v_np), scale


def mlx_ref_attn(q, k, v, scale, causal=False):
    """Unfused attention: materialize full attention matrix."""
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
    if causal:
        mask = mx.triu(mx.full(scores.shape[-2:], float("-inf"), dtype=scores.dtype), k=1)
        scores = scores + mask
    scores = mx.softmax(scores, axis=-1, precise=True)
    out = scores @ v_e

    if n_repeats > 1:
        out = mx.reshape(out, [B, n_q_heads, L, -1])
    return out


def mlx_fused_attn(q, k, v, scale, causal=False):
    """Fused SDPA — dispatches to vector, steel, or unfused VJP based on mode."""
    mask = "causal" if causal else None
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def do_vjp_bench(f, q, k, v, scale, causal=False):
    """Run VJP N_iter_func times, accumulating gradients to force computation."""
    def loss_fn(q, k, v):
        return f(q, k, v, scale, causal=causal).sum()

    grad_fn = mx.grad(loss_fn, argnums=(0, 1, 2))
    dq, dk, dv = grad_fn(q, k, v)
    for _ in range(N_iter_func - 1):
        dq_i, dk_i, dv_i = grad_fn(q, k, v)
        dq = dq + dq_i
        dk = dk + dk_i
        dv = dv + dv_i
    mx.eval(dq, dk, dv)


def bench_shape(B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, dtype,
                causal=False, ref_fn=mlx_ref_attn, fused_fn=mlx_fused_attn):
    q, k, v, scale = prepare_inputs(B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, dtype)

    # Warmup both paths
    for _ in range(N_warmup):
        do_vjp_bench(ref_fn, q, k, v, scale, causal)
        do_vjp_bench(fused_fn, q, k, v, scale, causal)

    # Interleaved measurement for thermal fairness
    times_unfused = []
    times_fused = []
    for _ in range(N_iter_bench):
        s = time.perf_counter_ns()
        do_vjp_bench(ref_fn, q, k, v, scale, causal)
        e = time.perf_counter_ns()
        times_unfused.append((e - s) * 1e-9)

        s = time.perf_counter_ns()
        do_vjp_bench(fused_fn, q, k, v, scale, causal)
        e = time.perf_counter_ns()
        times_fused.append((e - s) * 1e-9)

    times_unfused.sort()
    times_fused.sort()

    def stats(t):
        return t[len(t) // 2], t[int(len(t) * 0.9)]

    fused_p50, fused_p90 = stats(times_fused)
    unfused_p50, unfused_p90 = stats(times_unfused)

    # Correctness check
    def loss_ref(q, k, v):
        return ref_fn(q, k, v, scale, causal=causal).sum()

    def loss_fused(q, k, v):
        return fused_fn(q, k, v, scale, causal=causal).sum()

    grads_ref = mx.grad(loss_ref, argnums=(0, 1, 2))(q, k, v)
    grads_fused = mx.grad(loss_fused, argnums=(0, 1, 2))(q, k, v)
    mx.eval(grads_ref, grads_fused)

    atol = 1e-5 if dtype == "float32" else 1e-2
    for i, name in enumerate(["dQ", "dK", "dV"]):
        if not mx.allclose(grads_ref[i], grads_fused[i], atol=atol, rtol=atol):
            max_diff = mx.max(mx.abs(grads_ref[i] - grads_fused[i]))
            print(
                f"  ** {name} MISMATCH (B={B}, qsl={qsl}, ksl={ksl}, D={head_dim}, "
                f"qH={n_q_heads}, kvH={n_kv_heads}, {'causal' if causal else 'dense'}) "
                f"max|diff|={max_diff:3.2e}"
            )

    return (fused_p50, fused_p90), (unfused_p50, unfused_p90)


def print_header():
    print(
        f"{'B':>3s}, {'qsl':>5s}, {'ksl':>5s}, {'D':>4s}, {'qH':>4s}, {'kvH':>4s}, "
        f"{'mask':>6s}, {'dtype':>8s}, {'unf_p50':>8s}, {'fus_p50':>8s}, "
        f"{'speedup':>8s}, {'unf_p90':>8s}, {'fus_p90':>8s}"
    )


def print_row(B, qsl, ksl, D, qH, kvH, causal, dtype, unfused_p50, fused_p50,
              unfused_p90, fused_p90):
    speedup = unfused_p50 / fused_p50 if fused_p50 > 0 else float("inf")
    mask_str = "causal" if causal else "dense"
    print(
        f"{B:3d}, {qsl:5d}, {ksl:5d}, {D:4d}, {qH:4d}, {kvH:4d}, "
        f"{mask_str:>6s}, {dtype:>8s}, {unfused_p50:8.3f}, {fused_p50:8.3f}, "
        f"{speedup:7.2f}x, {unfused_p90:8.3f}, {fused_p90:8.3f}"
    )


def run_section(title, shapes, dtypes, set_mode=None):
    """Run a benchmark section. set_mode overrides MLX_SDPA_VJP_MODE."""
    print()
    print("=" * 100)
    print(f"  {title}")
    if set_mode:
        print(f"  (MLX_SDPA_VJP_MODE={set_mode})")
    print("=" * 100)
    print_header()

    old_mode = os.environ.get("MLX_SDPA_VJP_MODE")
    if set_mode:
        os.environ["MLX_SDPA_VJP_MODE"] = set_mode

    try:
        for dtype in dtypes:
            for B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, causal in shapes:
                (fused_p50, fused_p90), (unfused_p50, unfused_p90) = bench_shape(
                    B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, dtype, causal=causal
                )
                print_row(B, qsl, ksl, head_dim, n_q_heads, n_kv_heads, causal, dtype,
                          unfused_p50, fused_p50, unfused_p90, fused_p90)
    finally:
        if set_mode:
            if old_mode is not None:
                os.environ["MLX_SDPA_VJP_MODE"] = old_mode
            else:
                os.environ.pop("MLX_SDPA_VJP_MODE", None)


def run_vector_section():
    """Vector VJP: decode-time attention (qsl <= 8). All dtypes, all D values."""
    # fmt: off
    shapes = (
        # (  B,  qsl,   ksl, hdim, n_qh, n_kvh, causal)
          (  1,    1,   512,  128,   32,    32,  False),
          (  1,    1,  2048,  128,   32,    32,  False),
          (  1,    1,  4096,  128,   32,    32,  False),
          (  1,    1,  8192,  128,   32,    32,  False),
          (  1,    1, 16384,  128,   32,    32,  False),
          (  1,    1,  2048,   64,   32,    32,  False),
          (  1,    1,  2048,   96,   32,    32,  False),
          (  1,    4,  2048,  128,   32,    32,  False),
          (  1,    8,  2048,  128,   32,    32,  False),
          # D=256
          (  1,    1,  2048,  256,   32,    32,  False),
          (  1,    4,  2048,  256,   32,    32,  False),
          # GQA
          (  1,    1,  2048,  128,   32,     8,  False),
          (  1,    4,  2048,  128,   32,     8,  False),
    )
    # fmt: on
    run_section("VECTOR VJP (query seq len <= 8)", shapes, ("float16", "float32"))


def run_steel_section():
    """Steel VJP: fused two-kernel backward. Force fused mode to isolate performance."""
    # fmt: off
    shapes = (
        # (  B,  qsl,   ksl, hdim, n_qh, n_kvh, causal)
        # --- D=64 dense ---
          (  1,  512,   512,   64,   32,    32,  False),
          (  1, 1024,  1024,   64,   32,    32,  False),
          (  1, 2048,  2048,   64,   32,    32,  False),
          (  1, 4096,  4096,   64,   32,    32,  False),
        # --- D=64 causal ---
          (  1,  512,   512,   64,   32,    32,   True),
          (  1, 1024,  1024,   64,   32,    32,   True),
          (  1, 2048,  2048,   64,   32,    32,   True),
          (  1, 4096,  4096,   64,   32,    32,   True),
        # --- D=96 dense ---
          (  1,  512,   512,   96,   32,    32,  False),
          (  1, 1024,  1024,   96,   32,    32,  False),
          (  1, 2048,  2048,   96,   32,    32,  False),
        # --- D=96 causal ---
          (  1,  512,   512,   96,   32,    32,   True),
          (  1, 1024,  1024,   96,   32,    32,   True),
          (  1, 2048,  2048,   96,   32,    32,   True),
        # --- D=128 dense ---
          (  1,  512,   512,  128,   32,    32,  False),
          (  1, 1024,  1024,  128,   32,    32,  False),
          (  1, 2048,  2048,  128,   32,    32,  False),
        # --- D=128 causal ---
          (  1,  512,   512,  128,   32,    32,   True),
          (  1, 1024,  1024,  128,   32,    32,   True),
          (  1, 2048,  2048,  128,   32,    32,   True),
        # --- GQA (D=64) ---
          (  1,  512,   512,   64,   32,     8,  False),
          (  1, 1024,  1024,   64,   32,     8,  False),
          (  1, 2048,  2048,   64,   32,     8,  False),
        # --- GQA (D=128) ---
          (  1,  512,   512,  128,   32,     8,  False),
          (  1, 1024,  1024,  128,   32,     8,  False),
          (  1, 2048,  2048,  128,   32,     8,  False),
        # --- Unaligned ---
          (  1,  100,   100,   64,   32,    32,  False),
          (  1,  100,   100,  128,   32,    32,  False),
        # --- Batch > 1 ---
          (  2,  512,   512,  128,   32,    32,  False),
          (  4,  256,   256,  128,   32,    32,  False),
    )
    # fmt: on
    # Only test dtypes that steel VJP actually supports
    run_section(
        "STEEL VJP — Fused vs Unfused (D=64/96/128, float16 only)",
        shapes,
        ("float16",),
        set_mode="fused",
    )


def run_auto_section():
    """Test auto-dispatch threshold behavior. Uses default auto mode."""
    # fmt: off
    shapes = (
        # (  B,  qsl,   ksl, hdim, n_qh, n_kvh, causal)
        # L boundary around default threshold (1024)
          (  1,  512,   512,   64,   32,    32,  False),
          (  1, 1023,  1023,   64,   32,    32,  False),
          (  1, 1024,  1024,   64,   32,    32,  False),
          (  1, 1025,  1025,   64,   32,    32,  False),
          (  1, 2048,  2048,   64,   32,    32,  False),
        # Same for D=128
          (  1,  512,   512,  128,   32,    32,  False),
          (  1, 1023,  1023,  128,   32,    32,  False),
          (  1, 1024,  1024,  128,   32,    32,  False),
          (  1, 2048,  2048,  128,   32,    32,  False),
        # Batch scaling: B*H*L*L*2 >= 128MB triggers fused even for short L
        # B=4, H=32, L=512: 4*32*512*512*2 = 64MB (below threshold)
          (  4,  512,   512,  128,   32,    32,  False),
        # B=8, H=32, L=512: 8*32*512*512*2 = 128MB (at threshold)
          (  8,  512,   512,  128,   32,    32,  False),
    )
    # fmt: on
    l_thresh = os.environ.get("MLX_SDPA_VJP_LONG_L_THRESHOLD", "1024")
    bytes_thresh = os.environ.get("MLX_SDPA_VJP_ATTENTION_BYTES_THRESHOLD", str(1 << 30))
    run_section(
        f"AUTO-DISPATCH THRESHOLD (L_thresh={l_thresh}, bytes_thresh={int(bytes_thresh)/1e6:.0f}MB)",
        shapes,
        ("float16",),
        set_mode=None,  # Use whatever MLX_SDPA_VJP_MODE is set (default: auto)
    )


def run_memory_section():
    """Memory usage comparison: fused avoids O(L^2) attention matrix."""
    print()
    print("=" * 100)
    print("  MEMORY USAGE (peak bytes during VJP, float16)")
    print("=" * 100)

    # fmt: off
    mem_configs = [
        # (B, qsl, ksl, D, qH, kvH, causal)
        (1,  512,   512,   64, 32, 32, False),
        (1, 1024,  1024,   64, 32, 32, False),
        (1, 2048,  2048,   64, 32, 32, False),
        (1, 4096,  4096,   64, 32, 32, False),
        (1, 1024,  1024,   96, 32, 32, False),
        (1, 2048,  2048,   96, 32, 32, False),
        (1, 1024,  1024,  128, 32, 32, False),
        (1, 2048,  2048,  128, 32, 32, False),
        # Causal
        (1, 2048,  2048,   64, 32, 32, True),
        (1, 2048,  2048,  128, 32, 32, True),
        # GQA
        (1, 2048,  2048,  128, 32,  8, False),
    ]
    # fmt: on

    print(
        f"{'Config':>40s}, {'Unfused':>10s}, {'Fused':>10s}, "
        f"{'Savings':>8s}, {'Attn Matrix':>12s}"
    )

    for B, qsl, ksl, D, qH, kvH, causal in mem_configs:
        _scale = 1.0 / math.sqrt(D)
        attn_bytes = B * qH * qsl * ksl * 2  # float16
        mask_str = "causal" if causal else "dense"
        label = f"D={D},L={qsl},{mask_str},H={qH}{'/' + str(kvH) if qH != kvH else ''}"

        # Measure unfused
        os.environ["MLX_SDPA_VJP_MODE"] = "unfused"
        mx.clear_cache()
        q, k, v, scale = prepare_inputs(B, qsl, ksl, D, qH, kvH, "float16")
        mx.eval(q, k, v)

        def loss_ref(q, k, v):
            return mlx_ref_attn(q, k, v, _scale, causal=causal).sum()

        grad_ref = mx.grad(loss_ref, argnums=(0, 1, 2))
        mx.reset_peak_memory()
        mx.eval(grad_ref(q, k, v))
        mem_unfused = mx.get_peak_memory()

        # Measure fused
        os.environ["MLX_SDPA_VJP_MODE"] = "fused"
        mx.clear_cache()
        q, k, v, scale = prepare_inputs(B, qsl, ksl, D, qH, kvH, "float16")
        mx.eval(q, k, v)

        def loss_fused(q, k, v):
            return mlx_fused_attn(q, k, v, _scale, causal=causal).sum()

        grad_fused = mx.grad(loss_fused, argnums=(0, 1, 2))
        mx.reset_peak_memory()
        mx.eval(grad_fused(q, k, v))
        mem_fused = mx.get_peak_memory()

        os.environ.pop("MLX_SDPA_VJP_MODE", None)

        savings = 1.0 - mem_fused / mem_unfused if mem_unfused > 0 else 0.0
        print(
            f"{label:>40s}, {mem_unfused / 1e6:>8.1f}MB, "
            f"{mem_fused / 1e6:>8.1f}MB, {100 * savings:>6.1f}%, "
            f"{attn_bytes / 1e6:>10.1f}MB"
        )

    print()
    print("Note: Measured peak memory is allocator-reported and may be affected by")
    print("caching and memory pooling. Fused VJP avoids materializing the O(L^2)")
    print("attention matrix that unfused backward requires.")


# ─── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDPA VJP Benchmark")
    parser.add_argument(
        "--section",
        choices=["vector", "steel", "memory", "auto", "all"],
        default="all",
        help="Which section to run (default: all)",
    )
    args = parser.parse_args()

    print(f"Device: {device_name}")
    print()
    print("Benchmark measures fused (steel) vs unfused (materialized) VJP backward.")
    print("Speedup > 1.0x means fused is faster. P50 (median) of interleaved runs.")
    print()

    vjp_mode = os.environ.get("MLX_SDPA_VJP_MODE", "auto")
    l_thresh = os.environ.get("MLX_SDPA_VJP_LONG_L_THRESHOLD", "1024")
    bytes_thresh = os.environ.get("MLX_SDPA_VJP_ATTENTION_BYTES_THRESHOLD", str(1 << 30))
    print(f"Default dispatch: MLX_SDPA_VJP_MODE={vjp_mode}, "
          f"L_threshold={l_thresh}, bytes_threshold={int(bytes_thresh) / 1e6:.0f}MB")

    sections = {
        "vector": run_vector_section,
        "steel": run_steel_section,
        "auto": run_auto_section,
        "memory": run_memory_section,
    }

    if args.section == "all":
        for fn in sections.values():
            fn()
    else:
        sections[args.section]()
