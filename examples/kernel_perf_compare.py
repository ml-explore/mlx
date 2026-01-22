#!/usr/bin/env python3
"""Compare V1 vs V2 kernel performance with pre-generated data."""

import time
import numpy as np
import mlx.core as mx

print("="*60)
print("Kernel Performance: V1 (flat) vs V2 (per-row)")
print("="*60)

PROB_SCALE = 16384

# Pre-generate fake data for different sizes
sizes = [(64, 128), (128, 256), (256, 512)]

for out_dim, in_dim in sizes:
    print(f"\n--- {out_dim}x{in_dim} ---")
    
    n_streams = 64
    n_symbols = out_dim * in_dim
    
    # V1: Single flat encoded block
    max_len_v1 = n_symbols // n_streams + 10
    comp_v1 = np.random.randint(0, 256, n_streams * max_len_v1, dtype=np.uint8)
    lens_v1 = np.full(n_streams, max_len_v1, dtype=np.uint32)
    
    # V2: Per-row encoded blocks
    max_len_per_row = in_dim // n_streams + 4
    row_size = n_streams * max_len_per_row
    comp_v2 = np.random.randint(0, 256, out_dim * row_size, dtype=np.uint8)
    offsets_v2 = np.arange(out_dim, dtype=np.uint32) * row_size
    lens_v2 = np.full(out_dim * n_streams, max_len_per_row, dtype=np.uint32)
    
    # Frequency tables
    freq = np.full(16, PROB_SCALE // 16, dtype=np.uint16)
    cumfreq = np.arange(17, dtype=np.uint16) * (PROB_SCALE // 16)
    sym_table = np.repeat(np.arange(16, dtype=np.uint8), PROB_SCALE // 16)
    
    # MLX arrays
    mx_x = mx.random.normal((in_dim,))
    mx_scales = mx.ones((out_dim,))
    mx_biases = mx.zeros((out_dim,))
    mx_freq = mx.array(freq)
    mx_cumfreq = mx.array(cumfreq[:16])
    mx_sym = mx.array(sym_table)
    
    mx_comp_v1 = mx.array(comp_v1)
    mx_lens_v1 = mx.array(lens_v1)
    
    mx_comp_v2 = mx.array(comp_v2)
    mx_offsets_v2 = mx.array(offsets_v2)
    mx_lens_v2 = mx.array(lens_v2)
    
    # Warmup
    y1 = mx.entropy_coded_matmul(
        mx_comp_v1, mx_lens_v1, mx_freq, mx_cumfreq, mx_sym, mx_x,
        mx_scales, mx_biases, n_streams, n_symbols, max_len_v1,
        out_dim, in_dim, in_dim
    )
    mx.eval(y1)
    
    y2 = mx.entropy_coded_matmul_v2(
        mx_comp_v2, mx_offsets_v2, mx_lens_v2, mx_freq, mx_cumfreq, mx_sym,
        mx_x, mx_scales, mx_biases, n_streams, in_dim, out_dim
    )
    mx.eval(y2)
    
    # Benchmark V1
    n_iters = 50
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        y1 = mx.entropy_coded_matmul(
            mx_comp_v1, mx_lens_v1, mx_freq, mx_cumfreq, mx_sym, mx_x,
            mx_scales, mx_biases, n_streams, n_symbols, max_len_v1,
            out_dim, in_dim, in_dim
        )
        mx.eval(y1)
    mx.synchronize()
    v1_us = (time.perf_counter() - t0) / n_iters * 1e6
    
    # Benchmark V2
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        y2 = mx.entropy_coded_matmul_v2(
            mx_comp_v2, mx_offsets_v2, mx_lens_v2, mx_freq, mx_cumfreq, mx_sym,
            mx_x, mx_scales, mx_biases, n_streams, in_dim, out_dim
        )
        mx.eval(y2)
    mx.synchronize()
    v2_us = (time.perf_counter() - t0) / n_iters * 1e6
    
    # Theoretical decode ops
    v1_ops = out_dim * n_symbols  # Each row decodes ALL
    v2_ops = n_symbols            # Total across all rows
    
    print(f"  V1 (flat):    {v1_us:.0f} µs  (decodes {v1_ops:,} symbols)")
    print(f"  V2 (per-row): {v2_us:.0f} µs  (decodes {v2_ops:,} symbols)")
    print(f"  Speedup: {v1_us/v2_us:.1f}x")
    print(f"  Theory:  {v1_ops/v2_ops:.0f}x (rows={out_dim})")

print("\n" + "="*60)
print("V2 achieves O(n) decode vs V1's O(rows × n)")
print("="*60)
