#!/usr/bin/env python3
"""Benchmark V1 (flat encoding) vs V2 (per-row encoding)."""

import time
import numpy as np
import mlx.core as mx

PROB_BITS = 14
PROB_SCALE = 1 << PROB_BITS
RANS_BYTE_L = 1 << 23


def build_table(counts):
    counts = np.maximum(counts, 1).astype(np.float64)
    scaled = (counts / counts.sum() * PROB_SCALE).astype(np.int64)
    scaled[np.argmax(counts)] += PROB_SCALE - scaled.sum()
    freq = scaled.astype(np.uint16)
    cumfreq = np.zeros(17, dtype=np.uint16)
    cumfreq[1:] = np.cumsum(freq)
    sym_table = np.zeros(PROB_SCALE, dtype=np.uint8)
    for s in range(16):
        sym_table[cumfreq[s]:cumfreq[s+1]] = s
    return freq, cumfreq[:16], sym_table


def encode_flat(indices, freq, cumfreq, n_streams=64):
    """Flat encoding (V1) - entire matrix as one."""
    indices = indices.flatten().astype(np.uint32)
    stream_data, stream_lens = [], []
    for i in range(n_streams):
        syms = indices[i::n_streams]
        state = RANS_BYTE_L
        out = []
        for s in reversed(syms):
            f, c = int(freq[s]), int(cumfreq[s])
            while state >= ((RANS_BYTE_L >> PROB_BITS) << 8) * f:
                out.append(state & 0xFF)
                state >>= 8
            state = ((state // f) << PROB_BITS) + (state % f) + c
        out.extend([(state >> i*8) & 0xFF for i in range(4)])
        stream_data.append(bytes(reversed(out)))
        stream_lens.append(len(stream_data[-1]))
    max_len = max(stream_lens)
    matrix = np.zeros((n_streams, max_len), dtype=np.uint8)
    for i, d in enumerate(stream_data):
        matrix[i, :len(d)] = np.frombuffer(d, dtype=np.uint8)
    return matrix.T.flatten(), stream_lens, max_len


def encode_per_row(indices_2d, freq, cumfreq, n_streams=64):
    """Per-row encoding (V2) - each row independent."""
    out_dim, in_dim = indices_2d.shape
    all_compressed, row_offsets, all_stream_lens = [], [0], []
    
    for row in range(out_dim):
        row_data, lens, _ = encode_flat(indices_2d[row], freq, cumfreq, n_streams)
        all_compressed.append(row_data)
        all_stream_lens.append(lens)
        row_offsets.append(row_offsets[-1] + len(row_data))
    
    return (np.concatenate(all_compressed), 
            np.array(row_offsets[:-1], dtype=np.uint32),
            np.array(all_stream_lens, dtype=np.uint32).flatten())


print("="*60)
print("Fused Kernel Benchmark: V1 (flat) vs V2 (per-row)")
print("="*60)

# Test sizes
sizes = [(32, 64), (64, 128), (128, 256)]

for out_dim, in_dim in sizes:
    print(f"\n--- {out_dim}x{in_dim} ({out_dim*in_dim:,} params) ---")
    
    np.random.seed(42)
    w = np.random.randn(out_dim, in_dim).astype(np.float32) * 0.02
    w_min, w_max = w.min(), w.max()
    scale = (w_max - w_min) / 15
    indices = np.clip(np.round((w - w_min) / scale), 0, 15).astype(np.uint8)
    
    counts = np.bincount(indices.flatten(), minlength=16)
    freq, cumfreq, sym_table = build_table(counts)
    
    n_streams = 64
    
    # V1 encoding
    comp_v1, lens_v1, max_len_v1 = encode_flat(indices, freq, cumfreq, n_streams)
    
    # V2 encoding
    comp_v2, offsets_v2, lens_v2 = encode_per_row(indices, freq, cumfreq, n_streams)
    
    # MLX arrays
    mx_x = mx.random.normal((in_dim,))
    mx_scales = mx.array(np.full(out_dim, scale, dtype=np.float32))
    mx_biases = mx.array(np.full(out_dim, w_min, dtype=np.float32))
    mx_freq = mx.array(freq)
    mx_cumfreq = mx.array(cumfreq)
    mx_sym = mx.array(sym_table)
    
    # V1 arrays
    mx_comp_v1 = mx.array(comp_v1)
    mx_lens_v1 = mx.array(np.array(lens_v1, dtype=np.uint32))
    
    # V2 arrays
    mx_comp_v2 = mx.array(comp_v2)
    mx_offsets_v2 = mx.array(offsets_v2)
    mx_lens_v2 = mx.array(lens_v2)
    
    # Warmup V1
    y1 = mx.entropy_coded_matmul(
        mx_comp_v1, mx_lens_v1, mx_freq, mx_cumfreq, mx_sym, mx_x,
        mx_scales, mx_biases, n_streams, out_dim * in_dim, max_len_v1,
        out_dim, in_dim, in_dim
    )
    mx.eval(y1)
    
    # Warmup V2
    y2 = mx.entropy_coded_matmul_v2(
        mx_comp_v2, mx_offsets_v2, mx_lens_v2, mx_freq, mx_cumfreq, mx_sym,
        mx_x, mx_scales, mx_biases, n_streams, in_dim, out_dim
    )
    mx.eval(y2)
    
    # Benchmark V1
    n_iters = 20
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        y1 = mx.entropy_coded_matmul(
            mx_comp_v1, mx_lens_v1, mx_freq, mx_cumfreq, mx_sym, mx_x,
            mx_scales, mx_biases, n_streams, out_dim * in_dim, max_len_v1,
            out_dim, in_dim, in_dim
        )
        mx.eval(y1)
    mx.synchronize()
    v1_ms = (time.perf_counter() - t0) / n_iters * 1000
    
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
    v2_ms = (time.perf_counter() - t0) / n_iters * 1000
    
    # Standard matmul for comparison
    weights = mx.array(indices.astype(np.float32) * scale + w_min)
    mx.eval(weights)
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        y_std = mx_x @ weights.T
        mx.eval(y_std)
    mx.synchronize()
    std_ms = (time.perf_counter() - t0) / n_iters * 1000
    
    print(f"  Standard matmul: {std_ms:.3f} ms")
    print(f"  V1 (flat):       {v1_ms:.3f} ms ({v1_ms/std_ms:.1f}x overhead)")
    print(f"  V2 (per-row):    {v2_ms:.3f} ms ({v2_ms/std_ms:.1f}x overhead)")
    print(f"  V2 speedup vs V1: {v1_ms/v2_ms:.1f}x")

print("\n" + "="*60)
print("Summary: V2 per-row encoding reduces decode work from")
print("O(rows × total_symbols) to O(total_symbols)")
print("="*60)
