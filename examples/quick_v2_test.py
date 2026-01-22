#!/usr/bin/env python3
"""Quick test of V2 kernel."""

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

print("Testing V2 kernel...")

# Small test
out_dim, in_dim = 8, 16
n_streams = 16

np.random.seed(42)
indices = np.random.randint(0, 16, size=(out_dim, in_dim), dtype=np.uint8)

counts = np.bincount(indices.flatten(), minlength=16)
freq, cumfreq, sym_table = build_table(counts)

# Create fake per-row data (just use indices directly for test)
# Each row: n_streams * max_len bytes
max_len_per_row = 8
row_size = n_streams * max_len_per_row

compressed = np.zeros(out_dim * row_size, dtype=np.uint8)
row_offsets = np.arange(out_dim, dtype=np.uint32) * row_size
row_stream_lens = np.full((out_dim, n_streams), max_len_per_row, dtype=np.uint32).flatten()

# Fill with some pattern
for row in range(out_dim):
    for s in range(n_streams):
        for b in range(max_len_per_row):
            compressed[row * row_size + b * n_streams + s] = (row + s + b) % 256

print(f"  compressed shape: {compressed.shape}")
print(f"  row_offsets: {row_offsets}")
print(f"  row_stream_lens shape: {row_stream_lens.shape}")

# MLX arrays
mx_comp = mx.array(compressed)
mx_offsets = mx.array(row_offsets)
mx_lens = mx.array(row_stream_lens)
mx_freq = mx.array(freq)
mx_cumfreq = mx.array(cumfreq)
mx_sym = mx.array(sym_table)
mx_x = mx.ones((in_dim,))
mx_scales = mx.ones((out_dim,))
mx_biases = mx.zeros((out_dim,))

print("  Calling V2 kernel...")
try:
    y = mx.entropy_coded_matmul_v2(
        mx_comp, mx_offsets, mx_lens, mx_freq, mx_cumfreq, mx_sym,
        mx_x, mx_scales, mx_biases, n_streams, in_dim, out_dim
    )
    mx.eval(y)
    print(f"  Output shape: {y.shape}")
    print(f"  Output: {y}")
    print("✓ V2 kernel works!")
except Exception as e:
    print(f"✗ Error: {e}")
