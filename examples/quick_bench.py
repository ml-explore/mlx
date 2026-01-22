#!/usr/bin/env python3
"""Quick benchmark - minimal overhead."""

import time
import numpy as np
import mlx.core as mx

PROB_BITS = 14
PROB_SCALE = 1 << PROB_BITS
RANS_BYTE_L = 1 << 23

def quick_encode(indices, n_streams=64):
    """Fast encoding for benchmark."""
    indices = indices.flatten().astype(np.uint32)
    counts = np.bincount(indices, minlength=16)
    counts = np.maximum(counts, 1).astype(np.float64)
    scaled = (counts / counts.sum() * PROB_SCALE).astype(np.int64)
    scaled[np.argmax(counts)] += PROB_SCALE - scaled.sum()
    
    freq = scaled.astype(np.uint16)
    cumfreq = np.zeros(17, dtype=np.uint16)
    cumfreq[1:] = np.cumsum(freq)
    
    sym_table = np.zeros(PROB_SCALE, dtype=np.uint8)
    for s in range(16):
        sym_table[cumfreq[s]:cumfreq[s+1]] = s
    
    # Simple encoding
    stream_data = []
    stream_lens = []
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
    
    return matrix.T.flatten(), stream_lens, max_len, freq, cumfreq[:16], sym_table

print("="*50)
print("Quick Entropy-Coded Benchmark")
print("="*50)

# Small test
out_dim, in_dim = 64, 128
n_streams = 64

np.random.seed(42)
w = np.random.randn(out_dim, in_dim).astype(np.float32) * 0.02
w_min, w_max = w.min(), w.max()
scale = (w_max - w_min) / 15
indices = np.clip(np.round((w - w_min) / scale), 0, 15).astype(np.uint8)

# Entropy
counts = np.bincount(indices.flatten(), minlength=16)
probs = counts / counts.sum()
entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
print(f"Shape: {out_dim}x{in_dim}, Entropy: {entropy:.2f} bits")

# Encode
compressed, stream_lens, max_len, freq, cumfreq, sym_table = quick_encode(indices, n_streams)
ratio = (out_dim * in_dim * 0.5) / len(compressed)
print(f"Compression: {ratio:.2f}x over 4-bit")

# MLX arrays
mx_comp = mx.array(compressed)
mx_lens = mx.array(np.array(stream_lens, dtype=np.uint32))
mx_freq = mx.array(freq)
mx_cumfreq = mx.array(cumfreq)
mx_sym = mx.array(sym_table)
mx_x = mx.random.normal((in_dim,))
mx_scales = mx.array(np.full(out_dim, scale, dtype=np.float32))
mx_biases = mx.array(np.full(out_dim, w_min, dtype=np.float32))

# Warmup
y = mx.entropy_coded_matmul(
    mx_comp, mx_lens, mx_freq, mx_cumfreq, mx_sym, mx_x,
    mx_scales, mx_biases, n_streams, out_dim * in_dim, max_len, out_dim, in_dim, in_dim
)
mx.eval(y)

# Benchmark entropy kernel
n_iters = 10
mx.synchronize()
start = time.perf_counter()
for _ in range(n_iters):
    y = mx.entropy_coded_matmul(
        mx_comp, mx_lens, mx_freq, mx_cumfreq, mx_sym, mx_x,
        mx_scales, mx_biases, n_streams, out_dim * in_dim, max_len, out_dim, in_dim, in_dim
    )
    mx.eval(y)
mx.synchronize()
ec_time = (time.perf_counter() - start) / n_iters * 1000

# Standard path
indices_mx = mx.array(indices.astype(np.float32))
weights = indices_mx * scale + w_min
mx.eval(weights)

mx.synchronize()
start = time.perf_counter()
for _ in range(n_iters):
    y_std = mx_x @ weights.T
    mx.eval(y_std)
mx.synchronize()
std_time = (time.perf_counter() - start) / n_iters * 1000

print(f"\nResults ({n_iters} iters):")
print(f"  Standard matmul: {std_time:.3f} ms")
print(f"  Entropy-coded:   {ec_time:.3f} ms")
print(f"  Ratio: {ec_time/std_time:.2f}x")

# Verify correctness
y_ec = mx.entropy_coded_matmul(
    mx_comp, mx_lens, mx_freq, mx_cumfreq, mx_sym, mx_x,
    mx_scales, mx_biases, n_streams, out_dim * in_dim, max_len, out_dim, in_dim, in_dim
)
y_ref = weights @ mx_x
mx.eval(y_ec, y_ref)
diff = float(mx.max(mx.abs(y_ec - y_ref)))
print(f"  Max diff: {diff:.6f} ✓" if diff < 0.01 else f"  Max diff: {diff:.6f} ✗")
