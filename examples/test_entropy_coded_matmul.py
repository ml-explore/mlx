#!/usr/bin/env python3
"""Test entropy-coded matmul integration."""

import numpy as np
import mlx.core as mx
import mlx.nn as nn

# rANS Constants
PROB_BITS = 14
PROB_SCALE = 1 << PROB_BITS
RANS_BYTE_L = 1 << 23


def build_rans_table(counts: np.ndarray, n_symbols: int = 16):
    """Build rANS frequency table from counts."""
    counts = np.maximum(counts, 1).astype(np.float64)
    total = counts.sum()
    
    scaled = (counts / total * PROB_SCALE).astype(np.int64)
    diff = PROB_SCALE - scaled.sum()
    scaled[np.argmax(counts)] += diff
    
    freq = scaled.astype(np.uint16)
    cumfreq = np.zeros(n_symbols + 1, dtype=np.uint16)
    cumfreq[1:] = np.cumsum(freq)
    
    sym_table = np.zeros(PROB_SCALE, dtype=np.uint8)
    for s in range(n_symbols):
        start = cumfreq[s]
        end = cumfreq[s + 1]
        sym_table[start:end] = s
    
    return freq, cumfreq[:n_symbols], sym_table


def entropy_encode(indices: np.ndarray, freq: np.ndarray, cumfreq: np.ndarray, 
                   n_streams: int = 256):
    """Encode indices using interleaved rANS."""
    indices = indices.flatten().astype(np.uint32)
    n = len(indices)
    
    stream_bytes_list = []
    stream_lengths = []
    
    for stream_idx in range(n_streams):
        syms = indices[stream_idx::n_streams]
        if len(syms) == 0:
            stream_bytes_list.append(b'')
            stream_lengths.append(0)
            continue
        
        out_bytes = []
        state = RANS_BYTE_L
        
        for i in range(len(syms) - 1, -1, -1):
            s = syms[i]
            freq_s = int(freq[s])
            start_s = int(cumfreq[s])
            
            x_max = ((RANS_BYTE_L >> PROB_BITS) << 8) * freq_s
            while state >= x_max:
                out_bytes.append(state & 0xFF)
                state >>= 8
            
            state = ((state // freq_s) << PROB_BITS) + (state % freq_s) + start_s
        
        out_bytes.extend([
            (state >> 0) & 0xFF,
            (state >> 8) & 0xFF,
            (state >> 16) & 0xFF,
            (state >> 24) & 0xFF
        ])
        
        encoded = bytes(reversed(out_bytes))
        stream_bytes_list.append(encoded)
        stream_lengths.append(len(encoded))
    
    # Physically interleave
    max_stream_len = max(stream_lengths) if stream_lengths else 0
    
    stream_matrix = np.zeros((n_streams, max_stream_len), dtype=np.uint8)
    for i, stream_data in enumerate(stream_bytes_list):
        if len(stream_data) > 0:
            stream_matrix[i, :len(stream_data)] = np.frombuffer(stream_data, dtype=np.uint8)
    
    interleaved = stream_matrix.T.flatten()
    
    return interleaved, stream_lengths, max_stream_len


def test_entropy_coded_matmul():
    """Test the fused entropy-coded matmul kernel."""
    print("Testing entropy_coded_matmul...")
    
    # Create a simple weight matrix
    out_dim, in_dim = 64, 128
    np.random.seed(42)
    weights = np.random.randn(out_dim, in_dim).astype(np.float32)
    
    # Quantize to 4-bit
    w_min = weights.min()
    w_max = weights.max()
    scale = (w_max - w_min) / 15
    indices = np.clip(np.round((weights - w_min) / scale), 0, 15).astype(np.uint8)
    
    # Create per-row scales and biases
    scales = np.full((out_dim,), scale, dtype=np.float32)
    biases = np.full((out_dim,), w_min, dtype=np.float32)
    
    # Build rANS table
    counts = np.bincount(indices.flatten(), minlength=16)
    freq, cumfreq, sym_table = build_rans_table(counts)
    
    # Entropy encode
    n_streams = 256
    compressed, stream_lengths, max_stream_len = entropy_encode(
        indices.flatten(), freq, cumfreq, n_streams
    )
    
    # Calculate compression
    original_bytes = out_dim * in_dim * 0.5
    compressed_bytes = len(compressed)
    ratio = original_bytes / compressed_bytes
    print(f"  Compression ratio: {ratio:.2f}x over 4-bit")
    
    # Create input vector
    x = np.random.randn(in_dim).astype(np.float32)
    
    # Reference: dequantize and matmul
    dequant_weights = indices.astype(np.float32) * scale + w_min
    y_ref = dequant_weights @ x
    
    # Convert to MLX arrays
    mx_compressed = mx.array(compressed)
    mx_stream_lengths = mx.array(np.array(stream_lengths, dtype=np.uint32))
    mx_freq = mx.array(freq)
    mx_cumfreq = mx.array(cumfreq)
    mx_sym_table = mx.array(sym_table)
    mx_x = mx.array(x)
    mx_scales = mx.array(scales)
    mx_biases = mx.array(biases)
    
    # Call fused kernel
    n_symbols = out_dim * in_dim
    y = mx.entropy_coded_matmul(
        mx_compressed,
        mx_stream_lengths,
        mx_freq,
        mx_cumfreq,
        mx_sym_table,
        mx_x,
        mx_scales,
        mx_biases,
        n_streams,
        n_symbols,
        max_stream_len,
        out_dim,
        in_dim,
        in_dim  # group_size (per-row)
    )
    mx.eval(y)
    
    # Compare
    y_np = np.array(y)
    max_diff = np.max(np.abs(y_np - y_ref))
    mean_diff = np.mean(np.abs(y_np - y_ref))
    
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    
    # Check if close enough
    if max_diff < 0.1:
        print("  ✓ Test PASSED!")
        return True
    else:
        print("  ✗ Test FAILED!")
        print(f"  Reference (first 5): {y_ref[:5]}")
        print(f"  Got (first 5): {y_np[:5]}")
        return False


def test_entropy_coded_linear():
    """Test EntropyCodedLinear layer."""
    print("\nTesting EntropyCodedLinear layer...")
    
    # Create a simple Linear layer
    in_dim, out_dim = 128, 64
    linear = nn.Linear(in_dim, out_dim, bias=True)
    
    # Random input
    x = mx.random.normal((1, in_dim))
    
    # Get reference output
    y_ref = linear(x)
    mx.eval(y_ref)
    
    # Convert to entropy-coded
    from mlx.nn.layers.entropy_coded import EntropyCodedLinear
    ec_linear = EntropyCodedLinear.from_linear(
        linear, 
        n_streams=256, 
        decode_mode="cached",  # Use cached mode for comparison
        group_size=in_dim
    )
    
    # Get output from entropy-coded layer
    y_ec = ec_linear(x)
    mx.eval(y_ec)
    
    # Compare
    diff = np.abs(np.array(y_ref) - np.array(y_ec))
    max_diff = np.max(diff)
    
    print(f"  Max diff vs reference: {max_diff:.6f}")
    print(f"  Compression ratio: {ec_linear.compression_ratio:.2f}x")
    
    if max_diff < 0.5:
        print("  ✓ Test PASSED!")
        return True
    else:
        print("  ✗ Test FAILED!")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Entropy-Coded Quantization Integration Tests")
    print("=" * 60)
    
    test1 = test_entropy_coded_matmul()
    test2 = test_entropy_coded_linear()
    
    print("\n" + "=" * 60)
    if test1 and test2:
        print("All tests passed!")
    else:
        print("Some tests failed!")
