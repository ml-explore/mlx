#!/usr/bin/env python3
"""
Benchmark entropy-coded quantization vs standard 4-bit quantization.

Compares:
1. Memory usage (compression ratio)
2. Inference throughput (tokens/sec)
3. Quality (perplexity or output comparison)
"""

import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from pathlib import Path

# rANS Constants
PROB_BITS = 14
PROB_SCALE = 1 << PROB_BITS
RANS_BYTE_L = 1 << 23


def build_rans_table(counts: np.ndarray, n_symbols: int = 16):
    """Build rANS frequency table."""
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
        sym_table[cumfreq[s]:cumfreq[s + 1]] = s
    
    return freq, cumfreq[:n_symbols], sym_table


def entropy_encode(indices: np.ndarray, freq: np.ndarray, cumfreq: np.ndarray, 
                   n_streams: int = 256):
    """Encode using interleaved rANS."""
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
            (state >> 0) & 0xFF, (state >> 8) & 0xFF,
            (state >> 16) & 0xFF, (state >> 24) & 0xFF
        ])
        
        stream_bytes_list.append(bytes(reversed(out_bytes)))
        stream_lengths.append(len(stream_bytes_list[-1]))
    
    max_stream_len = max(stream_lengths) if stream_lengths else 0
    stream_matrix = np.zeros((n_streams, max_stream_len), dtype=np.uint8)
    for i, data in enumerate(stream_bytes_list):
        if len(data) > 0:
            stream_matrix[i, :len(data)] = np.frombuffer(data, dtype=np.uint8)
    
    return stream_matrix.T.flatten(), stream_lengths, max_stream_len


class BenchmarkLayer:
    """Wrapper for benchmarking a single layer."""
    
    def __init__(self, weight: np.ndarray, name: str = "layer"):
        self.name = name
        self.weight = weight
        self.out_dim, self.in_dim = weight.shape
        
        # Quantize to 4-bit
        w_min, w_max = weight.min(), weight.max()
        self.scale = (w_max - w_min) / 15 if w_max != w_min else 1e-8
        self.zero_point = w_min
        self.indices = np.clip(
            np.round((weight - w_min) / self.scale), 0, 15
        ).astype(np.uint8)
        
        # Calculate entropy
        counts = np.bincount(self.indices.flatten(), minlength=16)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        self.entropy = -np.sum(probs * np.log2(probs))
        
        # Entropy encode
        self.freq, self.cumfreq, self.sym_table = build_rans_table(counts)
        self.compressed, self.stream_lengths, self.max_stream_len = entropy_encode(
            self.indices, self.freq, self.cumfreq, n_streams=256
        )
        
        # Memory stats
        self.original_bytes = self.out_dim * self.in_dim * 4  # FP32
        self.quant_4bit_bytes = self.out_dim * self.in_dim * 0.5
        self.entropy_bytes = len(self.compressed)
        
    def get_stats(self):
        return {
            'name': self.name,
            'shape': (self.out_dim, self.in_dim),
            'params': self.out_dim * self.in_dim,
            'entropy': self.entropy,
            'original_mb': self.original_bytes / 1024**2,
            'quant_4bit_mb': self.quant_4bit_bytes / 1024**2,
            'entropy_mb': self.entropy_bytes / 1024**2,
            'compression_vs_4bit': self.quant_4bit_bytes / self.entropy_bytes,
        }


def benchmark_matmul(layer: BenchmarkLayer, n_iters: int = 20, batch_size: int = 1):
    """Benchmark standard vs entropy-coded matmul."""
    
    # Prepare inputs
    x = mx.random.normal((batch_size, layer.in_dim))
    
    # Standard 4-bit path: dequantize + matmul
    indices_mx = mx.array(layer.indices.astype(np.float32))
    scale_mx = mx.array(layer.scale)
    zp_mx = mx.array(layer.zero_point)
    
    # Warmup
    weights = indices_mx * scale_mx + zp_mx
    y = x @ weights.T
    mx.eval(y)
    
    # Benchmark standard path
    times_standard = []
    for _ in range(n_iters):
        mx.synchronize()
        start = time.perf_counter()
        weights = indices_mx * scale_mx + zp_mx
        y = x @ weights.T
        mx.eval(y)
        mx.synchronize()
        times_standard.append(time.perf_counter() - start)
    
    # Entropy-coded path (fused kernel)
    mx_compressed = mx.array(layer.compressed)
    mx_stream_lengths = mx.array(np.array(layer.stream_lengths, dtype=np.uint32))
    mx_freq = mx.array(layer.freq)
    mx_cumfreq = mx.array(layer.cumfreq)
    mx_sym_table = mx.array(layer.sym_table)
    mx_scales = mx.array(np.full(layer.out_dim, layer.scale, dtype=np.float32))
    mx_biases = mx.array(np.full(layer.out_dim, layer.zero_point, dtype=np.float32))
    
    n_symbols = layer.out_dim * layer.in_dim
    
    # Warmup entropy path
    y_ec = mx.entropy_coded_matmul(
        mx_compressed, mx_stream_lengths, mx_freq, mx_cumfreq, mx_sym_table,
        x[0], mx_scales, mx_biases,
        256, n_symbols, layer.max_stream_len, layer.out_dim, layer.in_dim, layer.in_dim
    )
    mx.eval(y_ec)
    
    # Benchmark entropy path
    times_entropy = []
    for _ in range(n_iters):
        mx.synchronize()
        start = time.perf_counter()
        y_ec = mx.entropy_coded_matmul(
            mx_compressed, mx_stream_lengths, mx_freq, mx_cumfreq, mx_sym_table,
            x[0], mx_scales, mx_biases,
            256, n_symbols, layer.max_stream_len, layer.out_dim, layer.in_dim, layer.in_dim
        )
        mx.eval(y_ec)
        mx.synchronize()
        times_entropy.append(time.perf_counter() - start)
    
    return {
        'standard_ms': np.mean(times_standard) * 1000,
        'standard_std': np.std(times_standard) * 1000,
        'entropy_ms': np.mean(times_entropy) * 1000,
        'entropy_std': np.std(times_entropy) * 1000,
    }


def load_model_weights(model_path: Path):
    """Load weights from safetensors."""
    try:
        from safetensors import safe_open
    except ImportError:
        print("Installing safetensors...")
        import subprocess
        subprocess.run(["pip", "install", "safetensors"], check=True)
        from safetensors import safe_open
    
    weights = {}
    with safe_open(model_path, framework='np') as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def benchmark_real_model(model_path: Path, max_layers: int = 3):
    """Benchmark on real model weights."""
    print(f"\nLoading model from {model_path}...")
    weights = load_model_weights(model_path)
    
    print(f"Found {len(weights)} tensors")
    
    results = []
    total_original = 0
    total_4bit = 0
    total_entropy = 0
    
    # Find linear layers (2D tensors with reasonable size)
    linear_weights = []
    for name, tensor in weights.items():
        if tensor.ndim == 2 and tensor.shape[0] >= 64 and tensor.shape[1] >= 64:
            linear_weights.append((name, tensor))
    
    print(f"Found {len(linear_weights)} linear layers")
    
    for i, (name, tensor) in enumerate(linear_weights[:max_layers]):
        print(f"\n{'='*60}")
        print(f"Layer {i+1}/{min(len(linear_weights), max_layers)}: {name}")
        print(f"  Shape: {tensor.shape} ({tensor.size:,} params)")
        
        # Create benchmark layer
        layer = BenchmarkLayer(tensor.astype(np.float32), name)
        stats = layer.get_stats()
        
        print(f"  Entropy: {stats['entropy']:.2f} bits")
        print(f"  Size: FP32={stats['original_mb']:.2f}MB, 4-bit={stats['quant_4bit_mb']:.2f}MB, ECQ={stats['entropy_mb']:.2f}MB")
        print(f"  Compression vs 4-bit: {stats['compression_vs_4bit']:.2f}x")
        
        # Benchmark
        print(f"  Benchmarking (20 iterations)...")
        timing = benchmark_matmul(layer, n_iters=20)
        
        print(f"  Standard 4-bit: {timing['standard_ms']:.3f} ± {timing['standard_std']:.3f} ms")
        print(f"  Entropy-coded:  {timing['entropy_ms']:.3f} ± {timing['entropy_std']:.3f} ms")
        
        speedup = timing['standard_ms'] / timing['entropy_ms'] if timing['entropy_ms'] > 0 else 0
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({**stats, **timing, 'speedup': speedup})
        
        total_original += stats['original_mb']
        total_4bit += stats['quant_4bit_mb']
        total_entropy += stats['entropy_mb']
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Layers analyzed: {len(results)}")
    print(f"Total size: FP32={total_original:.2f}MB, 4-bit={total_4bit:.2f}MB, ECQ={total_entropy:.2f}MB")
    print(f"Overall compression vs 4-bit: {total_4bit/total_entropy:.2f}x")
    
    avg_entropy = np.mean([r['entropy'] for r in results])
    avg_speedup = np.mean([r['speedup'] for r in results])
    print(f"Average entropy: {avg_entropy:.2f} bits")
    print(f"Average speedup: {avg_speedup:.2f}x")
    
    return results


def benchmark_synthetic(sizes=None):
    """Benchmark on synthetic weights with LLM-like distribution."""
    if sizes is None:
        sizes = [
            (128, 128),
            (256, 256),
            (512, 512),
        ]
    
    print("\n" + "="*60)
    print("SYNTHETIC BENCHMARK (Gaussian weights)")
    print("="*60)
    
    results = []
    for out_dim, in_dim in sizes:
        print(f"\nSize: {out_dim}x{in_dim} ({out_dim*in_dim:,} params)")
        
        # Create Gaussian weights (simulates LLM weight distribution)
        np.random.seed(42)
        weight = np.random.randn(out_dim, in_dim).astype(np.float32) * 0.02
        
        layer = BenchmarkLayer(weight, f"{out_dim}x{in_dim}")
        stats = layer.get_stats()
        
        print(f"  Entropy: {stats['entropy']:.2f} bits")
        print(f"  Compression vs 4-bit: {stats['compression_vs_4bit']:.2f}x")
        
        timing = benchmark_matmul(layer, n_iters=20)
        
        print(f"  Standard: {timing['standard_ms']:.3f} ms")
        print(f"  Entropy:  {timing['entropy_ms']:.3f} ms")
        
        speedup = timing['standard_ms'] / timing['entropy_ms'] if timing['entropy_ms'] > 0 else 0
        print(f"  Speedup: {speedup:.2f}x")
        
        results.append({**stats, **timing, 'speedup': speedup})
    
    return results


def main():
    print("="*60)
    print(" MLX Entropy-Coded Quantization Benchmark")
    print("="*60)
    print(f"MLX version: {mx.__version__}")
    print(f"Device: {mx.default_device()}")
    
    # Check for real models
    model_paths = [
        Path("/Users/dhikshithreddy/Developer/PQQ/models/Qwen2.5-0.5B/model.safetensors"),
        Path("/Users/dhikshithreddy/Developer/PQQ/models/SmolLM-135M"),
    ]
    
    # Run synthetic benchmark first
    benchmark_synthetic()
    
    # Run real model benchmark if available
    for path in model_paths:
        if path.exists():
            if path.is_dir():
                safetensor_files = list(path.glob("*.safetensors"))
                if safetensor_files:
                    path = safetensor_files[0]
                else:
                    continue
            
            print(f"\n{'='*60}")
            print(f"REAL MODEL BENCHMARK: {path.parent.name}")
            print(f"{'='*60}")
            benchmark_real_model(path, max_layers=10)
            break
    
    print("\n" + "="*60)
    print("Benchmark complete!")
    print("="*60)


if __name__ == "__main__":
    main()
