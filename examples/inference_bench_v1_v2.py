#!/usr/bin/env python3
"""
Benchmark: Compare V1 vs V2 fused kernels for full model inference.

This benchmark compares:
- CACHED mode: Decode at load time, fast standard matmul
- FUSED (V1): Decode in kernel, O(rows × n) decode work
- FUSED_V2: Per-row decode in kernel, O(n) decode work

Expected: V2 should be much faster than V1 for FUSED mode, approaching CACHED performance.
"""

import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# Import the entropy-coded layer
from mlx.nn.layers.entropy_coded import EntropyCodedLinear, DecodeMode


def create_simple_model(hidden_sizes, decode_mode):
    """Create a simple MLP with entropy-coded layers."""
    layers = []
    for i in range(len(hidden_sizes) - 1):
        in_dim = hidden_sizes[i]
        out_dim = hidden_sizes[i + 1]
        
        # Create a standard linear layer first
        linear = nn.Linear(in_dim, out_dim)
        
        # Convert to entropy-coded
        ec_layer = EntropyCodedLinear.from_linear(
            linear,
            n_streams=64,
            decode_mode=decode_mode,
            group_size=in_dim  # Per-tensor for best compression
        )
        layers.append(ec_layer)
    
    return layers


def run_inference(layers, x, n_iters=100, warmup=10):
    """Run inference through layers and measure time."""
    # Warmup
    for _ in range(warmup):
        y = x
        for layer in layers:
            y = layer(y)
        mx.eval(y)
    
    mx.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        y = x
        for layer in layers:
            y = layer(y)
        mx.eval(y)
    mx.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / n_iters * 1000  # ms per inference


def main():
    print("=" * 70)
    print("Inference Benchmark: V1 (flat) vs V2 (per-row) Fused Kernels")
    print("=" * 70)
    
    # Test different model sizes
    # Skip V1 for larger models (too slow and crashes Metal)
    model_configs = [
        ("Small (256→512→256)", [256, 512, 256], True),
        ("Medium (512→1024→512)", [512, 1024, 512], True),
        ("Large (1024→2048→1024)", [1024, 2048, 1024], True),
        ("XL (2048→4096→2048)", [2048, 4096, 2048], False),  # Skip V1
    ]
    
    n_iters = 50
    
    print(f"\nRunning {n_iters} iterations per benchmark")
    print("-" * 70)
    
    results = []
    
    for name, hidden_sizes, run_v1 in model_configs:
        print(f"\n{name}")
        
        # Create models with different decode modes
        np.random.seed(42)
        mx.random.seed(42)
        
        # CACHED mode (baseline)
        print("  Creating CACHED model...", end=" ", flush=True)
        layers_cached = create_simple_model(hidden_sizes, "cached")
        print("done")
        
        # FUSED V1 mode (skip for very large models)
        layers_v1 = None
        if run_v1:
            print("  Creating FUSED (V1) model...", end=" ", flush=True)
            np.random.seed(42)
            mx.random.seed(42)
            layers_v1 = create_simple_model(hidden_sizes, "fused")
            print("done")
        else:
            print("  Skipping FUSED (V1) - too slow for this size")
        
        # FUSED V2 mode
        print("  Creating FUSED_V2 model...", end=" ", flush=True)
        np.random.seed(42)
        mx.random.seed(42)
        layers_v2 = create_simple_model(hidden_sizes, "fused_v2")
        print("done")
        
        # Input
        x = mx.random.normal((hidden_sizes[0],))
        
        # Run benchmarks
        print("  Benchmarking...", flush=True)
        
        time_cached = run_inference(layers_cached, x, n_iters)
        time_v1 = run_inference(layers_v1, x, n_iters) if layers_v1 else None
        time_v2 = run_inference(layers_v2, x, n_iters)
        
        # Print results
        print(f"    CACHED:     {time_cached:.3f} ms")
        if time_v1 is not None:
            print(f"    FUSED (V1): {time_v1:.3f} ms")
        print(f"    FUSED_V2:   {time_v2:.3f} ms")
        if time_v1 is not None:
            print(f"    V1→V2 speedup: {time_v1/time_v2:.1f}x")
        print(f"    V2 vs CACHED:  {time_v2/time_cached:.1f}x overhead")
        
        results.append({
            "name": name,
            "cached_ms": time_cached,
            "v1_ms": time_v1,
            "v2_ms": time_v2,
            "v1_v2_speedup": time_v1 / time_v2 if time_v1 else None,
            "v2_cached_ratio": time_v2 / time_cached
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'CACHED':>10} {'V1':>12} {'V2':>10} {'V1→V2':>10} {'V2/CACHED':>10}")
    print("-" * 80)
    for r in results:
        v1_str = f"{r['v1_ms']:.2f}ms" if r['v1_ms'] else "N/A"
        speedup_str = f"{r['v1_v2_speedup']:.1f}x" if r['v1_v2_speedup'] else "N/A"
        print(f"{r['name']:<30} {r['cached_ms']:>9.2f}ms {v1_str:>12} {r['v2_ms']:>9.2f}ms {speedup_str:>10} {r['v2_cached_ratio']:>9.1f}x")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    v1_results = [r for r in results if r['v1_v2_speedup'] is not None]
    avg_speedup = sum(r['v1_v2_speedup'] for r in v1_results) / len(v1_results) if v1_results else 0
    avg_overhead = sum(r['v2_cached_ratio'] for r in results) / len(results)
    print(f"Average V1→V2 speedup: {avg_speedup:.1f}x")
    print(f"Average V2 overhead vs CACHED: {avg_overhead:.1f}x")
    print("\nV2 (per-row encoding) dramatically reduces fused kernel decode work")
    print("from O(rows × n) to O(n), making fused inference much more practical.")


if __name__ == "__main__":
    main()
