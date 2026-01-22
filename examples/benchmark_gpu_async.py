#!/usr/bin/env python3
"""
Benchmark: GPU_ASYNC Mode - Decode while computing.

This benchmark tests the GPU_ASYNC strategy that decodes layer N+1
while computing layer N, hiding decode latency.

Timeline:
  GPU Stream 1: [Compute L0] [Compute L1] [Compute L2] ...
  GPU Stream 2: [Decode L1]  [Decode L2]  [Decode L3]  ...
                     ↓           ↓           ↓
                Ready before GPU needs it!
"""

import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from mlx.nn.layers.entropy_coded import (
    EntropyCodedLinear, 
    DecodeMode, 
    get_prefetcher
)


class SimpleMLP(nn.Module):
    """Simple MLP for testing."""
    
    def __init__(self, hidden_sizes, decode_mode):
        super().__init__()
        layers = []
        for i in range(len(hidden_sizes) - 1):
            # Create standard linear layer
            linear = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            # Convert to entropy-coded
            ec_layer = EntropyCodedLinear.from_linear(
                linear,
                n_streams=64,
                decode_mode=decode_mode,
                group_size=hidden_sizes[i]  # Per-tensor
            )
            layers.append(ec_layer)
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def forward_with_prefetch(self, x):
        """Forward with async prefetching."""
        for i, layer in enumerate(self.layers):
            # Start prefetch for next layer while computing current
            if i + 1 < len(self.layers):
                self.layers[i + 1].prefetch_weights()
            x = layer(x)
        return x


def run_benchmark(model, x, n_iters=50, warmup=10, use_prefetch=False):
    """Run inference benchmark."""
    # Warmup
    for _ in range(warmup):
        if use_prefetch:
            y = model.forward_with_prefetch(x)
        else:
            y = model(x)
        mx.eval(y)
    
    mx.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iters):
        if use_prefetch:
            y = model.forward_with_prefetch(x)
        else:
            y = model(x)
        mx.eval(y)
    mx.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / n_iters * 1000  # ms per inference


def main():
    print("=" * 70)
    print("GPU_ASYNC Benchmark: Decode-while-computing strategy")
    print("=" * 70)
    
    # Test configurations
    configs = [
        ("4-layer (512)", [512, 512, 512, 512, 512]),
        ("6-layer (256)", [256, 256, 256, 256, 256, 256, 256]),
    ]
    
    n_iters = 50
    
    print(f"\nRunning {n_iters} iterations per benchmark")
    print("-" * 70)
    
    for name, hidden_sizes in configs:
        print(f"\n{name}")
        
        # Set seeds
        np.random.seed(42)
        mx.random.seed(42)
        
        # Create models with different modes
        print("  Creating CACHED model...", end=" ", flush=True)
        model_cached = SimpleMLP(hidden_sizes, "cached")
        print("done")
        
        np.random.seed(42)
        mx.random.seed(42)
        print("  Creating FUSED_V2 model...", end=" ", flush=True)
        model_v2 = SimpleMLP(hidden_sizes, "fused_v2")
        print("done")
        
        np.random.seed(42)
        mx.random.seed(42)
        print("  Creating GPU_ASYNC model...", end=" ", flush=True)
        model_async = SimpleMLP(hidden_sizes, "gpu_async")
        print("done")
        
        # Input
        x = mx.random.normal((hidden_sizes[0],))
        
        # Run benchmarks
        print("  Benchmarking...", flush=True)
        
        time_cached = run_benchmark(model_cached, x, n_iters)
        time_v2 = run_benchmark(model_v2, x, n_iters)
        time_async_no_prefetch = run_benchmark(model_async, x, n_iters, use_prefetch=False)
        time_async_prefetch = run_benchmark(model_async, x, n_iters, use_prefetch=True)
        
        print(f"    CACHED (baseline):      {time_cached:.3f} ms")
        print(f"    FUSED_V2:               {time_v2:.3f} ms")
        print(f"    GPU_ASYNC (no prefetch):{time_async_no_prefetch:.3f} ms")
        print(f"    GPU_ASYNC (prefetch):   {time_async_prefetch:.3f} ms")
        
        # Analysis
        v2_overhead = time_v2 / time_cached
        async_overhead = time_async_prefetch / time_cached
        async_improvement = time_async_no_prefetch / time_async_prefetch
        
        print(f"\n    V2 overhead vs CACHED:     {v2_overhead:.2f}x")
        print(f"    ASYNC overhead vs CACHED:  {async_overhead:.2f}x")
        print(f"    Prefetch improvement:      {async_improvement:.2f}x")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
GPU_ASYNC mode hides decode latency by decoding the next layer
while the current layer is computing. This works best when:
1. Model has many layers (more overlap opportunity)
2. Compute time per layer > decode time
3. GPU has spare compute capacity

For small models or compute-bound layers, the overhead of managing
async streams may outweigh benefits. Use FUSED_V2 in those cases.
    """)


if __name__ == "__main__":
    main()
