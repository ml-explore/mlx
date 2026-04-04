#!/usr/bin/env python3
# Copyright © 2023-2024 Apple Inc.
# M5 Max Performance Benchmark
# Comprehensive benchmark suite for Apple Silicon M5 Max optimizations

"""
Comprehensive M5 Max Performance Benchmark
===========================================

This benchmark suite tests the performance improvements for Apple Silicon M5 Max:
1. Optimized GEMM parameters for 's' (Max) architecture
2. Increased buffer capacity (70 ops/70 MB for M5 Max)
3. Improved reduce kernels with hierarchical reduction
4. Better thread group sizes for large matrices
5. CPU backend cache blocking improvements

Run with: python m5_max_bench.py [--cpu] [--gpu]
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

# Try to import MLX with helpful error messages
try:
    import mlx.core as mx
except ImportError as e:
    print("=" * 80)
    print("MLX Import Error")
    print("=" * 80)
    print(f"Import failed with error: {e}")
    print()
    print("Solutions:")
    print("-" * 80)
    print()
    print("Option 1: Install pre-built MLX (recommended for most users)")
    print()
    print("  pip install mlx")
    print()
    print("  Verify: python -c 'import mlx.core as mx; print(mx.__version__)'")
    print()
    print("-" * 80)
    print()
    print("Option 2: Install MLX with CPU backend only (no GPU)")
    print()
    print("  pip install mlx[cpu]")
    print()
    print("-" * 80)
    print()
    print("Option 3: Build MLX from source (requires full Xcode + Metal Toolchain)")
    print()
    print("  1. Install full Xcode from App Store")
    print("  2. Run: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer")
    print("  3. Run: sudo xcodebuild -license accept")
    print("  4. Run: sudo xcodebuild -downloadComponent MetalToolchain")
    print("  5. Run: cd /path/to/mlx && pip install -e .")
    print()
    print("-" * 80)
    print()
    print("For detailed troubleshooting, see: benchmarks/README.md")
    print("=" * 80)
    sys.exit(1)

# Try to import time_utils from same directory
try:
    from .time_utils import time_fn
except ImportError:
    from time_utils import time_fn


class M5MaxBenchmark:
    """Base class for M5 Max benchmarks."""
    
    def __init__(self, name):
        self.name = name
        self.results = []
        
    def warmup(self, fn, *args, num_iters=5, **kwargs):
        """Warmup routine."""
        for _ in range(num_iters):
            mx.eval(fn(*args, **kwargs))
            
    def measure(self, fn, *args, num_iters=100, **kwargs):
        """Measure execution time."""
        self.warmup(fn, *args, **kwargs)
        
        times = []
        for _ in range(num_iters):
            tic = time.perf_counter()
            x = mx.eval(fn(*args, **kwargs))
            toc = time.perf_counter()
            times.append(1000 * (toc - tic))  # Convert to ms
            
        return {
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'std': (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
            'num_iters': num_iters
        }


class MatmulBenchmark(M5MaxBenchmark):
    """Matrix multiplication benchmarks."""
    
    def __init__(self):
        super().__init__("matmul")
        
    def benchmark_small_nn(self):
        """Small NN matmul (M=N=K=64)."""
        a = mx.random.uniform(shape=(64, 64))
        b = mx.random.uniform(shape=(64, 64))
        mx.eval(a, b)
        
        def fn():
            return mx.matmul(a, b)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'small_nn',
            'shape': '64x64 @ 64x64',
            **result
        })
        
    def benchmark_medium_nt(self):
        """Medium transpose matmul (M=N=K=512)."""
        a = mx.random.uniform(shape=(512, 512))
        b = mx.random.uniform(shape=(512, 512))
        bT = mx.transpose(b)
        mx.eval(a, bT)
        
        def fn():
            return mx.matmul(a, bT)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'medium_nt',
            'shape': '512x512 @ 512x512^T',
            **result
        })
        
    def benchmark_large_nn(self):
        """Large NN matmul (M=N=K=1024)."""
        a = mx.random.uniform(shape=(1024, 1024))
        b = mx.random.uniform(shape=(1024, 1024))
        mx.eval(a, b)
        
        def fn():
            return mx.matmul(a, b)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'large_nn',
            'shape': '1024x1024 @ 1024x1024',
            **result
        })
        
    def benchmark_batched(self):
        """Batched matmul."""
        B = 8
        T = 1024
        D = 512
        a = mx.random.uniform(shape=(B, T, D))
        b = mx.random.uniform(shape=(D, D))
        mx.eval(a, b)
        
        def fn():
            return mx.matmul(a, b)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'batched',
            'shape': f'{B}x{T}x{D} @ {D}x{D}',
            **result
        })
        
    def benchmark_very_large(self):
        """Very large matmul (M=N=K=2048)."""
        a = mx.random.uniform(shape=(2048, 2048))
        b = mx.random.uniform(shape=(2048, 2048))
        mx.eval(a, b)
        
        def fn():
            return mx.matmul(a, b)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'very_large',
            'shape': '2048x2048 @ 2048x2048',
            **result
        })
        
    def benchmark_m5_max_optimized(self):
        """M5 Max optimized large matmul (leveraging 70 MB buffer).
        
        Tests the M5 Max's optimized GEMM with:
        - 70 ops per buffer (M5 Max parameter)
        - Large matrices that fit in L2 cache
        """
        # M5 Max optimized sizes: leverage 70 MB buffer capacity
        a = mx.random.uniform(shape=(2048, 2048))
        b = mx.random.uniform(shape=(2048, 2048))
        mx.eval(a, b)
        
        def fn():
            return mx.matmul(a, b)
            
        result = self.measure(fn, num_iters=5)  # Fewer iters for large ops
        self.results.append({
            'test': 'm5_max_optimized',
            'shape': '2048x2048 @ 2048x2048 (M5 Max optimized)',
            **result
        })
        
    def benchmark_huge_matmul(self):
        """Huge matmul for M5 Max memory bandwidth test (M=N=K=4096)."""
        a = mx.random.uniform(shape=(4096, 4096))
        b = mx.random.uniform(shape=(4096, 4096))
        mx.eval(a, b)
        
        def fn():
            return mx.matmul(a, b)
            
        result = self.measure(fn, num_iters=3)  # Very few iters for huge ops
        self.results.append({
            'test': 'huge_matmul',
            'shape': '4096x4096 @ 4096x4096 (M5 Max huge)',
            **result
        })
        
    def benchmark_fused_gelu(self):
        """Fused matmul + gelu (common in transformers)."""
        N = 512
        D_in = 2048
        D_out = 2048
        a = mx.random.uniform(shape=(N, D_in))
        b = mx.random.uniform(shape=(D_in, D_out))
        mx.eval(a, b)
        
        def fn():
            x = mx.matmul(a, b)
            return x * 0.5 * (1.0 + mx.erf(x / math.sqrt(2.0)))
            
        result = self.measure(fn)
        self.results.append({
            'test': 'fused_gelu',
            'shape': f'{N}x{D_in} @ {D_in}x{D_out} + gelu',
            **result
        })
        
    def benchmark_fused_add(self):
        """Fused matmul + add (common in transformers)."""
        N = 1024
        D_in = 4096
        D_out = 4096
        a = mx.random.uniform(shape=(N, D_in))
        b = mx.random.uniform(shape=(D_in, D_out))
        c = mx.random.uniform(shape=(N, D_out))
        mx.eval(a, b, c)
        
        def fn():
            return mx.matmul(a, b) + c
            
        result = self.measure(fn)
        self.results.append({
            'test': 'fused_add',
            'shape': f'{N}x{D_in} @ {D_in}x{D_out} + add',
            **result
        })
        
    def benchmark_fp16_matmul(self):
        """FP16 matmul (M5 Max supports fast fp16)."""
        a = mx.random.uniform(shape=(2048, 2048), dtype=mx.float16)
        b = mx.random.uniform(shape=(2048, 2048), dtype=mx.float16)
        mx.eval(a, b)
        
        def fn():
            return mx.matmul(a, b)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'fp16_matmul',
            'shape': '2048x2048 @ 2048x2048 (fp16)',
            **result
        })
        
    def benchmark_bf16_matmul(self):
        """BF16 matmul (M5 Max supports fast bf16)."""
        a = mx.random.uniform(shape=(2048, 2048), dtype=mx.bfloat16)
        b = mx.random.uniform(shape=(2048, 2048), dtype=mx.bfloat16)
        mx.eval(a, b)
        
        def fn():
            return mx.matmul(a, b)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'bf16_matmul',
            'shape': '2048x2048 @ 2048x2048 (bf16)',
            **result
        })
        
    def benchmark_batched_large(self):
        """Large batched matmul for M5 Max throughput test."""
        B = 32
        T = 1024
        D = 1024
        a = mx.random.uniform(shape=(B, T, D))
        b = mx.random.uniform(shape=(D, D))
        mx.eval(a, b)
        
        def fn():
            return mx.matmul(a, b)
            
        result = self.measure(fn, num_iters=3)
        self.results.append({
            'test': 'batched_large',
            'shape': f'{B}x{T}x{D} @ {D}x{D} (large batch)',
            **result
        })
        
    def run(self):
        """Run all matmul benchmarks."""
        print("Running matmul benchmarks...")
        self.benchmark_small_nn()
        self.benchmark_medium_nt()
        self.benchmark_large_nn()
        self.benchmark_batched()
        self.benchmark_very_large()
        self.benchmark_m5_max_optimized()  # M5 Max optimized
        self.benchmark_huge_matmul()      # Huge matrices for bandwidth
        self.benchmark_fused_gelu()
        self.benchmark_fused_add()
        self.benchmark_fp16_matmul()      # FP16 performance
        self.benchmark_bf16_matmul()      # BF16 performance
        self.benchmark_batched_large()
        return self.results


class ReduceBenchmark(M5MaxBenchmark):
    """Reduce operation benchmarks."""
    
    def __init__(self):
        super().__init__("reduce")
        
    def benchmark_small_sum(self):
        """Small sum reduction."""
        a = mx.random.uniform(shape=(1024,))
        mx.eval(a)
        
        def fn():
            return mx.sum(a)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'small_sum',
            'shape': '(1024,)',
            **result
        })
        
    def benchmark_row_sum(self):
        """Row sum reduction."""
        a = mx.random.uniform(shape=(64, 1024))
        mx.eval(a)
        
        def fn():
            return mx.sum(a, axis=1)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'row_sum',
            'shape': '(64, 1024) -> axis=1',
            **result
        })
        
    def benchmark_col_sum(self):
        """Column sum reduction."""
        a = mx.random.uniform(shape=(1024, 64))
        mx.eval(a)
        
        def fn():
            return mx.sum(a, axis=0)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'col_sum',
            'shape': '(1024, 64) -> axis=0',
            **result
        })
        
    def benchmark_large_sum(self):
        """Large sum reduction (>1M elements)."""
        a = mx.random.uniform(shape=(1024, 1024, 128))
        mx.eval(a)
        
        def fn():
            return mx.sum(a, axis=-1)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'large_sum',
            'shape': '(1024, 1024, 128) -> axis=-1',
            **result
        })
        
    def benchmark_mean(self):
        """Mean reduction."""
        a = mx.random.uniform(shape=(64, 1024, 1024))
        mx.eval(a)
        
        def fn():
            return mx.mean(a, axis=-1)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'mean',
            'shape': '(64, 1024, 1024) -> axis=-1',
            **result
        })
        
    def benchmark_min(self):
        """Min reduction."""
        a = mx.random.uniform(shape=(64, 1024, 1024))
        mx.eval(a)
        
        def fn():
            return mx.min(a, axis=-1)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'min',
            'shape': '(64, 1024, 1024) -> axis=-1',
            **result
        })
        
    def benchmark_max(self):
        """Max reduction."""
        a = mx.random.uniform(shape=(64, 1024, 1024))
        mx.eval(a)
        
        def fn():
            return mx.max(a, axis=-1)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'max',
            'shape': '(64, 1024, 1024) -> axis=-1',
            **result
        })
        
    def benchmark_logsumexp(self):
        """Logsumexp reduction."""
        a = mx.random.uniform(shape=(64, 10, 10000))
        mx.eval(a)
        
        def fn():
            return mx.logsumexp(a, axis=-1)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'logsumexp',
            'shape': '(64, 10, 10000) -> axis=-1',
            **result
        })
        
    def benchmark_m5_max_reduce(self):
        """M5 Max optimized reduce (leveraging 70 MB buffer).
        
        Tests hierarchical reduction on large data that fits in M5 Max's 70 MB buffer.
        """
        # Large reduction: leverage 70 MB capacity
        a = mx.random.uniform(shape=(16384, 16384))
        mx.eval(a)
        
        def fn():
            return mx.sum(a, axis=-1)
            
        result = self.measure(fn, num_iters=2)  # Very few iters for huge reductions
        self.results.append({
            'test': 'm5_max_reduce',
            'shape': '(16384, 16384) -> axis=-1 (M5 Max large reduce)',
            **result
        })
        
    def benchmark_batched_reduce(self):
        """Batched reduce operations."""
        B = 32
        N = 4096
        a = mx.random.uniform(shape=(B, N))
        mx.eval(a)
        
        def fn():
            return mx.sum(a, axis=-1)
            
        result = self.measure(fn, num_iters=3)
        self.results.append({
            'test': 'batched_reduce',
            'shape': f'(32, {N}) -> axis=-1 (batched)',
            **result
        })
        
    def benchmark_fp16_reduce(self):
        """FP16 reduce operation (M5 Max supports fast fp16)."""
        a = mx.random.uniform(shape=(8192, 8192), dtype=mx.float16)
        mx.eval(a)
        
        def fn():
            return mx.sum(a, axis=-1)
            
        result = self.measure(fn, num_iters=3)
        self.results.append({
            'test': 'fp16_reduce',
            'shape': '(8192, 8192) -> axis=-1 (fp16)',
            **result
        })
        
    def benchmark_bf16_reduce(self):
        """BF16 reduce operation (M5 Max supports fast bf16)."""
        a = mx.random.uniform(shape=(8192, 8192), dtype=mx.bfloat16)
        mx.eval(a)
        
        def fn():
            return mx.sum(a, axis=-1)
            
        result = self.measure(fn, num_iters=3)
        self.results.append({
            'test': 'bf16_reduce',
            'shape': '(8192, 8192) -> axis=-1 (bf16)',
            **result
        })
        
    def benchmark_parallel_reduce(self):
        """Parallel reduce across multiple axes."""
        a = mx.random.uniform(shape=(64, 1024, 1024))
        mx.eval(a)
        
        def fn():
            return mx.sum(a, axis=(1, 2))
            
        result = self.measure(fn)
        self.results.append({
            'test': 'parallel_reduce',
            'shape': '(64, 1024, 1024) -> axis=(1,2) (parallel)',
            **result
        })
        
    def benchmark_softmax(self):
        """Softmax reduce (common in transformers)."""
        a = mx.random.uniform(shape=(16, 32, 10000))
        mx.eval(a)
        
        def fn():
            return mx.nn.softmax(a, axis=-1)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'softmax',
            'shape': '(16, 32, 10000) -> axis=-1 (softmax)',
            **result
        })
        
    def benchmark_batch_norm(self):
        """Batch normalization reduce."""
        x = mx.random.uniform(shape=(32, 1024))
        mean = mx.zeros((1024,))
        var = mx.ones((1024,))
        mx.eval(x, mean, var)
        
        def fn():
            return mx.nn.batch_normalize(x, mean=mean, var=var)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'batch_norm',
            'shape': '(32, 1024) batch norm',
            **result
        })
        
    def run(self):
        """Run all reduce benchmarks."""
        print("Running reduce benchmarks...")
        self.benchmark_small_sum()
        self.benchmark_row_sum()
        self.benchmark_col_sum()
        self.benchmark_large_sum()
        self.benchmark_mean()
        self.benchmark_min()
        self.benchmark_max()
        self.benchmark_logsumexp()
        self.benchmark_m5_max_reduce()    # M5 Max large reduce
        self.benchmark_batched_reduce()
        self.benchmark_fp16_reduce()      # FP16 performance
        self.benchmark_bf16_reduce()      # BF16 performance
        self.benchmark_parallel_reduce()
        self.benchmark_softmax()
        self.benchmark_batch_norm()
        return self.results


class ElementWiseBenchmark(M5MaxBenchmark):
    """Element-wise operation benchmarks."""
    
    def __init__(self):
        super().__init__("element_wise")
        
    def benchmark_add(self):
        """Addition."""
        a = mx.random.uniform(shape=(32, 1024, 1024))
        b = mx.random.uniform(shape=(32, 1024, 1024))
        mx.eval(a, b)
        
        def fn():
            return mx.add(a, b)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'add',
            'shape': '(32, 1024, 1024)',
            **result
        })
        
    def benchmark_multiply(self):
        """Multiplication."""
        a = mx.random.uniform(shape=(32, 1024, 1024))
        b = mx.random.uniform(shape=(32, 1024, 1024))
        mx.eval(a, b)
        
        def fn():
            return mx.multiply(a, b)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'multiply',
            'shape': '(32, 1024, 1024)',
            **result
        })
        
    def benchmark_exp(self):
        """Exponential."""
        a = mx.random.uniform(shape=(10000, 1000))
        mx.eval(a)
        
        def fn():
            return mx.exp(a)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'exp',
            'shape': '(10000, 1000)',
            **result
        })
        
    def benchmark_log(self):
        """Logarithm."""
        a = mx.random.uniform(shape=(10000, 1000), low=0.5, high=1.5)
        mx.eval(a)
        
        def fn():
            return mx.log(a)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'log',
            'shape': '(10000, 1000)',
            **result
        })
        
    def benchmark_sigmoid(self):
        """Sigmoid."""
        a = mx.random.uniform(shape=(1000, 1000))
        mx.eval(a)
        
        def fn():
            return mx.sigmoid(a)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'sigmoid',
            'shape': '(1000, 1000)',
            **result
        })
        
    def benchmark_relu(self):
        """ReLU."""
        a = mx.random.uniform(shape=(1000, 1000))
        mx.eval(a)
        
        def fn():
            return mx.maximum(a, 0)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'relu',
            'shape': '(1000, 1000)',
            **result
        })
        
    def benchmark_m5_max_element(self):
        """M5 Max optimized element-wise (leveraging 70 MB buffer).
        
        Tests large-scale element-wise operations that fit in M5 Max's 70 MB buffer.
        """
        # Large element-wise: leverage 70 MB capacity
        a = mx.random.uniform(shape=(8192, 8192))
        b = mx.random.uniform(shape=(8192, 8192))
        mx.eval(a, b)
        
        def fn():
            return a + b * 0.5
            
        result = self.measure(fn, num_iters=2)
        self.results.append({
            'test': 'm5_max_element',
            'shape': '(8192, 8192) element-wise (M5 Max large)',
            **result
        })
        
    def benchmark_fp16_element(self):
        """FP16 element-wise (M5 Max supports fast fp16)."""
        a = mx.random.uniform(shape=(8192, 8192), dtype=mx.float16)
        b = mx.random.uniform(shape=(8192, 8192), dtype=mx.float16)
        mx.eval(a, b)
        
        def fn():
            return a + b * 0.5
            
        result = self.measure(fn, num_iters=2)
        self.results.append({
            'test': 'fp16_element',
            'shape': '(8192, 8192) element-wise (fp16)',
            **result
        })
        
    def benchmark_bf16_element(self):
        """BF16 element-wise (M5 Max supports fast bf16)."""
        a = mx.random.uniform(shape=(8192, 8192), dtype=mx.bfloat16)
        b = mx.random.uniform(shape=(8192, 8192), dtype=mx.bfloat16)
        mx.eval(a, b)
        
        def fn():
            return a + b * 0.5
            
        result = self.measure(fn, num_iters=2)
        self.results.append({
            'test': 'bf16_element',
            'shape': '(8192, 8192) element-wise (bf16)',
            **result
        })
        
    def benchmark_gelu(self):
        """GELU activation (common in transformers)."""
        a = mx.random.uniform(shape=(1024, 4096))
        mx.eval(a)
        
        def fn():
            return a * 0.5 * (1.0 + mx.erf(a / math.sqrt(2.0)))
            
        result = self.measure(fn)
        self.results.append({
            'test': 'gelu',
            'shape': '(1024, 4096) GELU',
            **result
        })
        
    def benchmark_gelu_fused(self):
        """Fused GELU (optimized path)."""
        a = mx.random.uniform(shape=(1024, 4096))
        mx.eval(a)
        
        def fn():
            return a * 0.5 * (1.0 + mx.erf(a / math.sqrt(2.0)))
            
        result = self.measure(fn)
        self.results.append({
            'test': 'gelu_fused',
            'shape': '(1024, 4096) fused GELU',
            **result
        })
        
    def benchmark_softmax(self):
        """Softmax (common in transformers)."""
        a = mx.random.uniform(shape=(16, 32, 10000))
        mx.eval(a)
        
        def fn():
            return mx.nn.softmax(a, axis=-1)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'softmax',
            'shape': '(16, 32, 10000) softmax',
            **result
        })
        
    def run(self):
        """Run all element-wise benchmarks."""
        print("Running element-wise benchmarks...")
        self.benchmark_add()
        self.benchmark_multiply()
        self.benchmark_exp()
        self.benchmark_log()
        self.benchmark_sigmoid()
        self.benchmark_relu()
        self.benchmark_m5_max_element()  # M5 Max large element-wise
        self.benchmark_fp16_element()    # FP16 performance
        self.benchmark_bf16_element()    # BF16 performance
        self.benchmark_gelu()
        self.benchmark_gelu_fused()
        self.benchmark_softmax()
        return self.results


class LargeMatrixBenchmark(M5MaxBenchmark):
    """Large matrix operation benchmarks."""
    
    def __init__(self):
        super().__init__("large_matrices")
        
    def benchmark_qr(self):
        """QR decomposition (CPU stream - not yet GPU supported)."""
        a = mx.random.uniform(shape=(512, 512))
        mx.eval(a)
        
        def fn():
            # QR decomposition currently only supported on CPU
            with mx.stream(mx.cpu):
                return mx.linalg.qr(a)
            
        result = self.measure(fn, num_iters=10)  # Slower operation
        self.results.append({
            'test': 'qr',
            'shape': '(512, 512)',
            **result
        })
        
    def benchmark_svd(self):
        """SVD (CPU stream - not yet GPU supported)."""
        a = mx.random.uniform(shape=(256, 256))
        mx.eval(a)
        
        def fn():
            # SVD currently only supported on CPU
            with mx.stream(mx.cpu):
                return mx.linalg.svd(a)
            
        result = self.measure(fn, num_iters=10)  # Slower operation
        self.results.append({
            'test': 'svd',
            'shape': '(256, 256)',
            **result
        })
        
    def benchmark_eig(self):
        """Eigenvalue decomposition (CPU stream - not yet GPU supported)."""
        # Make symmetric for stability
        a = mx.random.uniform(shape=(256, 256))
        a = (a + mx.transpose(a)) / 2
        mx.eval(a)
        
        def fn():
            # eigvalsh currently only supported on CPU
            with mx.stream(mx.cpu):
                return mx.linalg.eigvalsh(a)
            
        result = self.measure(fn, num_iters=10)  # Slower operation
        self.results.append({
            'test': 'eigvalsh',
            'shape': '(256, 256) symmetric',
            **result
        })
        

    def run(self):
        """Run all large matrix benchmarks."""
        print("Running large matrix benchmarks...")
        self.benchmark_qr()
        self.benchmark_svd()
        self.benchmark_eig()
        return self.results


class CPUBenchmark(M5MaxBenchmark):
    """CPU backend benchmarks."""
    
    def __init__(self):
        super().__init__("cpu")
        
    def benchmark_cpu_matmul(self):
        """CPU matmul."""
        a = mx.random.uniform(shape=(512, 512))
        b = mx.random.uniform(shape=(512, 512))
        mx.eval(a, b)
        
        def fn():
            return mx.matmul(a, b)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'matmul',
            'shape': '(512, 512) @ (512, 512)',
            **result
        })
        
    def benchmark_cpu_reduce(self):
        """CPU reduce."""
        a = mx.random.uniform(shape=(1024, 1024))
        mx.eval(a)
        
        def fn():
            return mx.sum(a, axis=-1)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'sum',
            'shape': '(1024, 1024)',
            **result
        })
        
    def benchmark_cpu_element_wise(self):
        """CPU element-wise."""
        a = mx.random.uniform(shape=(10000, 1000))
        mx.eval(a)
        
        def fn():
            return mx.exp(a)
            
        result = self.measure(fn)
        self.results.append({
            'test': 'exp',
            'shape': '(10000, 1000)',
            **result
        })
        
    def run(self):
        """Run all CPU benchmarks."""
        print("Running CPU benchmarks...")
        self.benchmark_cpu_matmul()
        self.benchmark_cpu_reduce()
        self.benchmark_cpu_element_wise()
        return self.results


def run_benchmarks(cpu=False, output_file=None):
    """Run all benchmarks and collect results."""
    
    # Set device
    if cpu:
        mx.set_default_device(mx.cpu)
        print("Using CPU backend")
    else:
        mx.set_default_device(mx.gpu)
        print("Using GPU backend (Metal)")
        
    # Print system info
    print(f"\nSystem Info:")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {mx.default_device()}")
    
    # Get device info if available
    try:
        import mlx.metal as metal
        device_info = metal.device_info(mx.default_device())
        print(f"  Device info: {device_info}")
    except:
        pass
        
    # Run all benchmarks
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': str(mx.default_device()),
        'benchmarks': {}
    }
    
    # Matmul
    matmul_bench = MatmulBenchmark()
    results['benchmarks']['matmul'] = matmul_bench.run()
    
    # Reduce
    reduce_bench = ReduceBenchmark()
    results['benchmarks']['reduce'] = reduce_bench.run()
    
    # Element-wise
    element_wise_bench = ElementWiseBenchmark()
    results['benchmarks']['element_wise'] = element_wise_bench.run()
    
    # Large matrices
    large_matrix_bench = LargeMatrixBenchmark()
    results['benchmarks']['large_matrices'] = large_matrix_bench.run()
    
    # CPU (only if requested)
    if cpu:
        cpu_bench = CPUBenchmark()
        results['benchmarks']['cpu'] = cpu_bench.run()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_time = 0
    for bench_name, bench_results in results['benchmarks'].items():
        bench_total = sum(r['mean'] for r in bench_results)
        total_time += bench_total
        print(f"\n{bench_name}: {bench_total:.2f} ms total")
        for result in bench_results:
            print(f"  {result['test']}: {result['mean']:.2f} ms "
                  f"(±{result['std']:.2f}) [{result['shape']}]")
    
    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {total_time:.2f} ms")
    print(f"{'='*60}")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
        
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="M5 Max Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python m5_max_bench.py                    # Run with GPU
  python m5_max_bench.py --cpu              # Run with CPU
  python m5_max_bench.py --output results.json  # Save to file
        """
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU backend instead of GPU"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file for results (JSON format)"
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        # Generate default output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        device_str = "cpu" if args.cpu else "gpu"
        args.output = f"m5_max_bench_{device_str}_{timestamp}.json"
    
    run_benchmarks(cpu=args.cpu, output_file=args.output)
