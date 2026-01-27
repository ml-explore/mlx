#!/usr/bin/env python3
"""Test script for CUDA radix partition implementation."""

import mlx.core as mx
import numpy as np

def test_partition_basic():
    """Test basic partition functionality."""
    print("Testing basic partition...")
    
    # Test with different sizes
    for size in [100, 500, 1000, 5000, 10000]:
        for k in [0, size // 4, size // 2, size - 1]:
            x = mx.random.uniform(shape=(size,))
            mx.eval(x)
            
            # Test partition
            result = mx.partition(x, k)
            mx.eval(result)
            
            # Verify: element at k should be the k-th smallest
            x_np = np.array(x)
            result_np = np.array(result)
            
            expected_kth = np.partition(x_np, k)[k]
            actual_kth = result_np[k]
            
            # All elements before k should be <= kth element
            assert np.all(result_np[:k] <= actual_kth), f"Failed for size={size}, k={k}"
            # All elements after k should be >= kth element
            assert np.all(result_np[k+1:] >= actual_kth), f"Failed for size={size}, k={k}"
            
    print("  Basic partition: PASSED")

def test_argpartition_basic():
    """Test basic argpartition functionality."""
    print("Testing basic argpartition...")
    
    for size in [100, 500, 1000, 5000]:
        for k in [0, size // 4, size // 2, size - 1]:
            x = mx.random.uniform(shape=(size,))
            mx.eval(x)
            
            # Test argpartition
            indices = mx.argpartition(x, k)
            mx.eval(indices)
            
            # Verify using the indices
            x_np = np.array(x)
            indices_np = np.array(indices)
            
            result_np = x_np[indices_np]
            kth_val = result_np[k]
            
            assert np.all(result_np[:k] <= kth_val), f"Failed for size={size}, k={k}"
            assert np.all(result_np[k+1:] >= kth_val), f"Failed for size={size}, k={k}"
            
    print("  Basic argpartition: PASSED")

def test_partition_2d():
    """Test partition on 2D arrays."""
    print("Testing 2D partition...")
    
    for shape in [(10, 100), (50, 200), (100, 500)]:
        for axis in [0, 1, -1]:
            k = shape[axis if axis >= 0 else len(shape) + axis] // 2
            
            x = mx.random.uniform(shape=shape)
            mx.eval(x)
            
            result = mx.partition(x, k, axis=axis)
            mx.eval(result)
            
            # Verify
            x_np = np.array(x)
            result_np = np.array(result)
            expected = np.partition(x_np, k, axis=axis)
            
            # Check that the k-th element along the axis is correct
            if axis == 0 or axis == -2:
                for j in range(shape[1]):
                    assert np.all(result_np[:k, j] <= result_np[k, j])
                    assert np.all(result_np[k+1:, j] >= result_np[k, j])
            else:
                for i in range(shape[0]):
                    assert np.all(result_np[i, :k] <= result_np[i, k])
                    assert np.all(result_np[i, k+1:] >= result_np[i, k])
                    
    print("  2D partition: PASSED")

def test_partition_dtypes():
    """Test partition with different data types."""
    print("Testing different dtypes...")
    
    dtypes = [mx.float32, mx.float16, mx.int32, mx.int64, mx.uint32]
    
    for dtype in dtypes:
        x = mx.random.uniform(shape=(1000,))
        if dtype in [mx.int32, mx.int64, mx.uint32]:
            x = (x * 1000).astype(dtype)
        else:
            x = x.astype(dtype)
        mx.eval(x)
        
        k = 500
        result = mx.partition(x, k)
        mx.eval(result)
        
        result_np = np.array(result.astype(mx.float32))
        kth_val = result_np[k]
        
        assert np.all(result_np[:k] <= kth_val), f"Failed for dtype={dtype}"
        assert np.all(result_np[k+1:] >= kth_val), f"Failed for dtype={dtype}"
        
    print("  Different dtypes: PASSED")

def test_partition_non_contiguous():
    """Test partition on non-contiguous arrays."""
    print("Testing non-contiguous arrays...")
    
    # Transposed array
    x = mx.random.uniform(shape=(100, 200))
    mx.eval(x)
    x_t = mx.transpose(x)
    
    k = 50
    result = mx.partition(x_t, k, axis=1)
    mx.eval(result)
    
    result_np = np.array(result)
    for i in range(result_np.shape[0]):
        assert np.all(result_np[i, :k] <= result_np[i, k])
        assert np.all(result_np[i, k+1:] >= result_np[i, k])
    
    # Sliced array
    x = mx.random.uniform(shape=(200, 300))
    mx.eval(x)
    x_slice = x[::2, ::3]
    
    k = 25
    result = mx.partition(x_slice, k, axis=1)
    mx.eval(result)
    
    result_np = np.array(result)
    for i in range(result_np.shape[0]):
        assert np.all(result_np[i, :k] <= result_np[i, k])
        assert np.all(result_np[i, k+1:] >= result_np[i, k])
        
    print("  Non-contiguous arrays: PASSED")

def benchmark_partition():
    """Benchmark partition vs sort."""
    print("\nBenchmarking partition vs sort...")
    
    import time
    
    sizes = [10000, 100000, 1000000]
    
    for size in sizes:
        x = mx.random.uniform(shape=(size,))
        mx.eval(x)
        k = size // 2
        
        # Warm up
        _ = mx.partition(x, k)
        _ = mx.sort(x)
        mx.eval(_)
        
        # Benchmark partition
        start = time.time()
        for _ in range(10):
            result = mx.partition(x, k)
            mx.eval(result)
        partition_time = (time.time() - start) / 10
        
        # Benchmark sort
        start = time.time()
        for _ in range(10):
            result = mx.sort(x)
            mx.eval(result)
        sort_time = (time.time() - start) / 10
        
        speedup = sort_time / partition_time
        print(f"  Size {size:>10}: partition={partition_time*1000:.2f}ms, sort={sort_time*1000:.2f}ms, speedup={speedup:.2f}x")

if __name__ == "__main__":
    print("=" * 60)
    print("CUDA Radix Partition Tests")
    print("=" * 60)
    
    try:
        test_partition_basic()
        test_argpartition_basic()
        test_partition_2d()
        test_partition_dtypes()
        test_partition_non_contiguous()
        
        print("\n" + "=" * 60)
        print("All tests PASSED!")
        print("=" * 60)
        
        benchmark_partition()
        
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
