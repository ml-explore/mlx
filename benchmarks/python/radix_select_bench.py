#!/usr/bin/env python3
"""
Benchmark script for MLX argpartition/partition operations.
Compares radix select implementation against full argsort.
"""

import argparse
import ctypes
import time

import mlx.core as mx
import numpy as np

# Mapping from string names to MLX dtype objects
DTYPE_MAP = {
    "bool": mx.bool_,
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
    "float64": mx.float64,
    "int8": mx.int8,
    "int16": mx.int16,
    "int32": mx.int32,
    "int64": mx.int64,
    "uint8": mx.uint8,
    "uint16": mx.uint16,
    "uint32": mx.uint32,
    "uint64": mx.uint64,
}

# Benchmark-side model for cross-GPU small-kernel dispatch policy.
RADIX_ITEMS_BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
MAX_RADIX_ITEMS_PER_THREAD = 64
SMALL_RADIX_SIZE = 32
WARP_SIZE = 32
CUDA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK = 8
CUDA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97

DTYPE_SIZE_BYTES = {
    "bool_": 1,
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "uint16": 2,
    "uint32": 4,
    "uint64": 8,
}


def _dtype_size_bytes(dtype):
    dtype_name = str(dtype).split(".")[-1]
    return DTYPE_SIZE_BYTES[dtype_name]


def _cuda_max_shared_mem_per_block(default=48 * 1024):
    """Query max(base, optin) shared memory per block; fallback to 48KB."""
    try:
        cudart = ctypes.CDLL("libcudart.so")

        cuda_get_device = cudart.cudaGetDevice
        cuda_get_device.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cuda_get_device.restype = ctypes.c_int

        cuda_device_get_attribute = cudart.cudaDeviceGetAttribute
        cuda_device_get_attribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        cuda_device_get_attribute.restype = ctypes.c_int

        dev = ctypes.c_int()
        if cuda_get_device(ctypes.byref(dev)) != 0:
            return default

        smem_base = ctypes.c_int()
        if (
            cuda_device_get_attribute(
                ctypes.byref(smem_base),
                CUDA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK,
                dev.value,
            )
            != 0
        ):
            return default

        smem_optin = ctypes.c_int()
        optin_rc = cuda_device_get_attribute(
            ctypes.byref(smem_optin),
            CUDA_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            dev.value,
        )
        if optin_rc == 0:
            return max(int(smem_base.value), int(smem_optin.value))
        return int(smem_base.value)
    except Exception:
        return default


def _radix_small_block_threads(vocab_size):
    if vocab_size <= 128:
        return 16
    if vocab_size <= 256:
        return 32
    if vocab_size <= 512:
        return 64
    if vocab_size <= 1024:
        return 128
    return 256


def _radix_small_dispatch_items(required_items):
    for bucket in RADIX_ITEMS_BUCKETS:
        if required_items <= bucket:
            return bucket
    return None


def _radix_small_shared_mem_bytes(dtype_size, block_threads, items_per_thread):
    tile_size = block_threads * items_per_thread
    num_warps = block_threads // WARP_SIZE
    return (
        tile_size * dtype_size
        + tile_size * 4
        + SMALL_RADIX_SIZE * 4
        + (2 + 3 * num_warps + 6) * 4
    )


def estimate_small_kernel_limit(dtype):
    """Estimate max small-kernel axis for dtype under current CUDA radix policy."""
    dtype_size = _dtype_size_bytes(dtype)
    smem_limit = _cuda_max_shared_mem_per_block()
    max_axis = 0
    # 256 is the largest block_threads in sort.cu launch selection.
    for v in range(1, 256 * MAX_RADIX_ITEMS_PER_THREAD + 1):
        block_threads = _radix_small_block_threads(v)
        required_items = (v + block_threads - 1) // block_threads
        if required_items > MAX_RADIX_ITEMS_PER_THREAD:
            continue
        items_per_thread = _radix_small_dispatch_items(required_items)
        if items_per_thread is None:
            continue
        if (
            _radix_small_shared_mem_bytes(dtype_size, block_threads, items_per_thread)
            <= smem_limit
        ):
            max_axis = v
    return {
        "max_axis": max_axis,
        "smem_limit": smem_limit,
    }


def parse_dtypes(dtype_str):
    """Parse comma-separated dtype string into MLX dtype objects."""
    dtypes = []
    for dtype_str_item in dtype_str.split(","):
        dtype_str_item = dtype_str_item.strip().lower()
        if not dtype_str_item:
            continue
        if dtype_str_item not in DTYPE_MAP:
            raise ValueError(
                f"Unknown dtype: {dtype_str_item}. "
                f"Supported dtypes: {', '.join(DTYPE_MAP.keys())}"
            )
        dtypes.append((DTYPE_MAP[dtype_str_item], dtype_str_item))
    return dtypes


def benchmark_argpartition(b, v, k, dtype=mx.bfloat16, warmup=5, iterations=100):
    x = mx.random.uniform(shape=(b, v)).astype(dtype)
    mx.eval(x)
    for _ in range(warmup):
        mx.eval(mx.argpartition(x, kth=k, axis=-1))
    start = time.perf_counter()
    for _ in range(iterations):
        mx.eval(mx.argpartition(x, kth=k, axis=-1))
    return (time.perf_counter() - start) / iterations * 1000


def benchmark_argsort(b, v, dtype=mx.bfloat16, warmup=5, iterations=100):
    x = mx.random.uniform(shape=(b, v)).astype(dtype)
    mx.eval(x)
    for _ in range(warmup):
        mx.eval(mx.argsort(x, axis=-1))
    start = time.perf_counter()
    for _ in range(iterations):
        mx.eval(mx.argsort(x, axis=-1))
    return (time.perf_counter() - start) / iterations * 1000


def verify_correctness(b, v, k, dtype=mx.float32):
    # Quantize random values to induce duplicates and stress tie handling.
    x = mx.random.uniform(shape=(b, v))
    x = mx.floor(x * 257.0).astype(mx.float32)
    x = x.astype(dtype)
    mx.eval(x)

    indices = mx.argpartition(x, kth=k, axis=-1)
    mx.eval(indices)

    # NumPy does not always expose bfloat16 buffers reliably in this environment.
    x_np = np.array(x.astype(mx.float32)) if dtype == mx.bfloat16 else np.array(x)
    indices_np = np.array(indices)
    is_float = np.issubdtype(x_np.dtype, np.floating)

    assert indices_np.shape == (
        b,
        v,
    ), f"Unexpected argpartition output shape: got {indices_np.shape}, expected {(b, v)}"
    assert np.issubdtype(
        indices_np.dtype, np.integer
    ), f"Argpartition indices must be integer, got {indices_np.dtype}"

    for i in range(b):
        row = x_np[i]
        row_idx = indices_np[i]

        assert np.all(
            (row_idx >= 0) & (row_idx < v)
        ), f"Row {i}: out-of-range indices found"
        assert (
            np.unique(row_idx).size == v
        ), f"Row {i}: indices are not a permutation of [0, {v})"

        pv = row[row_idx]
        pivot = pv[k]
        left = pv[:k]
        right = pv[k + 1 :]

        if is_float and np.isnan(pivot):
            non_nan_count = np.count_nonzero(~np.isnan(row))
            assert (
                non_nan_count <= k
            ), f"Row {i}: pivot is NaN before all finite values are placed"
            assert np.all(
                np.isnan(pv[k:])
            ), f"Row {i}: values after NaN pivot must all be NaN"
            continue

        if is_float:
            left_ok = np.all((~np.isnan(left)) & (left <= pivot))
            right_ok = np.all(np.isnan(right) | (right >= pivot))
        else:
            left_ok = np.all(left <= pivot)
            right_ok = np.all(right >= pivot)

        assert left_ok, f"Row {i}: elements before kth violate partition property"
        assert right_ok, f"Row {i}: elements after kth violate partition property"

        # Rank consistency: kth must lie within [count(<pivot), count(<=pivot)).
        less = np.count_nonzero(row < pivot)
        less_equal = np.count_nonzero(row <= pivot)
        assert (
            less <= k < less_equal
        ), f"Row {i}: kth rank inconsistent (less={less}, less_equal={less_equal}, kth={k})"

    return True


def verify_tie_determinism(b=64, v=1024, k=None, dtype=mx.float32, axis=-1):
    """Check repeated argpartition calls return identical outputs on all-equal input."""
    if k is None:
        k = v // 2

    x = mx.zeros((b, v), dtype=dtype)
    mx.eval(x)

    outputs = []
    for _ in range(8):
        idx = mx.argpartition(x, kth=k, axis=axis)
        mx.eval(idx)
        outputs.append(np.array(idx))

    unique_outputs = len({out.tobytes() for out in outputs})
    if unique_outputs != 1:
        raise AssertionError(
            f"Non-deterministic tie ordering detected: "
            f"{unique_outputs}/8 unique outputs for all-equal input "
            f"(shape=({b}, {v}), kth={k}, dtype={dtype})"
        )

    # If deterministic, verify tie ordering matches original merge-sort order.
    expected = mx.argsort(x, axis=axis)
    mx.eval(expected)
    expected_np = np.array(expected)
    if not np.array_equal(outputs[0], expected_np):
        raise AssertionError(
            "Deterministic tie ordering does not match merge-sort baseline "
            f"(shape=({b}, {v}), kth={k}, dtype={dtype})"
        )
    return True


def sweep_kernel(
    dtype=mx.bfloat16,
    k_ratio=0.004,
    warmup=10,
    iterations=50,
    verify=False,
    small_kernel=False,
):
    dtype_name = str(dtype).split(".")[-1]
    limit = estimate_small_kernel_limit(dtype)
    max_small_axis = limit["max_axis"]
    smem_kb = limit["smem_limit"] / 1024.0
    print(
        f"\nDtype={dtype_name}  k=vocab*{k_ratio:.3f}  "
        f"small-kernel-limit≈{max_small_axis}  "
        f"smem={smem_kb:.1f}KB"
    )
    print()

    candidate_vocab = {
        32,
        64,
        96,
        160,
        256,
        384,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
        131072,
    }

    if small_kernel:
        batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        vocab_sizes = sorted({int(v) for v in candidate_vocab if v <= max_small_axis})
    else:
        batch_sizes = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048]
        vocab_sizes = sorted({int(v) for v in candidate_vocab if v > small_kernel})

    col_w = 10
    print(f"{'':>8}", end="")
    for v in vocab_sizes:
        label = f"v={v}"
        print(f"  {label:^{col_w}}", end="")
    print()

    for b in batch_sizes:
        print(f"b={b:<6}", end="")
        for v in vocab_sizes:
            k = max(1, int(v * k_ratio))
            try:
                x = mx.random.uniform(shape=(b, v)).astype(dtype)
                mx.eval(x)
                for _ in range(warmup):
                    mx.eval(mx.argpartition(x, kth=k, axis=-1))
                start = time.perf_counter()
                for _ in range(iterations):
                    mx.eval(mx.argpartition(x, kth=k, axis=-1))
                radix_ms = (time.perf_counter() - start) / iterations * 1000
                for _ in range(warmup):
                    mx.eval(mx.argsort(x, axis=-1))
                start = time.perf_counter()
                for _ in range(iterations):
                    mx.eval(mx.argsort(x, axis=-1))
                argsort_ms = (time.perf_counter() - start) / iterations * 1000

                if verify:
                    verify_correctness(b, v, k, dtype=dtype)
                    verify_tie_determinism(b, v, k, dtype=dtype)

                speedup = argsort_ms / radix_ms
                cell = f"{speedup:>5.2f}x"
                print(f"  {cell:^{col_w}}", end="")
            except Exception:
                print(f"  {'ERR':^{col_w}}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MLX radix select implementation"
    )
    parser.add_argument(
        "--large-kernel-sweep",
        action="store_true",
        help="Enable large-kernel-focused sweep (default: disabled)",
    )
    parser.add_argument(
        "--small-kernel-sweep",
        action="store_true",
        help="Enable small-kernel-focused sweep around the estimated boundary",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable correctness verification (default: disabled). "
        "Disabled when --large-kernel-sweep is enabled.",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        default="bfloat16,float32",
        help="Comma-separated data types to test (default: bfloat16,float32). "
        "Supported: bool, bfloat16, float16, float32, float64, "
        "int8, int16, int32, int64, uint8, uint16, uint32, uint64",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MLX Radix Select Benchmark")
    print("=" * 70)

    configs = [
        (2048, 8192, 32),
        (2048, 4096, 32),
        (1024, 4096, 16),
        (512, 2048, 64),
        (256, 1024, 32),
        (128, 512, 16),
        (1, 128000, 64),
        (1, 512, 32),
        (16, 8192, 32),
        (32, 8192, 32),
        (64, 8192, 32),
    ]

    try:
        dtypes = parse_dtypes(args.dtypes)
    except ValueError as e:
        print(f"Error: {e}")
        return

    if args.large_kernel_sweep and args.small_kernel_sweep:
        print("Error: choose only one of --large-kernel-sweep or --small-kernel-sweep")
        return

    if not args.large_kernel_sweep and not args.small_kernel_sweep:
        if args.verify:
            print("\n1. Correctness Verification")
            print("-" * 40)
            for dtype, dtype_name in dtypes:
                for b, v, k in configs:
                    try:
                        verify_correctness(b, v, k, dtype=dtype)
                        print(f"  [PASS] b={b}, v={v}, k={k}, dtype={dtype_name}")
                    except AssertionError as e:
                        print(f"  [FAIL] b={b}, v={v}, k={k}, dtype={dtype_name}: {e}")

            print("\n2. Tie Determinism Verification")
            print("-" * 40)
            for dtype, dtype_name in dtypes:
                for b, v, k in configs:
                    try:
                        verify_tie_determinism(b=b, v=v, k=k, dtype=dtype)
                        print(
                            f"  [PASS] all-equal input "
                            f"(b={b}, v={v}, k={k}), dtype={dtype_name}, runs=8"
                        )
                    except AssertionError as e:
                        print(
                            f"  [FAIL] all-equal input "
                            f"(b={b}, v={v}, k={k}), dtype={dtype_name}, runs=8: {e}"
                        )

            print("\n3. Performance Benchmarks")
        else:
            print("\nPerformance Benchmarks")
        print("-" * 70)

        for dtype, dtype_name in dtypes:
            print(f"\nDtype: {dtype_name}")
            print(
                f"{'Config':<25} {'ArgPartition':>14} {'ArgSort':>12} {'Speedup':>10}"
            )
            print("-" * 80)

            for b, v, k in configs:
                try:
                    argpart_ms = benchmark_argpartition(
                        b, v, k, dtype, warmup=3, iterations=50
                    )
                    argsort_ms = benchmark_argsort(b, v, dtype, warmup=3, iterations=50)
                    speedup = argsort_ms / argpart_ms
                    config_str = f"b={b}, v={v}, k={k}"
                    print(
                        f"{config_str:<25} {argpart_ms:>12.3f}ms"
                        f" {argsort_ms:>10.3f}ms  {speedup:>5.2f}x"
                    )
                except Exception as e:
                    print(f"b={b}, v={v}, k={k}: Error - {e}")

    if args.large_kernel_sweep or args.small_kernel_sweep:
        print("\nKernel Sweep" + (" (with verification)" if args.verify else ""))
        print("-" * 70)
        for dtype, dtype_name in dtypes:
            sweep_kernel(
                dtype, verify=args.verify, small_kernel=args.small_kernel_sweep
            )

    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
