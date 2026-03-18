# Copyright © 2023-2024 Apple Inc.

import argparse

import mlx.core as mx
import torch
from time_utils import measure_runtime


def benchmark_slice_update_mlx(dst_shape, slice_shape, slice_range, dtype, iters=10):
    def slice_update(arguments):
        for i in range(iters):
            arguments["dst"] = (
                arguments["dst"].at[slice_range].add(arguments["updates"])
            )
        mx.eval(arguments)

    dtype = getattr(mx, dtype)
    arguments = {
        "dst": mx.random.normal(dst_shape).astype(dtype),
        "updates": mx.random.normal(slice_shape).astype(dtype),
    }

    runtime = measure_runtime(slice_update, arguments=arguments)
    bytes_processed = (
        arguments["dst"][slice_range].nbytes * 2 + arguments["updates"].nbytes
    ) * iters
    bandwidth_gb_s = bytes_processed / runtime / 1e6
    return runtime, bandwidth_gb_s


def benchmark_slice_update_torch(
    dst_shape, slice_shape, slice_range, device, dtype, iters=10
):
    def slice_update(dst, updates, slice_range):
        for i in range(iters):
            dst[slice_range] = dst[slice_range] + updates
        if device == torch.device("mps"):
            torch.mps.synchronize()

    dtype = getattr(torch, dtype)
    updates = torch.randn(slice_shape, dtype=dtype).to(device)
    dst = torch.randn(dst_shape, dtype=dtype).to(device)

    runtime = measure_runtime(
        slice_update, dst=dst, updates=updates, slice_range=slice_range
    )
    bytes_processed = (dst[slice_range].nbytes * 2 + updates.nbytes) * iters
    bandwidth_gb_s = bytes_processed / runtime / 1e6
    return runtime, bandwidth_gb_s


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Slice update benchmarks.")
    parser.add_argument("--cpu", action="store_true", help="Use the CPU.")
    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)
        device = torch.device("cpu")
    elif torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError()

    dtypes = ["float32", "bfloat16"]

    test_cases = [
        ((10_000_000,), slice(0, 1_000_000), (1_000_000,)),
        ((100_000,), slice(10_000, 20_000), (10_000,)),
        ((1000, 64), slice(100, 200), (100, 64)),
        ((100, 100, 64), slice(20, 40), (20, 100, 64)),
        (
            (2048, 2048, 128),
            (slice(500, 1500), slice(200, 1200), slice(32, 96)),
            (1000, 1000, 64),
        ),
        (
            (2048, 2048, 128),
            (slice(1800, 1850), slice(100, 200), slice(64, 128)),
            (50, 100, 64),
        ),
        (
            (2048, 2048, 128),
            (slice(1000, 1010), slice(1000, 1010), slice(64, 128)),
            (10, 10, 64),
        ),
    ]

    print(
        f"{'Dtype':<12} {'Dst Shape':<25} {'Update Shape':<20} "
        f"{'MLX (ms)':<12} {'MLX GB/s':<12} {'Torch (ms)':<12} {'Torch GB/s':<12}"
    )
    print("-" * 110)

    for dtype in dtypes:
        for dst_shape, slice_range, update_shape in test_cases:
            mlx_time, mlx_bw = benchmark_slice_update_mlx(
                dst_shape, update_shape, slice_range, dtype
            )
            torch_time, torch_bw = benchmark_slice_update_torch(
                dst_shape, update_shape, slice_range, device, dtype
            )
            print(
                f"{dtype:<12} {str(dst_shape):<25} {str(update_shape):<20} "
                f"{mlx_time:<12.3f} {mlx_bw:<12.2f} {torch_time:<12.3f} {torch_bw:<12.2f}"
            )
