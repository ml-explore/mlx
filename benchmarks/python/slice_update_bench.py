# Copyright © 2023-2024 Apple Inc.

import argparse

import mlx.core as mx
import torch
from time_utils import measure_runtime


def benchmark_slice_update_mlx(dst_shape, slice_shape, slice_range):
    def slice_update(dst, updates):
        for i in range(10):
            dst = dst.at[slice_range].add(updates)
        mx.eval(dst)

    updates = mx.random.normal(slice_shape).astype(mx.float32)
    dst = mx.random.normal(dst_shape).astype(mx.float32)

    runtime = measure_runtime(slice_update, dst=dst, updates=updates)
    bytes_processed = (dst[slice_range].nbytes * 2 + updates.nbytes) * 10
    bandwidth_gb_s = bytes_processed / runtime / 1e6
    print(f"MLX: {runtime:.3f}ms, {bandwidth_gb_s:.2f} GB/s")


def benchmark_slice_update_torch(dst_shape, slice_shape, slice_range, device):
    def slice_update(dst, updates, slice_range):
        for i in range(10):
            dst[slice_range] = dst[slice_range] + updates
        if device == torch.device("mps"):
            torch.mps.synchronize()

    updates = torch.randn(slice_shape, dtype=torch.float32).to(device)
    dst = torch.randn(dst_shape, dtype=torch.float32).to(device)

    runtime = measure_runtime(
        slice_update, dst=dst, updates=updates, slice_range=slice_range
    )
    bytes_processed = (dst[slice_range].nbytes * 2 + updates.nbytes) * 10
    bandwidth_gb_s = bytes_processed / runtime / 1e6
    print(f"PyTorch: {runtime:.3f}ms, {bandwidth_gb_s:.2f} GB/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Slice update benchmarks.")
    parser.add_argument("--cpu", action="store_true", help="Use the CPU.")
    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)
        device = torch.device("cpu")
    else:
        device = torch.device("mps")

    dst_shapes = [
        (10_000_000,),
        (1_000_000,),
        (100_000,),
        (10_000,),
        (1_000,),
        (1000, 64),
        (100, 100, 64),
        (10, 100, 100, 64),
    ]
    slice_ranges = [
        slice(0, 1_000_000),
        slice(0, 100_000),
        slice(0, 10_000),
        slice(0, 1_000),
        slice(0, 100),
        slice(0, 100),
        slice(0, 100),
        slice(0, 10),
    ]
    update_shapes = [
        (1_000_000,),
        (100_000,),
        (10_000,),
        (1_000,),
        (100,),
        (100, 64),
        (100, 100, 64),
        (10, 100, 100, 64),
    ]

    for dst_shape, slice_range, update_shape in zip(
        dst_shapes, slice_ranges, update_shapes
    ):
        print("=" * 40)
        print(f"Dst: {dst_shape}, Slice: {slice_range}, Updates: {update_shape}")
        benchmark_slice_update_mlx(dst_shape, update_shape, slice_range)
        benchmark_slice_update_torch(
            dst_shape, update_shape, slice_range, device=device
        )
