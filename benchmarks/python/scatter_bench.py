# Copyright © 2023-2024 Apple Inc.

import argparse

import mlx.core as mx
import torch
from time_utils import measure_runtime


def benchmark_scatter_mlx(dst_shape, x_shape, idx_shape):
    def scatter(dst, x, idx):
        dst[idx] = x
        mx.eval(dst)

    idx = mx.random.randint(0, dst_shape[0] - 1, idx_shape)
    x = mx.random.normal(x_shape).astype(mx.float32)
    dst = mx.random.normal(dst_shape).astype(mx.float32)

    runtime = measure_runtime(scatter, dst=dst, x=x, idx=idx)
    print(f"MLX: {runtime:.3f}ms")


def benchmark_scatter_torch(dst_shape, x_shape, idx_shape, device):
    def gather(dst, x, idx, device):
        dst[idx] = x
        if device == torch.device("mps"):
            torch.mps.synchronize()

    idx = torch.randint(0, dst_shape[0] - 1, idx_shape).to(device)
    x = torch.randn(x_shape, dtype=torch.float32).to(device)
    dst = torch.randn(dst_shape, dtype=torch.float32).to(device)

    runtime = measure_runtime(gather, dst=dst, x=x, idx=idx, device=device)
    print(f"PyTorch: {runtime:.3f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gather benchmarks.")
    parser.add_argument("--cpu", action="store_true", help="Use the CPU.")
    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)
        device = torch.device("cpu")
    else:
        device = torch.device("mps")

    dst_shapes = [(10, 64), (100_000, 64), (1_000_000, 64)]
    idx_shapes = [(1_000_000,), (1_000_000,), (100_000,)]
    x_shapes = [(1_000_000, 64), (1_000_000, 64), (100_000, 64)]

    for dst_shape, x_shape, idx_shape in zip(dst_shapes, x_shapes, idx_shapes):
        print("=" * 20)
        print(f"X {x_shape}, Indices {idx_shape}")
        benchmark_scatter_mlx(dst_shape, x_shape, idx_shape)
        benchmark_scatter_torch(dst_shape, x_shape, idx_shape, device=device)
