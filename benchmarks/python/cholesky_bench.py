# Copyright © 2025 Apple Inc.

import argparse

from time_utils import measure_runtime

# Imports are deferred so this script can run in an environment that has only
# one of the two frameworks installed. That matters in practice: MLX depends on
# nvidia-nccl-cu12 while recent torch wheels want a newer nccl, so the two can
# be awkward to co-install. `--only mlx` / `--only torch` lets each half be
# timed with its own interpreter and the numbers compared afterwards.

SHAPES = [
    # Single matrix, small to large.
    (32, 32),
    (128, 128),
    (512, 512),
    (2048, 2048),
    (4096, 4096),
    # Batched.
    (64, 32, 32),
    (256, 32, 32),
    (64, 128, 128),
    (16, 512, 512),
]


def bench_mlx(shape, upper):
    import mlx.core as mx

    n = shape[-1]
    a = mx.random.uniform(shape=shape)
    # Scaled to O(1) entries and diagonally dominant, so the factorization is
    # well conditioned and the timing is not dominated by denormals.
    A = a @ mx.swapaxes(a, -1, -2) / n + mx.eye(n)
    mx.eval(A)

    def run(A):
        mx.eval(mx.linalg.cholesky(A, upper=upper))

    return measure_runtime(run, A=A)


def bench_torch(shape, upper, device):
    import torch

    n = shape[-1]
    a = torch.rand(shape, dtype=torch.float32, device=device)
    A = a @ a.transpose(-1, -2) / n + torch.eye(n, device=device)

    def run(A):
        torch.linalg.cholesky(A, upper=upper)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    return measure_runtime(run, A=A)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cholesky benchmarks.")
    parser.add_argument("--cpu", action="store_true", help="Use the CPU.")
    parser.add_argument(
        "--only",
        choices=["mlx", "torch", "both"],
        default="both",
        help="Benchmark only one framework (useful when they cannot co-install).",
    )
    args = parser.parse_args()

    device = None
    if args.only in ("mlx", "both"):
        import mlx.core as mx

        mx.set_default_device(mx.cpu if args.cpu else mx.gpu)

    if args.only in ("torch", "both"):
        import torch

        if args.cpu:
            device = torch.device("cpu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("mps")

    for shape in SHAPES:
        for upper in (False, True):
            print("=" * 20)
            print(f"Shape: {shape}, upper={upper}")
            mlx_ms = torch_ms = None
            if args.only in ("mlx", "both"):
                mlx_ms = bench_mlx(shape, upper)
                print(f"MLX:     {mlx_ms:.3f}ms")
            if args.only in ("torch", "both"):
                torch_ms = bench_torch(shape, upper, device)
                print(f"PyTorch: {torch_ms:.3f}ms")
            if mlx_ms is not None and torch_ms is not None:
                print(f"MLX / PyTorch: {mlx_ms / torch_ms:.2f}x")
