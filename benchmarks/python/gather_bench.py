from time import time

import mlx.core as mx
import torch


def measure_runtime(fn, **kwargs):
    # Warmup
    for _ in range(5):
        fn(**kwargs)

    tic = time()
    iters = 10
    for _ in range(iters):
        fn(**kwargs)
    return (time() - tic) * 1000 / iters


def benchmark_gather_mlx(x_shape, idx_shape):
    def gather(x, idx):
        mx.eval(x[idx])

    idx = mx.random.randint(0, x_shape[0] - 1, idx_shape)
    x = mx.random.normal(x_shape).astype(mx.float32)

    runtime = measure_runtime(gather, x=x, idx=idx)
    print(f"MLX: {runtime:.3f}ms")


def benchmark_gather_torch(x_shape, idx_shape, device):
    def gather(x, idx, device):
        _ = x[idx]
        if device == torch.device("mps"):
            torch.mps.synchronize()

    idx = torch.randint(0, x_shape[0] - 1, idx_shape).to(device)
    x = torch.randn(x_shape, dtype=torch.float32).to(device)

    runtime = measure_runtime(gather, x=x, idx=idx, device=device)
    print(f"PyTorch: {runtime:.3f}ms")


if __name__ == "__main__":
    cpu = False
    if cpu:
        mx.set_default_device(mx.cpu)
        device = torch.device("cpu")
    else:
        device = torch.device("mps")

    idx_shapes = [(1_000_000,), (100_000,)]
    x_shapes = [(100, 64), (100, 1024)]

    for x_shape, idx_shape in zip(x_shapes, idx_shapes):
        print("=" * 20)
        print(f"X {x_shape}, Indices {idx_shape}")
        benchmark_gather_mlx(x_shape, idx_shape)
        benchmark_gather_torch(x_shape, idx_shape, device=device)
