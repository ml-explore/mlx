# Copyright © 2026 Apple Inc.

import math
import time

import mlx.core as mx
import numpy as np
import torch

N_WARMUP = 5
N_BENCH = 20


def bench_mlx(a, b):
    for _ in range(N_WARMUP):
        mx.eval(a @ b)

    times = []
    for _ in range(N_BENCH):
        start = time.perf_counter_ns()
        mx.eval(a @ b)
        end = time.perf_counter_ns()
        times.append((end - start) * 1e-9)

    return np.mean(times), np.std(times)


@torch.no_grad()
def bench_torch(a, b):
    for _ in range(N_WARMUP):
        _ = a @ b
        torch.mps.synchronize()

    times = []
    for _ in range(N_BENCH):
        start = time.perf_counter_ns()
        _ = a @ b
        torch.mps.synchronize()
        end = time.perf_counter_ns()
        times.append((end - start) * 1e-9)

    return np.mean(times), np.std(times)


def check_correctness(out_mx, out_pt, rtol, M, N, K):
    if not np.allclose(out_pt, out_mx, rtol=rtol, atol=0):
        abs_diff = np.abs(out_pt - out_mx)
        rel_diff = abs_diff / np.maximum(np.abs(out_pt), 1e-10)

        print(
            f"  WARNING: Correctness failed at {M}x{N}x{K}: "
            f"max_abs={np.max(abs_diff):.6e}, max_rel={np.max(rel_diff):.6e}"
        )


def bench_gemm(M, N, K, dtype, rtol):
    scale = 0.5 / math.sqrt(K)
    a_np = np.random.uniform(0, scale, (M, K)).astype(np.float32)
    b_np = np.random.uniform(0, scale, (K, N)).astype(np.float32)

    a_mx = mx.array(a_np).astype(getattr(mx, dtype))
    b_mx = mx.array(b_np).astype(getattr(mx, dtype))

    a_pt = torch.from_numpy(a_np).to(dtype=getattr(torch, dtype), device="mps")
    b_pt = torch.from_numpy(b_np).to(dtype=getattr(torch, dtype), device="mps")
    torch.mps.synchronize()

    torch_mean, torch_std = bench_torch(a_pt, b_pt)
    mlx_mean, mlx_std = bench_mlx(a_mx, b_mx)

    out_mx = (a_mx @ b_mx).astype(mx.float32)
    out_pt = (a_pt @ b_pt).to(torch.float32).to("cpu").numpy(force=True)
    check_correctness(out_mx, out_pt, rtol, M, N, K)

    return mlx_mean, mlx_std, torch_mean, torch_std


if __name__ == "__main__":
    dtypes = ("bfloat16", "float16", "float32")

    rtols = {
        "float32": 1e-3,
        "float16": 5e-3,
        "bfloat16": 1e-2,
    }

    shapes = (
        (2048, 2048, 10240),
        (2048, 3072, 10240),
        (3072, 3072, 10240),
        (3072, 3072, 12288),
        (3072, 4096, 12288),
        (4096, 4096, 12288),
        (4096, 4096, 18432),
        (4096, 4096, 21504),
        (4096, 6144, 21504),
        (6144, 6144, 21504),
    )

    for dtype in dtypes:
        print(f"\nPerformance ({dtype}):")
        print(
            f"{'M':>5s} {'N':>5s} {'K':>6s}  "
            f"{'MLX (ms)':>15s}  {'Torch (ms)':>15s}  {'Speedup':>10s}"
        )
        print("-" * 80)

        for M, N, K in shapes:
            mlx_mean, mlx_std, torch_mean, torch_std = bench_gemm(
                M, N, K, dtype, rtols[dtype]
            )
            speedup = torch_mean / mlx_mean

            print(
                f"{M:5d} {N:5d} {K:6d}  "
                f"{mlx_mean*1000:7.2f}±{mlx_std*1000:5.2f}  "
                f"{torch_mean*1000:7.2f}±{torch_std*1000:5.2f}  "
                f"{speedup:8.2f}x"
            )
