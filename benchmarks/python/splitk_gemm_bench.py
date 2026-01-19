# Copyright © 2026 Apple Inc.

import os
import subprocess
import sys

N_warmup = 5
N_iter_bench = 20


def run_gemm_subprocess(M, N, K, dtype, disable_splitk):
    env = os.environ.copy()
    env["MLX_DISABLE_SPLITK_NAX"] = "1" if disable_splitk else "0"

    code = f"""
import time
import mlx.core as mx

mx.set_default_device(mx.gpu)
a = mx.random.normal(({M}, {K})).astype(mx.{dtype})
b = mx.random.normal(({K}, {N})).astype(mx.{dtype})
mx.eval(a)
mx.eval(b)

for _ in range({N_warmup}):
    mx.eval(a @ b)

s = time.perf_counter_ns()
for _ in range({N_iter_bench}):
    mx.eval(a @ b)
e = time.perf_counter_ns()

print((e - s) * 1e-9)
"""

    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed: {result.stderr}")

    return float(result.stdout.strip())


def bench_shape_splitk(M, N, K, dtype):
    time_regular = run_gemm_subprocess(M, N, K, dtype, disable_splitk=True)
    time_splitk = run_gemm_subprocess(M, N, K, dtype, disable_splitk=False)
    speedup = time_regular / time_splitk
    return time_regular, time_splitk, speedup


if __name__ == "__main__":
    dtypes = ("bfloat16", "float16", "float32")

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
            f"{'M':>5s} {'N':>5s} {'K':>6s}  {'Regular':>10s}  {'Split-K':>10s}  {'Speedup':>10s}"
        )
        print("-" * 70)

        for M, N, K in shapes:
            t_reg, t_sp, speedup = bench_shape_splitk(M, N, K, dtype)

            print(
                f"{M:5d} {N:5d} {K:6d}  "
                f"{t_reg*1000:8.2f}ms  {t_sp*1000:8.2f}ms  "
                f"{speedup:8.2f}x"
            )
