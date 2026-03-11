# Copyright © 2023-2024 Apple Inc.

import time

import mlx.core as mx


def time_fn(fn, *args, **kwargs):
    msg = kwargs.pop("msg", None)
    if msg:
        print(f"Timing {msg} ...", end=" ")
    else:
        print(f"Timing {fn.__name__} ...", end=" ")

    # warmup
    for _ in range(5):
        mx.eval(fn(*args, **kwargs))

    num_iters = 100
    tic = time.perf_counter()
    for _ in range(num_iters):
        x = mx.eval(fn(*args, **kwargs))
    toc = time.perf_counter()

    msec = 1e3 * (toc - tic) / num_iters
    print(f"{msec:.5f} msec")


def measure_runtime(fn, num_warmup=15, num_iters=100, num_runs=5, **kwargs):
    """Run fn repeatedly and return median ms per call. More stable than a single run."""
    # Warmup (enough for GPU to settle)
    for _ in range(num_warmup):
        fn(**kwargs)

    times_ms = []
    for _ in range(num_runs):
        tic = time.perf_counter()
        for _ in range(num_iters):
            fn(**kwargs)
        toc = time.perf_counter()
        times_ms.append((toc - tic) * 1000 / num_iters)
    times_ms.sort()
    return times_ms[num_runs // 2]  # median
