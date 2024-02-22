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


def measure_runtime(fn, **kwargs):
    # Warmup
    for _ in range(5):
        fn(**kwargs)

    tic = time.time()
    iters = 100
    for _ in range(iters):
        fn(**kwargs)
    return (time.time() - tic) * 1000 / iters
