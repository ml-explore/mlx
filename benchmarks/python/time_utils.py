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
        _x = mx.eval(fn(*args, **kwargs))
    toc = time.perf_counter()

    msec = 1e3 * (toc - tic) / num_iters
    print(f"{msec:.5f} msec")


def measure_runtime(fn, *, warmup=5, iters=100, **kwargs):
    """Return mean runtime in milliseconds, evaluating returned MLX arrays.

    Other return values are ignored. Callbacks for other backends must sync
    their own device work. Remaining keyword arguments are passed to fn.
    """
    if not isinstance(warmup, int) or not isinstance(iters, int):
        raise TypeError("warmup and iters must be integers")
    if warmup < 0 or iters <= 0:
        raise ValueError("warmup must be nonnegative and iters must be positive")

    # Warmup
    for _ in range(warmup):
        mx.eval(fn(**kwargs))

    tic = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn(**kwargs))
    return (time.perf_counter() - tic) * 1000 / iters
