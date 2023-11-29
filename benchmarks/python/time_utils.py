import time

import mlx.core as mx


def time_fn(fn, *args, **kwargs):
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
