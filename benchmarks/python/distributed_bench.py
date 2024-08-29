# Copyright © 2024 Apple Inc.

"""
Run with:
    mpirun -n 2 python /path/to/distributed_bench.py
"""

import time

import mlx.core as mx


def time_fn(fn, *args, **kwargs):
    msg = kwargs.pop("msg", None)
    world = mx.distributed.init()
    if world.rank() == 0:
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
    if world.rank() == 0:
        print(f"{msec:.5f} msec")


def time_all_sum():
    shape = (4096,)
    x = mx.random.uniform(shape=shape)
    mx.eval(x)

    def sine(x):
        for _ in range(20):
            x = mx.sin(x)
        return x

    time_fn(sine, x)

    def all_sum_plain(x):
        for _ in range(20):
            x = mx.distributed.all_sum(x)
        return x

    time_fn(all_sum_plain, x)

    def all_sum_with_sine(x):
        for _ in range(20):
            x = mx.sin(x)
            x = mx.distributed.all_sum(x)
        return x

    time_fn(all_sum_with_sine, x)


if __name__ == "__main__":
    time_all_sum()
