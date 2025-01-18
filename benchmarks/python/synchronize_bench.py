import time

import mlx.core as mx

rank = mx.distributed.init().rank()


def timeit(fn, a):

    # warmup
    for _ in range(5):
        mx.eval(fn(a))

    its = 10
    tic = time.perf_counter()
    for _ in range(its):
        mx.eval(fn(a))
    toc = time.perf_counter()
    ms = 1000 * (toc - tic) / its
    return ms


def all_reduce_benchmark():
    a = mx.ones((5, 5), mx.int32)

    its_per_eval = 100

    def fn(x):
        for _ in range(its_per_eval):
            x = mx.distributed.all_sum(x)
            x = x - 1
        return x

    ms = timeit(fn, a) / its_per_eval
    if rank == 0:
        print(f"All Reduce: time per iteration {ms:.6f} (ms)")


def all_gather_benchmark():
    a = mx.ones((5, 5), mx.int32)
    its_per_eval = 100

    def fn(x):
        for _ in range(its_per_eval):
            x = mx.distributed.all_gather(x)[0]
        return x

    ms = timeit(fn, a) / its_per_eval
    if rank == 0:
        print(f"All gather: time per iteration {ms:.6f} (ms)")


if __name__ == "__main__":
    all_reduce_benchmark()
    all_gather_benchmark()
