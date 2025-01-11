import time

import mlx.core as mx

world = mx.distributed.init()

a = mx.ones((5, 5), mx.int32)
its = 10
its_per_eval = 100


def fn(x):
    for _ in range(its_per_eval):
        x = mx.distributed.all_sum(x)
        x = x - 1
    return x


# warmup
for _ in range(5):
    x = fn(a)
    assert mx.array_equal(x, mx.ones_like(x))

tic = time.perf_counter()

for _ in range(its):
    x = fn(a)
    mx.eval(x)

toc = time.perf_counter()
ms = 1000 * (toc - tic) / (its * its_per_eval)
print(f"Time per iteration {ms:.6f} (ms)")
