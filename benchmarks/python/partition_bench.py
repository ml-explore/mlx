# Copyright © 2026 Apple Inc.

import mlx.core as mx
from time_utils import time_fn


def time_partition(shape, k, dtype):
    x = mx.random.normal(shape, dtype=dtype)
    mx.eval(x)
    name = f"{shape[0]}x{shape[1]} {dtype} k={k}"
    time_fn(mx.partition, x, -k, axis=-1, msg=f"partition {name}")
    time_fn(mx.argpartition, x, -k, axis=-1, msg=f"argpartition {name}")
    # The merge sort is the path partition took before the radix kernel
    time_fn(mx.sort, x, axis=-1, msg=f"sort {name}")


if __name__ == "__main__":
    cases = (
        ((4096, 8192), 2048),
        ((2048, 32768), 2048),
        ((8, 131072), 2048),
        ((64, 128000), 40),
        ((2048, 8192), 32),
        ((4096, 256), 8),
        ((8192, 128), 8),
        ((8192, 129), 8),
        ((511, 512), 32),
        ((512, 512), 32),
        ((2048, 1023), 32),
        ((2048, 1024), 32),
        ((32, 4095), 32),
        ((32, 4096), 32),
    )
    for dtype in (mx.float32, mx.bfloat16):
        for shape, k in cases:
            time_partition(shape, k, dtype)
