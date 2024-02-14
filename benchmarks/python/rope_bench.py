# Copyright © 2023-2024 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
from time_utils import time_fn


def time_rope():
    rope = nn.RoPE(4096)

    # vec
    x = mx.random.uniform(shape=(1, 4096)).astype(mx.float16)
    mx.eval(x)

    def rope_vec(x):
        for _ in range(32):
            x = rope(x)
        return x

    time_fn(rope_vec, x)

    # matrix
    x = mx.random.uniform(shape=(1024, 4096)).astype(mx.float16)
    mx.eval(x)

    def rope_mat(x):
        for _ in range(32):
            x = rope(x)
        return x

    time_fn(rope_mat, x)


if __name__ == "__main__":
    time_rope()
