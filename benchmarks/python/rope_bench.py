# Copyright © 2023-2024 Apple Inc.

from time_utils import time_fn

import mlx.core as mx
import mlx.nn as nn


def time_rope():
    rope = nn.RoPE(64)

    # vec
    x = mx.random.uniform(shape=(1, 32, 1, 128)).astype(mx.float16)
    mx.eval(x)

    def rope_vec(x):
        for _ in range(32):
            x = rope(x, offset=100)
        return x

    time_fn(rope_vec, x)

    # matrix
    x = mx.random.uniform(shape=(1, 32, 1024, 128)).astype(mx.float16)
    mx.eval(x)

    def rope_mat(x):
        for _ in range(32):
            x = rope(x)
        return x

    time_fn(rope_mat, x)


if __name__ == "__main__":
    time_rope()
