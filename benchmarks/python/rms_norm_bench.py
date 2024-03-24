# Copyright © 2023-2024 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
from time_utils import time_fn


def rms_norm(x, w, eps):
    ot = x.dtype
    x = x.astype(mx.float32)
    n = mx.rsqrt(x.square().mean(-1, keepdims=True) + eps)
    return (x * n).astype(ot) * w


def time_rms_norm():
    f1 = lambda x, w, y: (rms_norm(x, w, 1e-5) * y).sum()
    f2 = lambda x, w, y: (mx.fast.rms_norm(x, w, 1e-5) * y).sum()
    g1 = mx.grad(f1, argnums=(0, 1))
    g2 = mx.grad(f2, argnums=(0, 1))

    x = mx.random.uniform(shape=(8, 1024, 4096)).astype(mx.float16)
    w = mx.random.uniform(shape=(4096,)).astype(mx.float16)
    y = mx.random.uniform(shape=(8, 1024, 4096)).astype(mx.float16)
    mx.eval(x, w, y)

    def rms_norm_loop(g, x, w):
        gx, gw = x, w
        for _ in range(32):
            gx, gw = g(gx, gw, y)
        return gx, gw

    time_fn(rms_norm_loop, g1, x, w)
    time_fn(rms_norm_loop, g2, x, w)
    time_fn(rms_norm_loop, mx.compile(g1), x, w)
    time_fn(rms_norm_loop, mx.compile(g2), x, w)


if __name__ == "__main__":
    time_rms_norm()
