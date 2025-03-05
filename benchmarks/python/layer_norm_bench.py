# Copyright © 2023-2024 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
from time_utils import time_fn


def layer_norm(x, w, b, eps):
    ot = x.dtype
    x = x.astype(mx.float32)
    mu = mx.mean(x, -1, keepdims=True)
    v = mx.var(x, -1, keepdims=True)
    y = (x - mu) * mx.rsqrt(v + eps)
    if w is not None:
        y = y * w
    if b is not None:
        y = y + b
    return y


def time_layer_norm():
    f1 = lambda x, w, b, y: (layer_norm(x, w, b, 1e-5) * y).sum()
    f2 = lambda x, w, b, y: (mx.fast.layer_norm(x, w, b, 1e-5) * y).sum()
    g1 = mx.grad(f1, argnums=(0, 1, 2))
    g2 = mx.grad(f2, argnums=(0, 1, 2))

    x = mx.random.uniform(shape=(8, 1024, 4096)).astype(mx.float16)
    w = mx.random.uniform(shape=(4096,)).astype(mx.float16)
    b = mx.random.uniform(shape=(4096,)).astype(mx.float16)
    y = mx.random.uniform(shape=(8, 1024, 4096)).astype(mx.float16)
    mx.eval(x, w, b, y)

    def layer_norm_loop(g, x, w, b):
        gx, gw, gb = x, w, b
        for _ in range(32):
            gx, gw, gb = g(gx, gw, gb, y)
        return gx, gw, gb

    time_fn(layer_norm_loop, g1, x, w, b)
    time_fn(layer_norm_loop, g2, x, w, b)
    time_fn(layer_norm_loop, mx.compile(g1), x, w, b)
    time_fn(layer_norm_loop, mx.compile(g2), x, w, b)

    f1 = lambda x, y: (layer_norm(x, None, None, 1e-5) * y).sum()
    f2 = lambda x, y: (mx.fast.layer_norm(x, None, None, 1e-5) * y).sum()
    g1 = mx.grad(f1, argnums=(0,))
    g2 = mx.grad(f2, argnums=(0,))

    x = mx.random.uniform(shape=(8, 1024, 4096)).astype(mx.float16)
    w = mx.random.uniform(shape=(4096,)).astype(mx.float16)
    b = mx.random.uniform(shape=(4096,)).astype(mx.float16)
    y = mx.random.uniform(shape=(8, 1024, 4096)).astype(mx.float16)
    mx.eval(x, w, b, y)

    def layer_norm_loop(g, x):
        gx = x
        for _ in range(32):
            gx = g(gx, y)
        return gx

    time_fn(layer_norm_loop, g1, x)
    time_fn(layer_norm_loop, g2, x)
    time_fn(layer_norm_loop, mx.compile(g1), x)
    time_fn(layer_norm_loop, mx.compile(g2), x)


if __name__ == "__main__":
    time_layer_norm()
