# Copyright © 2023-2024 Apple Inc.

from functools import partial

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


def time_layer_norm(N, dt):
    L = 1024
    f1 = lambda x, w, b, y: (layer_norm(x, w, b, 1e-5) * y).sum()
    f2 = lambda x, w, b, y: (mx.fast.layer_norm(x, w, b, 1e-5) * y).sum()
    g1 = mx.grad(f1, argnums=(0, 1, 2))
    g2 = mx.grad(f2, argnums=(0, 1, 2))

    x = mx.random.uniform(shape=(8, L, N)).astype(dt)
    w = mx.random.uniform(shape=(N,)).astype(dt)
    b = mx.random.uniform(shape=(N,)).astype(dt)
    y = mx.random.uniform(shape=(8, L, N)).astype(dt)
    mx.eval(x, w, b, y)

    def layer_norm_loop(f, x, w, b):
        for _ in range(32):
            x = f(x, w, b)
        return x

    time_fn(layer_norm_loop, partial(layer_norm, eps=1e-5), x, w, b)
    time_fn(layer_norm_loop, partial(mx.fast.layer_norm, eps=1e-5), x, w, b)

    def layer_norm_grad_loop(g, x, w, b):
        gx, gw, gb = x, w, b
        for _ in range(32):
            gx, gw, gb = g(gx, gw, gb, y)
        return gx, gw, gb

    time_fn(layer_norm_grad_loop, g1, x, w, b)
    time_fn(layer_norm_grad_loop, g2, x, w, b)
    time_fn(layer_norm_grad_loop, mx.compile(g1), x, w, b)
    time_fn(layer_norm_grad_loop, mx.compile(g2), x, w, b)

    f1 = lambda x, y: (layer_norm(x, None, None, 1e-5) * y).sum()
    f2 = lambda x, y: (mx.fast.layer_norm(x, None, None, 1e-5) * y).sum()
    g1 = mx.grad(f1, argnums=(0,))
    g2 = mx.grad(f2, argnums=(0,))

    x = mx.random.uniform(shape=(8, L, N)).astype(dt)
    w = mx.random.uniform(shape=(N,)).astype(dt)
    b = mx.random.uniform(shape=(N,)).astype(dt)
    y = mx.random.uniform(shape=(8, L, N)).astype(dt)
    mx.eval(x, w, b, y)

    def layer_norm_grad_x_loop(g, x):
        gx = x
        for _ in range(32):
            gx = g(gx, y)
        return gx

    time_fn(layer_norm_grad_x_loop, g1, x)
    time_fn(layer_norm_grad_x_loop, g2, x)
    time_fn(layer_norm_grad_x_loop, mx.compile(g1), x)
    time_fn(layer_norm_grad_x_loop, mx.compile(g2), x)


if __name__ == "__main__":
    for dt in [mx.float32, mx.float16, mx.bfloat16]:
        for n in [1024, 2048, 4096, 8192, 8192 + 1024]:
            print(dt, n)
            time_layer_norm(n, dt)
