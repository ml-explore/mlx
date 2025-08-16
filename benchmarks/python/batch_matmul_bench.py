# Copyright © 2023 Apple Inc.

import argparse

import mlx.core as mx
import numpy as np
from time_utils import time_fn

B = 8
T = 1024
D = 512


def time_batch_matmul(dtype):
    a = mx.array(np.random.uniform(size=(B * T, D)).astype(dtype))
    b = mx.array(np.random.uniform(size=(D, D)).astype(dtype))
    c = mx.array(np.random.uniform(size=(B * T, D)).astype(dtype))
    mx.eval(a, b, c)

    time_fn(mx.matmul, a, b)

    def batch_vjp_first():
        return mx.vjp(mx.matmul, [a, b], [c])[1][0]

    time_fn(batch_vjp_first)

    def batch_vjp_second():
        return mx.vjp(mx.matmul, [a, b], [c])[1][1]

    time_fn(batch_vjp_second)


def time_unbatch_matmul(dtype):
    a = mx.array(np.random.uniform(size=(B * T, D)).astype(dtype))
    b = mx.array(np.random.uniform(size=(D, D)).astype(dtype))
    c = mx.array(np.random.uniform(size=(B * T, D)).astype(dtype))
    mx.eval(a, b, c)
    time_fn(mx.matmul, a, b)

    def unbatch_vjp_first():
        return mx.matmul(c, mx.transpose(b))

    time_fn(unbatch_vjp_first)

    def unbatch_vjp_second():
        return mx.matmul(mx.transpose(a), c)

    time_fn(unbatch_vjp_second)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLX benchmarks.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    args = parser.parse_args()
    if args.gpu:
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)

    for dtype in ("complex64", "float32", "float16"):
        time_batch_matmul(dtype)
        time_unbatch_matmul(dtype)
