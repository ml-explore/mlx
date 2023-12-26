# Copyright © 2023 Apple Inc.

import argparse

import mlx.core as mx
from time_utils import time_fn


def time_add():
    a = mx.random.uniform(shape=(32, 1024, 1024))
    b = mx.random.uniform(shape=(32, 1024, 1024))
    mx.eval(a, b)
    time_fn(mx.add, a, b)

    aT = mx.transpose(a, [0, 2, 1])
    mx.eval(aT)

    def transpose_add(a, b):
        return mx.add(a, b)

    time_fn(transpose_add, aT, b)

    b = mx.random.uniform(shape=(1024,))
    mx.eval(b)

    def slice_add(a, b):
        return mx.add(a, b)

    time_fn(slice_add, a, b)

    b = mx.reshape(b, (1, 1024, 1))
    mx.eval(b)

    def mid_slice_add(a, b):
        return mx.add(a, b)

    time_fn(mid_slice_add, a, b)


def time_matmul():
    a = mx.random.uniform(shape=(1024, 1024))
    b = mx.random.uniform(shape=(1024, 1024))
    mx.eval(a, b)
    time_fn(mx.matmul, a, b)


def time_negative():
    a = mx.random.uniform(shape=(10000, 1000))
    mx.eval(a)

    def negative(a):
        return -a

    mx.eval(a)

    time_fn(negative, a)


def time_exp():
    a = mx.random.uniform(shape=(1000, 100))
    mx.eval(a)
    time_fn(mx.exp, a)


def time_logsumexp():
    a = mx.random.uniform(shape=(64, 10, 10000))
    mx.eval(a)
    time_fn(mx.logsumexp, a, axis=-1)


def time_take():
    a = mx.random.uniform(shape=(10000, 500))
    ids = mx.random.randint(low=0, high=10000, shape=(20, 10))
    ids = [mx.reshape(idx, (-1,)) for idx in ids]
    mx.eval(ids)

    def random_take():
        return [mx.take(a, idx, 0) for idx in ids]

    time_fn(random_take)


def time_reshape_transposed():
    x = mx.random.uniform(shape=(256, 256, 128))
    mx.eval(x)

    def reshape_transposed():
        return mx.reshape(mx.transpose(x, (1, 0, 2)), (-1,))

    time_fn(reshape_transposed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLX benchmarks.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    args = parser.parse_args()
    if args.gpu:
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)

    time_add()
    time_matmul()
    time_exp()
    time_negative()
    time_logsumexp()
    time_take()
    time_reshape_transposed()
