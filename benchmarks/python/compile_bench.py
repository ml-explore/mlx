# Copyright © 2023-2024 Apple Inc.

import argparse
import math
import random

import mlx.core as mx
from time_utils import time_fn


def bench_gelu():

    def gelu(x):
        return x * (1 + mx.erf(x / math.sqrt(2))) / 2

    x = mx.random.uniform(shape=(1000, 1024))

    def gen_fun(fun):
        def bench_fun(x):
            for _ in range(10):
                x = fun(x)
            return x

        return bench_fun

    time_fn(gen_fun(gelu), x, msg="fixed gelu")
    time_fn(gen_fun(mx.compile(gelu)), x, msg="compiled fixed gelu")

    def randint():
        return random.randint(1, x.shape[0])

    def gen_fun(fun):
        def bench_fun(x, y):
            x = x[: randint()]
            for _ in range(10):
                x = fun(x)
                y = fun(y)
            return x, y

        return bench_fun

    y = mx.random.uniform(shape=(1000, 1024))
    time_fn(gen_fun(gelu), x, y, msg="variable gelu")
    time_fn(gen_fun(mx.compile(gelu)), x, y, msg="compiled variable gelu")
    time_fn(
        gen_fun(mx.compile(gelu, shapeless=True)),
        x,
        y,
        msg="shapeless variable gelu",
    )


def bench_layernorm():

    weight = mx.random.uniform(shape=(4096,)).astype(mx.float16)
    bias = mx.random.uniform(shape=(4096,)).astype(mx.float16)
    mx.eval(weight, bias)

    def layernorm(x):
        x = x.astype(mx.float32)
        means = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - means) * mx.rsqrt(var + 1e-4)
        x = x.astype(mx.float16)
        return weight * x + bias

    x = mx.random.uniform(shape=(1000, 4096)).astype(mx.float16)

    def gen_fun(fun):
        def bench_fun(x):
            for _ in range(10):
                x = fun(x)
            return x

        return bench_fun

    time_fn(gen_fun(layernorm), x, msg="fixed layernorm")
    time_fn(gen_fun(mx.compile(layernorm)), x, msg="compiled fixed layernorm")

    def randint():
        return random.randint(1, x.shape[0])

    def gen_fun(fun):
        def bench_fun(x):
            x = x[: randint()]
            for _ in range(10):
                x = fun(x)
            return x

        return bench_fun

    random.seed(0)
    time_fn(gen_fun(layernorm), x, msg="variable layernorm")
    random.seed(0)
    time_fn(gen_fun(mx.compile(layernorm)), x, msg="compiled variable layernorm")
    random.seed(0)
    time_fn(
        gen_fun(mx.compile(layernorm, shapeless=True)),
        x,
        msg="shapeless variable layernorm",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compile benchmarks.")
    args = parser.parse_args()

    bench_gelu()
    bench_layernorm()
