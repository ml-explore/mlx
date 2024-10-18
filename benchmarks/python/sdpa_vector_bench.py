import argparse
import math

import mlx.core as mx
from time_utils import time_fn

L = 1024
H = 32
H_k = 32 // 4
D = 128


def attention(q, k, v):
    B, Hq, L, D = q.shape
    _, Hk, S, _ = k.shape
    q = q.reshape(B, Hk, Hq // Hk, L, D)
    k = k[:, :, None, :, :]
    v = v[:, :, None, :, :]
    s = q @ k.transpose(0, 1, 2, 4, 3)
    p = mx.softmax(s.astype(mx.float32), axis=-1).astype(s.dtype)
    o = p @ v
    return o.reshape(B, Hq, L, D)


def sdpa(q, k, v):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)


def time_self_attention_primitives():
    mx.random.seed(3)
    q = mx.random.uniform(shape=(1, H, 1, D))
    k = mx.random.uniform(shape=(1, H_k, L, D))
    v = mx.random.uniform(shape=(1, H_k, L, D))
    mx.eval(q, k, v)
    time_fn(attention, q, k, v)


def time_self_attention_sdpa():
    mx.random.seed(3)
    q = mx.random.uniform(shape=(1, H, 1, D))
    k = mx.random.uniform(shape=(1, H_k, L, D))
    v = mx.random.uniform(shape=(1, H_k, L, D))
    mx.eval(q, k, v)
    time_fn(sdpa, q, k, v)


if __name__ == "__main__":
    time_self_attention_sdpa()
    time_self_attention_primitives()
