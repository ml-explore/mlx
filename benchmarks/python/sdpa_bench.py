import argparse
import math

import mlx.core as mx
from time_utils import time_fn

MAX_SEQ = 300
START_SEQ = 100
SEQ_INCREMENT = 50


def time_self_attention_primitives():

    mx.random.seed(3)
    B = 2
    H = 38
    D = 64
    for R in range(START_SEQ, MAX_SEQ, SEQ_INCREMENT):
        q = mx.random.uniform(shape=(B, H, R, D))
        k = mx.random.uniform(shape=(B, H, R, D))
        v = mx.random.uniform(shape=(B, H, R, D))
        scale = 1.0 / math.sqrt(float(D))
        mx.eval(q, k, v)

        def sdpa_primitives(qs, ks, vs, alpha):
            s = (alpha * qs) @ ks.transpose(0, 1, 3, 2)
            p = mx.softmax(s.astype(mx.float32), axis=-1).astype(s.dtype)
            o = p @ vs
            return o

        time_fn(sdpa_primitives, q, k, v, scale)


def time_self_attention_sdpa():

    mx.random.seed(3)
    B = 2
    H = 38
    D = 64
    for R in range(START_SEQ, MAX_SEQ, SEQ_INCREMENT):
        q = mx.random.uniform(shape=(B, H, R, D))
        k = mx.random.uniform(shape=(B, H, R, D))
        v = mx.random.uniform(shape=(B, H, R, D))
        scale = 1.0 / math.sqrt(float(D))
        mx.eval(q, k, v)

        def sdpa_fused(qs, ks, vs, alpha):
            o = mx.fast.scaled_dot_product_attention(qs, ks, vs, scale=alpha)
            return o

        time_fn(sdpa_fused, q, k, v, scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MLX benchmarks.")
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    args = parser.parse_args()
    if args.gpu:
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)

    time_self_attention_sdpa()
    time_self_attention_primitives()
