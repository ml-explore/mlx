import argparse
import math

import mlx.core as mx
from time_utils import time_fn

L = 16384
H = 32
H_k = H // 4
D = 128
V = 128
dtype = mx.float16
loops = 10


def upproject(x, w):
    if w is None:
        return x
    else:
        return x @ w.T


def attention(q, k, v, mask=None, w=None):
    def _sdpa(q, k, v):
        B, Hq, L, D = q.shape
        _, Hk, S, _ = k.shape
        _, _, _, V = v.shape
        q = q.reshape(B, Hk, Hq // Hk, L, D)
        k = k[:, :, None, :, :]
        v = v[:, :, None, :, :]
        s = q @ k.transpose(0, 1, 2, 4, 3)
        if mask is not None:
            m = mx.broadcast_to(mask, (B, Hq, L, S)).reshape(B, Hk, Hq // Hk, L, S)
            s = mx.where(m, s, mx.finfo(s.dtype).min)
        p = mx.softmax(s.astype(mx.float32), axis=-1).astype(s.dtype)
        o = p @ v
        return o.reshape(B, Hq, L, V)

    for i in range(loops):
        q = _sdpa(q, k, v)
        q = upproject(q, w)
    return q


def sdpa(q, k, v, mask=None, w=None):
    for i in range(loops):
        q = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        q = upproject(q, w)
    return q


def time_self_attention_primitives():
    mx.random.seed(3)
    q = mx.random.uniform(shape=(1, H, 1, D)).astype(dtype)
    k = mx.random.uniform(shape=(1, H_k, L, D)).astype(dtype)
    v = mx.random.uniform(shape=(1, H_k, L, V)).astype(dtype)
    w = mx.random.uniform(shape=(D, V)).astype(dtype) if V != D else None
    mx.eval(q, k, v, w)
    time_fn(attention, q, k, v, w=w)


def time_self_attention_sdpa():
    mx.random.seed(3)
    q = mx.random.uniform(shape=(1, H, 1, D)).astype(dtype)
    k = mx.random.uniform(shape=(1, H_k, L, D)).astype(dtype)
    v = mx.random.uniform(shape=(1, H_k, L, V)).astype(dtype)
    w = mx.random.uniform(shape=(D, V)).astype(dtype) if V != D else None
    mx.eval(q, k, v, w)
    time_fn(sdpa, q, k, v, w=w)


def time_self_attention_sdpa_with_mask():
    mx.random.seed(3)
    q = mx.random.uniform(shape=(1, H, 1, D)).astype(dtype)
    k = mx.random.uniform(shape=(1, H_k, L, D)).astype(dtype)
    v = mx.random.uniform(shape=(1, H_k, L, V)).astype(dtype)
    w = mx.random.uniform(shape=(D, V)).astype(dtype) if V != D else None
    mask = mx.full((L,), True)
    mask[L // 2 :] = False
    mx.eval(q, k, v, mask, w)

    def sdpa_mask(*args):
        return sdpa(*args, mask=mask, w=w)

    def attention_mask(*args):
        return attention(*args, mask=mask, w=w)

    time_fn(attention_mask, q, k, v)
    time_fn(sdpa_mask, q, k, v)


if __name__ == "__main__":
    time_self_attention_sdpa()
    time_self_attention_primitives()
    time_self_attention_sdpa_with_mask()
