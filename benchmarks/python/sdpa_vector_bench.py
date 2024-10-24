import mlx.core as mx
import numpy as np
from mlx.utils import tree_map
from time_utils import time_fn

L = 65536
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
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=None)


def quant_sdpa(q, k, v, bits=4):
    return mx.fast.quantized_scaled_dot_product_attention(
        q, *k, *v, scale=1.0, mask=None, bits=bits
    )


def quant_attention(q, k, v, bits=4):
    B, Hq, L, D = q.shape
    Hk = k[0].shape[1]

    q = q.reshape((B, Hk, Hq // Hk, L, D))
    k = tree_map(lambda x: mx.expand_dims(x, axis=2), k)
    v = tree_map(lambda x: mx.expand_dims(x, axis=2), v)

    scores = mx.quantized_matmul(q, *k, transpose=True, bits=bits)
    scores = mx.softmax(scores, axis=-1)

    out = mx.quantized_matmul(scores, *v, transpose=False, bits=bits)
    out = out.reshape((B, Hq, L, D))
    return out


def time_self_attention_primitives(q, k, v):
    time_fn(attention, q, k, v)


def time_self_attention_sdpa(q, k, v):
    time_fn(sdpa, q, k, v)


def time_self_attention_quant_sdpa(q, k, v, bits=4):
    time_fn(quant_sdpa, q, k, v, bits)


def time_self_attention_quant_primitives(q, k, v, bits=4):
    time_fn(quant_attention, q, k, v)


if __name__ == "__main__":
    mx.random.seed(3)
    q = mx.random.uniform(shape=(1, H, 1, D))
    k = mx.random.uniform(shape=(1, H_k, L, D))
    v = mx.random.uniform(shape=(1, H_k, L, D))
    mx.eval(q, k, v)

    bits = 4
    k_quant = mx.quantize(k, bits=bits)
    v_quant = mx.quantize(v, bits=bits)
    mx.eval(k_quant, v_quant)

    time_self_attention_sdpa(q, k, v)
    time_self_attention_quant_sdpa(q, k_quant, v_quant, bits)
    time_self_attention_primitives(q, k, v)
    time_self_attention_quant_primitives(q, k_quant, v_quant, bits)
