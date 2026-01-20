import mlx.core as mx
from mlx.utils import tree_map
from time_utils import time_fn

L = 32768
H = 32
H_k = H // 4
D = 128
dtype = mx.float16
bits = 4
mode = "mxfp8" if bits == 8 else "mxfp4"

loops = 20


def attention(q, k, v):
    for _ in range(loops):
        B, Hq, Lq, Dq = q.shape
        _, Hk, S, _ = k.shape
        q = q.reshape(B, Hk, Hq // Hk, Lq, Dq)
        ke = k[:, :, None, :, :]
        ve = v[:, :, None, :, :]
        scores = q @ ke.transpose(0, 1, 2, 4, 3)
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        q = probs @ ve
        q = q.reshape(B, Hq, Lq, Dq)
    return q


def sdpa(q, k, v):
    for _ in range(loops):
        q = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=None)
    return q


def quant_sdpa(q, k, v, bits=4, mode="mxfp4"):
    for _ in range(loops):
        q = mx.fast.quantized_scaled_dot_product_attention(
            q, *k, *v, scale=1.0, mask=None, bits=bits, mode=mode
        )
    return q


def quant_attention(q, k, v, bits=4, mode="mxfp4"):
    for _ in range(loops):
        B, Hq, Lq, Dq = q.shape
        Hk = k[0].shape[1]

        q = q.reshape((B, Hk, Hq // Hk, Lq, Dq))
        ke = tree_map(lambda x: mx.expand_dims(x, axis=2), k)
        ve = tree_map(lambda x: mx.expand_dims(x, axis=2), v)

        scores = mx.quantized_matmul(q, *ke, transpose=True, bits=bits, mode=mode)
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)

        q = mx.quantized_matmul(scores, *ve, transpose=False, bits=bits, mode=mode)
        q = q.reshape((B, Hq, Lq, Dq))
    return q


def time_self_attention_primitives(q, k, v):
    time_fn(attention, q, k, v)


def time_self_attention_sdpa(q, k, v):
    time_fn(sdpa, q, k, v)


def time_self_attention_quant_sdpa(q, k, v, bits, mode):
    time_fn(quant_sdpa, q, k, v, bits, mode)


def time_self_attention_quant_primitives(q, k, v, bits, mode):
    time_fn(quant_attention, q, k, v, bits, mode)


if __name__ == "__main__":
    mx.random.seed(3)
    q = mx.random.uniform(shape=(1, H, 1, D), dtype=dtype)
    k = mx.random.uniform(shape=(1, H_k, L, D), dtype=dtype)
    v = mx.random.uniform(shape=(1, H_k, L, D), dtype=dtype)
    mx.eval(q, k, v)

    k_quant = mx.quantize(k, bits=bits, mode=mode)
    v_quant = mx.quantize(v, bits=bits, mode=mode)
    mx.eval(k_quant, v_quant)

    k = mx.dequantize(*k_quant, bits=bits, mode=mode)
    v = mx.dequantize(*v_quant, bits=bits, mode=mode)

    time_self_attention_sdpa(q, k, v)
    time_self_attention_quant_sdpa(q, k_quant, v_quant, bits, mode)
    time_self_attention_primitives(q, k, v)
    time_self_attention_quant_primitives(q, k_quant, v_quant, bits, mode)
