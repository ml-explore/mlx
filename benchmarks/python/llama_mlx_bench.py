# Copyright © 2023 Apple Inc.

import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.utils


class LlamaAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.rope = nn.RoPE(dims // num_heads, True)
        self.query_proj = nn.Linear(dims, dims, False)
        self.key_proj = nn.Linear(dims, dims, False)
        self.value_proj = nn.Linear(dims, dims, False)
        self.out_proj = nn.Linear(dims, dims, False)

    def __call__(self, queries, keys, values, mask=None, cache=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        queries = mx.transpose(mx.reshape(queries, (B, L, num_heads, -1)), (0, 2, 1, 3))
        keys = mx.transpose(mx.reshape(keys, (B, L, num_heads, -1)), (0, 2, 1, 3))
        values = mx.transpose(mx.reshape(values, (B, L, num_heads, -1)), (0, 2, 1, 3))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Dimensions are [batch x num heads x sequence x hidden dim]
        scale = mx.array(math.sqrt(1 / queries.shape[-1]), dtype=queries.dtype)
        scores = (queries * scale) @ mx.transpose(keys, (0, 1, 3, 2))
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        values_hat = mx.reshape(mx.transpose(scores @ values, (0, 2, 1, 3)), (B, L, -1))

        return self.out_proj(values_hat), (keys, values)


class LlamaEncoderLayer(nn.Module):
    def __init__(self, dims: int, mlp_dims: int, num_heads: int):
        super().__init__()

        self.attention = LlamaAttention(dims, num_heads)

        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)

        self.linear1 = nn.Linear(dims, mlp_dims, False)
        self.linear2 = nn.Linear(dims, mlp_dims, False)
        self.linear3 = nn.Linear(mlp_dims, dims, False)

    def __call__(self, x, mask=None, cache=None):
        y = self.norm1(x)
        y, cache = self.attention(y, y, y, mask, cache)
        x = x + y

        y = self.norm2(x)
        a = self.linear1(y)
        b = self.linear2(y)
        y = a * mx.sigmoid(a) * b
        y = self.linear3(y)
        x = x + y

        return x, cache


def measure(model, x, cache):
    for i in range(5):
        y, c = model(x, mask=None, cache=cache)
        mx.eval(y, c)

    start = time.time()
    rs = []
    for i in range(5):
        y, c = model(x, mask=None, cache=cache)
        rs.append((y, c))
    mx.eval(rs)
    end = time.time()

    return (end - start) * 1000 / 5


if __name__ == "__main__":
    H = 32
    D = 4096
    F = 43 * 256
    C = 1000
    mx.set_default_device(mx.gpu)
    dtype = mx.float16

    layer = LlamaEncoderLayer(D, F, H)
    layer.update(mlx.utils.tree_map(lambda x: x.astype(dtype), layer.parameters()))
    k1, k2, k3 = mx.random.split(mx.random.key(0), 3)
    x = mx.random.normal([1, 1, D], dtype=dtype)
    cache = [
        mx.random.normal([1, H, C, D // H], dtype=dtype),
        mx.random.normal([1, H, C, D // H], dtype=dtype),
    ]
    mx.eval(x, cache)

    T = measure(layer, x, cache)

    print("Time per layer per token:", T, "ms")
    print("Lower bound total time per token:", T * 32, "ms")
