import math
import time

import jax
import jax.numpy as jnp
from flax import linen as nn


class RoPE(nn.Module):
    dims: int
    traditional: bool = False

    def _compute_rope(self, costheta, sintheta, x):
        x1 = x[..., : self.dims // 2]
        x2 = x[..., self.dims // 2 : self.dims]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            rx = jnp.concatenate([rx1, rx2, x[..., self.dims :]], axis=-1)
        else:
            rx = jnp.concatenate([rx1, rx2], axis=-1)

        return rx

    def _compute_traditional_rope(self, costheta, sintheta, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            raise NotImplementedError(
                "RoPE doesn't implement partial traditional application"
            )

        rx = jnp.concatenate([rx1[..., None], rx2[..., None]], axis=-1)

        return rx

    @staticmethod
    def create_cos_sin_theta(
        N: int,
        D: int,
        offset: int = 0,
        base: float = 10000,
        dtype=jnp.float32,
    ):
        D = D // 2
        positions = jnp.arange(offset, N, dtype=dtype)
        freqs = jnp.exp(-jnp.arange(0, D, dtype=dtype) * (math.log(base) / D))
        theta = positions.reshape((-1, 1)) * freqs.reshape((1, -1))
        costheta = jnp.cos(theta)
        sintheta = jnp.sin(theta)

        return costheta, sintheta

    @nn.compact
    def __call__(self, x, offset: int = 0):
        shape = x.shape
        x = x.reshape((-1, shape[-2], shape[-1]))
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N, self.dims, offset=offset, dtype=x.dtype
        )

        rope = (
            self._compute_traditional_rope if self.traditional else self._compute_rope
        )
        rx = rope(costheta, sintheta, x)

        return rx.reshape(shape)


class LlamaAttention(nn.Module):
    dims: int
    num_heads: int
    dtype: jnp.dtype

    def setup(self):
        num_heads = self.num_heads
        dims = self.dims

        self.rope = RoPE(dims // num_heads, True)
        self.query_proj = nn.Dense(dims, use_bias=False, param_dtype=self.dtype)
        self.key_proj = nn.Dense(dims, use_bias=False, param_dtype=self.dtype)
        self.value_proj = nn.Dense(dims, use_bias=False, param_dtype=self.dtype)
        self.out_proj = nn.Dense(dims, use_bias=False, param_dtype=self.dtype)

    def __call__(self, queries, keys, values, mask=None, cache=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        queries = queries.reshape((B, L, num_heads, -1)).transpose((0, 2, 1, 3))
        keys = keys.reshape((B, L, num_heads, -1)).transpose((0, 2, 1, 3))
        values = values.reshape((B, L, num_heads, -1)).transpose((0, 2, 1, 3))

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = jnp.concatenate([key_cache, keys], axis=2)
            values = jnp.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Dimensions are [batch x num heads x sequence x hidden dim]
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose((0, 1, 3, 2))
        if mask is not None:
            scores = scores + mask
        scores = jax.nn.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose((0, 2, 1, 3)).reshape((B, L, -1))

        return self.out_proj(values_hat), (keys, values)


class LlamaEncoderLayer(nn.Module):
    dims: int
    mlp_dims: int
    num_heads: int
    dtype: jnp.dtype

    def setup(self):
        dims = self.dims
        mlp_dims = self.mlp_dims
        num_heads = self.num_heads

        self.attention = LlamaAttention(dims, num_heads, dtype)

        self.norm1 = nn.RMSNorm(param_dtype=self.dtype)
        self.norm2 = nn.RMSNorm(param_dtype=self.dtype)

        self.linear1 = nn.Dense(mlp_dims, use_bias=False, param_dtype=self.dtype)
        self.linear2 = nn.Dense(mlp_dims, use_bias=False, param_dtype=self.dtype)
        self.linear3 = nn.Dense(dims, use_bias=False, param_dtype=self.dtype)

    def __call__(self, x, mask=None, cache=None):
        y = self.norm1(x)
        y, cache = self.attention(y, y, y, mask, cache)
        x = x + y

        y = self.norm2(x)
        a = self.linear1(y)
        b = self.linear2(y)
        y = jax.nn.silu(a) * b
        y = self.linear3(y)
        x = x + y

        return x, cache


def measure(model, x, cache):
    for i in range(5):
        y, c = model(x, mask=None, cache=cache)
        jax.block_until_ready((y, c))

    start = time.time()
    for i in range(5):
        y, c = model(x, mask=None, cache=cache)
        jax.block_until_ready((y, c))

    end = time.time()
    return (end - start) * 1000 / 5


if __name__ == "__main__":
    H = 32
    D = 4096
    F = 43 * 256
    C = 1000
    dtype = jnp.float16

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)

    x = jax.random.normal(k1, (1, 1, D), dtype)
    cache = [
        jax.random.normal(k2, [1, H, C, D // H], dtype),
        jax.random.normal(k3, [1, H, C, D // H], dtype),
    ]

    layer = LlamaEncoderLayer(D, F, H, dtype=dtype)
    params = layer.init(k4, x, mask=None, cache=cache)["params"]

    @jax.jit
    def model_fn(x, mask, cache):
        return layer.apply({"params": params}, x, mask=mask, cache=cache)

    T = measure(model_fn, x, cache)

    print("Time per layer per token:", T, "ms")
    print("Lower bound total time per token:", T * 32, "ms")
