# Copyright © 2023 Apple Inc.

import math
import time

import torch
import torch.mps
import torch.nn as nn


def sync_if_needed(x):
    if x.device != torch.device("cpu"):
        torch.mps.synchronize()


class RoPE(nn.Module):
    def __init__(self, dims: int, traditional: bool = False):
        super().__init__()
        self.dims = dims
        self.traditional = traditional

    def _compute_rope(self, costheta, sintheta, x):
        x1 = x[..., : self.dims // 2]
        x2 = x[..., self.dims // 2 : self.dims]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            rx = torch.cat([rx1, rx2, x[..., self.dims :]], dim=-1)
        else:
            rx = torch.cat([rx1, rx2], dim=-1)

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

        rx = torch.cat([rx1[..., None], rx2[..., None]], dim=-1)

        return rx

    def forward(self, x, offset: int = 0):
        shape = x.shape
        x = x.view(-1, shape[-2], shape[-1])
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N, self.dims, offset=offset, device=x.device, dtype=x.dtype
        )

        rope = (
            self._compute_traditional_rope if self.traditional else self._compute_rope
        )
        rx = rope(costheta, sintheta, x)

        return rx.view(*shape)

    @staticmethod
    def create_cos_sin_theta(
        N: int,
        D: int,
        offset: int = 0,
        base: float = 10000,
        device="cpu",
        dtype=torch.float32,
    ):
        D = D // 2
        positions = torch.arange(offset, N, dtype=dtype, device=device)
        freqs = torch.exp(
            -torch.arange(0, D, dtype=dtype, device=device) * (math.log(base) / D)
        )
        theta = positions.view(-1, 1) * freqs.view(1, -1)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)

        return costheta, sintheta


class RMSNorm(nn.Module):
    def __init__(self, dims: int, epsilon: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((dims,)))
        self.epsilon = epsilon

    def forward(self, x):
        n = torch.rsqrt(x.square().mean(dim=-1, keepdims=True) + self.epsilon)
        return self.gamma * x * n


class LlamaAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.rope = RoPE(dims // num_heads, True)
        self.query_proj = nn.Linear(dims, dims, bias=False)
        self.key_proj = nn.Linear(dims, dims, bias=False)
        self.value_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

    def forward(self, queries, keys, values, mask=None, cache=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        num_heads = self.num_heads
        B, L, D = queries.shape
        queries = queries.view(B, L, num_heads, -1).permute(0, 2, 1, 3)
        keys = keys.view(B, L, num_heads, -1).permute(0, 2, 1, 3)
        values = values.view(B, L, num_heads, -1).permute(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = torch.cat([key_cache, keys], dim=2)
            values = torch.cat([value_cache, values], dim=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Dimensions are [batch x num heads x sequence x hidden dim]
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.permute(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        scores = torch.softmax(scores, dim=-1)
        values_hat = (scores @ values).permute(0, 2, 1, 3).reshape(B, L, -1)

        return self.out_proj(values_hat), (keys, values)


class LlamaEncoderLayer(nn.Module):
    def __init__(self, dims: int, mlp_dims: int, num_heads: int):
        super().__init__()

        self.attention = LlamaAttention(dims, num_heads)

        self.norm1 = RMSNorm(dims)
        self.norm2 = RMSNorm(dims)

        self.linear1 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear2 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear3 = nn.Linear(mlp_dims, dims, bias=False)

    def forward(self, x, mask=None, cache=None):
        y = self.norm1(x)
        y, cache = self.attention(y, y, y, mask, cache)
        x = x + y

        y = self.norm2(x)
        a = self.linear1(y)
        b = self.linear2(y)
        y = torch.nn.functional.silu(a) * b
        y = self.linear3(y)
        x = x + y

        return x, cache


@torch.no_grad()
def measure(model, x, cache):
    for i in range(5):
        y, c = model(x, mask=None, cache=cache)
    sync_if_needed(x)

    start = time.time()
    for i in range(5):
        y, c = model(x, mask=None, cache=cache)
    sync_if_needed(x)
    end = time.time()
    return (end - start) * 1000 / 5


if __name__ == "__main__":
    H = 32
    D = 4096
    F = 43 * 256
    C = 1000
    device = torch.device("mps")
    dtype = torch.float16

    layer = LlamaEncoderLayer(D, F, H).to(device).to(dtype)
    x = torch.randn(1, 1, D).to(device).to(dtype)
    cache = [
        torch.randn(1, H, C, D // H).to(device).to(dtype),
        torch.randn(1, H, C, D // H).to(device).to(dtype),
    ]

    T = measure(layer, x, cache)

    print("Time per layer per token:", T, "ms")
    print("Lower bound total time per token:", T * 32, "ms")
