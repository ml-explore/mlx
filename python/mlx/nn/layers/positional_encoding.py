# Copyright © 2023 Apple Inc.

import math
from typing import Optional

import mlx.core as mx
from mlx.nn.layers.base import Module


class RoPE(Module):
    """Implements the rotary positional encoding [1].

    The traditional implementation rotates consecutive pairs of elements in the
    feature dimension while the default implementation rotates pairs with
    stride half the feature dimensions for efficiency.

    [1]: https://arxiv.org/abs/2104.09864

    Args:
        dims (int): The feature dimensions to be rotated. If the input feature
                    is larger than dims then the rest is left unchanged.
        traditional (bool): If set to True choose the traditional
                            implementation which is slightly less efficient.
    """

    def __init__(self, dims: int, traditional: bool = False):
        super().__init__()
        self.dims = dims
        self.traditional = traditional

    def _extra_repr(self):
        return f"{self.dims}, traditional={self.traditional}"

    def _compute_rope(self, costheta, sintheta, x):
        x1 = x[..., : self.dims // 2]
        x2 = x[..., self.dims // 2 : self.dims]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            rx = mx.concatenate([rx1, rx2, x[..., self.dims :]], axis=-1)
        else:
            rx = mx.concatenate([rx1, rx2], axis=-1)

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

        rx = mx.concatenate([rx1[..., None], rx2[..., None]], axis=-1)

        return rx

    def __call__(self, x, offset: int = 0):
        shape = x.shape
        x = mx.reshape(x, (-1, shape[-2], shape[-1]))
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N, self.dims, offset=offset, dtype=x.dtype
        )

        rope = (
            self._compute_traditional_rope if self.traditional else self._compute_rope
        )
        rx = rope(costheta, sintheta, x)

        return mx.reshape(rx, shape)

    @staticmethod
    def create_cos_sin_theta(
        N: int, D: int, offset: int = 0, base: float = 10000, dtype=mx.float32
    ):
        D = D // 2
        positions = mx.arange(offset, N, dtype=dtype)
        freqs = mx.exp(-mx.arange(0.0, D, dtype=dtype) * (math.log(base) / D))
        theta = mx.reshape(positions, (-1, 1)) * mx.reshape(freqs, (1, -1))
        costheta = mx.cos(theta)
        sintheta = mx.sin(theta)

        return costheta, sintheta


class SinusoidalPositionalEncoding(Module):
    """Implements sinusoidal positional encoding similar to [1].

    [1]: https://arxiv.org/abs/1706.03762

    Args:
        dims (int): The dimensionality of the resulting positional embeddings.
        min_freq (float): The minimum frequency expected (default: 0.0001)
        max_freq (float): The maximum frequency expected (default: 1)
        scale (float): Scale the embeddings by that number (default: sqrt(dims//2))
        cos_first (bool): If set to True embed using ``[cos(x); sin(x)]``
            instead of the other way around (default: False)
        full_turns (bool): If set to True multiply the frequencies
            with ``2 pi`` (default: False)
    """

    def __init__(
        self,
        dims: int,
        min_freq: float = 0.0001,
        max_freq: float = 1,
        scale: Optional[float] = None,
        cos_first: bool = False,
        full_turns: bool = False,
    ):
        super().__init__()

        one_zero = 1 - mx.arange(0, dims // 2) / (dims // 2 - 1)
        min_freq = math.log(min_freq)
        max_freq = math.log(max_freq)

        # Start with underscore so it is not included in the parameters
        self._sigmas = mx.exp(one_zero * (max_freq - min_freq) + min_freq)
        if full_turns:
            self._sigmas = self._sigmas * (2 * math.pi)

        # Save some constants that define the implementation
        self.scale = scale or (2 / dims) ** 0.5
        self.cos_first = cos_first

    def __call__(self, x):
        y = x[..., None] * self._sigmas
        cosy = mx.cos(y)
        siny = mx.sin(y)

        if self.cos_first:
            y = mx.concatenate([cosy, siny], axis=-1)
        else:
            y = mx.concatenate([siny, cosy], axis=-1)

        if self.scale != 1:
            y = y * self.scale

        return y
