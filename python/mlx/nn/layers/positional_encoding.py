# Copyright © 2023 Apple Inc.

import math
from typing import Optional

import mlx.core as mx
from mlx.nn.layers.base import Module


class RoPE(Module):
    """Implements the rotary positional encoding.

    The traditional implementation rotates consecutive pairs of elements in the
    feature dimension while the default implementation rotates pairs with
    stride half the feature dimensions for efficiency.

    For more details see `RoFormer: Enhanced Transformer with Rotary Position
    Embedding <https://arxiv.org/abs/2104.09864>`_.

    Args:
        dims (int): The feature dimensions to be rotated. If the input feature
            is larger than dims then the rest is left unchanged.
        traditional (bool, optional): If set to True choose the traditional
            implementation which is slightly less efficient. Default: ``False``.
        base (float, optional): The base used to compute angular frequency for
            each dimension in the positional encodings. Default: ``10000``.
        scale (float, optional): The scale used to scale the positions. Default: ``1.0``.
        max_position (int, optional): The maximum position for which to precompute the sin and cos values. Default: 2048.
        dtype: Data type of the computation. Default: mx.float32.

    Attributes:
        _cos_sin_theta_key (tuple): Cached key for the precomputed cosine and sine values.
        _cos_sin_theta_value (tuple): Cached cosine and sine values.
        _freq (mx.ndarray): Frequency terms used in computing theta values.
    """

    _cos_sin_theta_key = None
    _cos_sin_theta_value = None
    _freq = None

    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
        max_position: int = 2048,
        dtype=mx.float32,
    ):
        super().__init__()
        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.scale = scale
        self.max_position = max_position
        self._set_cos_sin_cache(dims, max_position, base=base, scale=scale, dtype=dtype)

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
        if N > self.max_position:
            self._set_cos_sin_cache(
                self.dims,
                self.max_position,
                seq_len=N,
                base=self.base,
                scale=self.scale,
                dtype=x.dtype,
            )

        rope = (
            self._compute_traditional_rope if self.traditional else self._compute_rope
        )

        costheta, sintheta = self._cos_sin_theta_value
        rx = rope(costheta[offset:N], sintheta[offset:N], x)

        return mx.reshape(rx, shape)

    @classmethod
    def _set_cos_sin_cache(
        cls,
        dims: int,
        seq_len: int,
        base: float = 10000,
        scale: float = 1.0,
        dtype=mx.float32,
    ):
        if (dims, seq_len, base, scale, dtype) != cls._cos_sin_theta_key:
            D = dims // 2
            cls._freqs = mx.exp(-mx.arange(0.0, D, dtype=dtype) * (math.log(base) / D))
            positions = mx.arange(0, seq_len, dtype=dtype) * scale
            theta = mx.reshape(positions, (-1, 1)) * mx.reshape(cls._freqs, (1, -1))
            cls._cos_sin_theta_key = (dims, seq_len, base, scale, dtype)
            cls._cos_sin_theta_value = (mx.cos(theta), mx.sin(theta))

        return cls._cos_sin_theta_value


class SinusoidalPositionalEncoding(Module):
    r"""Implements sinusoidal positional encoding.

    For more details see the paper `Attention Is All You Need
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        dims (int): The dimensionality of the resulting positional embeddings.
        min_freq (float, optional): The minimum frequency expected. Default:
            ``0.0001``.
        max_freq (float, optional): The maximum frequency expected. Default:
            ``1``.
        scale (float, optional): A multiplicative scale for the embeddings.
            Default: ``sqrt(dims//2)``.
        cos_first (bool, optional): If ``True`` embed using ``[cos(x); sin(x)]``
            instead of the reverse. Default: ``False``.
        full_turns (bool, optional): If ``True`` multiply the frequencies with
            :math:`2\pi`. Default: ``False``.
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


class ALiBi(Module):
    @staticmethod
    def create_alibi_matrix(
        q_sequence_length: int,
        k_sequence_length: int,
        num_heads: int,
        offset: int,
        dtype=mx.float32,
    ):
        x1 = mx.arange(offset, q_sequence_length)
        x2 = mx.arange(0, k_sequence_length)
        distance_matrix = -mx.abs(
            mx.expand_dims(x1[:, None] - x2[None, :], axis=(0, 1))
        )
        alibi_slope = ALiBi.create_alibi_slope(num_heads=num_heads)
        alibi_mask = (distance_matrix * alibi_slope).astype(dtype)
        return alibi_mask

    @staticmethod
    def create_alibi_slope(num_heads):
        x = (2**8) ** (1 / num_heads)
        out = mx.power(x, -mx.arange(1, num_heads + 1))
        return mx.expand_dims(out, axis=(-1, -2))

    def __call__(self, attention_scores, offset=0, mask=None):
        alibi_mask = ALiBi.create_alibi_matrix(
            q_sequence_length=attention_scores.shape[-2] + offset,
            k_sequence_length=attention_scores.shape[-1],
            num_heads=attention_scores.shape[1],
            offset=offset,
            dtype=attention_scores.dtype,
        )
        if mask is not None:
            alibi_mask = alibi_mask + mask
        return attention_scores + alibi_mask
