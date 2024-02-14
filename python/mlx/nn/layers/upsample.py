# Copyright © 2023-2024 Apple Inc.

from typing import Literal, Tuple, Union

import mlx.core as mx
from mlx.nn.layers.base import Module


def upsample2d_nearest(x: mx.array, scale_factor: Tuple[float, float]):
    # Integer scale_factors means we can simply expand-broadcast and reshape
    if tuple(map(int, scale_factor)) == scale_factor:
        sh, sw = map(int, scale_factor)
        B, H, W, C = x.shape
        x = x[:, :, None, :, None]
        x = mx.broadcast_to(x, (B, H, sh, W, sw, C))
        x = x.reshape(B, H * sh, W * sw, C)
        return x

    # Floating point scale_factor means we need to do indexing
    else:
        sh, sw = scale_factor
        B, H, W, C = x.shape
        new_H = int(H * sh)
        new_W = int(W * sw)
        idx_y = (mx.arange(0, new_H) / sh).astype(mx.int32)
        idx_x = (mx.arange(0, new_W) / sw).astype(mx.int32)
        return x[:, idx_y[:, None], idx_x[None]]


def upsample2d_bilinear(x: mx.array, scale_factor: Tuple[float, float]):
    sh, sw = scale_factor
    B, H, W, C = x.shape
    new_H = int(H * sh)
    new_W = int(W * sw)
    idx_y = mx.arange(0, new_H) * ((H - 1) / (new_H - 1))
    idx_x = mx.arange(0, new_W) * ((W - 1) / (new_W - 1))
    # Compute the sampling grid
    idx_y_t = mx.floor(idx_y).astype(mx.int32)
    idx_y_b = mx.ceil(idx_y).astype(mx.int32)
    idx_x_l = mx.floor(idx_x).astype(mx.int32)
    idx_x_r = mx.ceil(idx_x).astype(mx.int32)
    # Sample
    a = x[:, idx_y_t[:, None], idx_x_l[None]]
    b = x[:, idx_y_t[:, None], idx_x_r[None]]
    c = x[:, idx_y_b[:, None], idx_x_l[None]]
    d = x[:, idx_y_b[:, None], idx_x_r[None]]
    # Compute bilinear interpolation weights
    y_weight = (idx_y - idx_y_t)[:, None, None]
    x_weight = (idx_x - idx_x_l)[None, :, None]
    w_a = (1 - x_weight) * (1 - y_weight)
    w_b = x_weight * (1 - y_weight)
    w_c = y_weight * (1 - x_weight)
    w_d = x_weight * y_weight
    # Interpolate
    return w_a * a + w_b * b + w_c * c + w_d * d


class Upsample(Module):
    r"""Upsamples the given spatial data.

    The input is assumed to be a 4D tensor or 5D tensor where the channels are expected to be last.

    In case of 4D input, the input shape should be :math:`(N, H, W, C)` where:
        - ``N`` is the batch dimension
        - ``H`` is the input image height
        - ``W`` is the input image width
        - ``C`` is the number of input channels

    In case of 5D input, the input shape should be :math:`(N, D, H, W, C)` where:
        - ``N`` is the batch dimension
        - ``D`` is the first spatial dimension
        - ``H`` is the second spatial dimension
        - ``W`` is the third spatial dimension
        - ``C`` is the number of input channels

    Parameters:
        scale_factor (float or Tuple[float, float]): The multiplier for the spatial size.
            If a ``float`` is provided, it is the multiplier for all spatial dimensions.
            Otherwise, the first element of the tuple is the first spatial dimension multiplier,
            the second element of the tuple is the second spatial dimension multipler, and
            the third element of the tuple is the third spatial dimension multiplier.
        mode (str, optional): The upsampling algorithm: one of ``"nearest"`` and
            ``"bilinear"``. Default: ``"nearest"``.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> x = mx.arange(1, 5).reshape((1, 2, 2, 1))
        >>> x
        array([[[[1],
                 [2]],
                [[3],
                 [4]]]], dtype=int32)
        >>> n = nn.Upsample(scale_factor=2, mode='nearest')
        >>> n(x).squeeze()
        array([[1, 1, 2, 2],
               [1, 1, 2, 2],
               [3, 3, 4, 4],
               [3, 3, 4, 4]], dtype=int32)
        >>> b = nn.Upsample(scale_factor=2, mode='bilinear')
        >>> b(x).squeeze()
        array([[1, 1.33333, 1.66667, 2],
               [1.66667, 2, 2.33333, 2.66667],
               [2.33333, 2.66667, 3, 3.33333],
               [3, 3.33333, 3.66667, 4]], dtype=float32)
    """

    def __init__(
        self,
        scale_factor: Union[float, Tuple[float, float]],
        mode: Literal["nearest", "bilinear"] = "nearest",
    ):
        super().__init__()
        if mode not in ["nearest", "bilinear"]:
            raise ValueError(f"[Upsample] Got unsupported upsampling algorithm: {mode}")
        if isinstance(scale_factor, (list, tuple)):
            self.scale_factor = tuple(map(float, scale_factor))
        else:
            self.scale_factor = (float(scale_factor), float(scale_factor))
        self.mode = mode

    def _extra_repr(self) -> str:
        return f"scale_factor={self.scale_factor}, mode={self.mode!r}"

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim != 4:
            raise ValueError(
                f"[Upsample] The input tensor is {x.ndim}D. Currently, only 4D input is currently supported."
            )
        if self.mode == "bilinear":
            return upsample2d_bilinear(x, self.scale_factor)
        else:
            return upsample2d_nearest(x, self.scale_factor)
