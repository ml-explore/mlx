# Copyright © 2023-2024 Apple Inc.

from typing import List, Literal, Tuple, Union

import mlx.core as mx
from mlx.nn.layers.base import Module


def upsample2d_nearest(x: mx.array, scale: Tuple[float, float]):
    # Integer scales means we can simply expand-broadcast and reshape
    if tuple(map(int, scale)) == scale:
        sh, sw = map(int, scale)
        B, H, W, C = x.shape
        x = x[:, :, None, :, None]
        x = mx.broadcast_to(x, (B, H, sh, W, sw, C))
        x = x.reshape(B, H * sh, W * sw, C)
        return x

    # Floating point scale means we need to do indexing
    else:
        sh, sw = scale
        B, H, W, C = x.shape
        new_H = int(H * sh)
        new_W = int(W * sw)
        idx_y = (mx.arange(0, new_H) / sh).astype(mx.int32)
        idx_x = (mx.arange(0, new_W) / sw).astype(mx.int32)
        return x[:, idx_y[:, None], idx_x[None]]


def upsample2d_bilinear(x: mx.array, scale: Tuple[float, float]):
    sh, sw = scale
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


class Upsample2d(Module):
    r"""Upsamples the given spatial data.

    The input  is assumed to be a 4D tensor where the channels are expected to be last.
    Thus, the input shape should be :math:`(N, H, W, C)` where:
        - ``N`` is the batch dimension
        - ``H`` is the input image height
        - ``W`` is the input image width
        - ``C`` is the number of input channels

    Parameters:
        scale (float or Tuple[float, float]): The multiplier for spatial size.
            If a single number is provided, the provided value is the
            multiplier for both the height and width. Otherwise, the first
            element of the tuple is the height multipler, while the second is
            the width multipler.
        mode (str, optional): The upsampling algorithm: one of ``nearest`` and
            ``bilinear``. Default: ``nearest``.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> x = mx.arange(1, 5).reshape((1, 2, 2, 1))
        >>> x
        array([[[[1],
                 [2]],
                [[3],
                 [4]]]], dtype=int32)
        >>> n = nn.Upsample2d(scale=2, mode='nearest')
        >>> n(x)
        array([[[[1],
                 [1],
                 [2],
                 [2]],
                [[1],
                 [1],
                 [2],
                 [2]],
                [[3],
                 [3],
                 [4],
                 [4]],
                [[3],
                 [3],
                 [4],
                 [4]]]], dtype=int32)
        >>> b = nn.Upsample2d(scale=2, mode='bilinear')
        >>> b(x)
        array([[[[1],
                 [1.33333],
                 [1.66667],
                 [2]],
                [[1.66667],
                 [2],
                 [2.33333],
                 [2.66667]],
                [[2.33333],
                 [2.66667],
                 [3],
                 [3.33333]],
                [[3],
                 [3.33333],
                 [3.66667],
                 [4]]]], dtype=float32)
    """

    def __init__(
        self,
        scale: Union[float, Tuple[float, float]],
        mode: Literal["nearest", "bilinear"] = "nearest",
    ):
        super().__init__()
        if mode not in ["nearest", "bilinear"]:
            raise ValueError("[upsample2d] unsupported upsampling algorithm")
        if isinstance(scale, (list, tuple)):
            self.scale = tuple(map(float, scale))
        else:
            self.scale = (float(scale), float(scale))
        self.mode = mode

    def _extra_repr(self) -> str:
        return f"scale={self.scale}, mode={self.mode!r}"

    def __call__(self, x: mx.array) -> mx.array:
        if self.mode == "bilinear":
            return upsample2d_bilinear(x, self.scale)
        else:
            return upsample2d_nearest(x, self.scale)
