# Copyright © 2023 Apple Inc.

from typing import List, Literal, Tuple, Union

import mlx.core as mx
from mlx.nn.layers.base import Module


class Upsample2d(Module):
    r"""Upsamples the given spatial data.

    The input  is assumed to be a 4D tensor where the channels are expected to be last.
    Thus, the input shape should be :math:`(N, H, W, C)` where:
        - ``N`` is the batch dimension
        - ``H`` is the input image height
        - ``W`` is the input image width
        - ``C`` is the number of input channels

    Parameters:
        scale (int or Tuple[int, int]):
            The multiplier for spatial size. If the single integer is provided, the provided value 
            is the multiplier for both height and width axis. Otherwise, the first element of the
            tuple is the height multipler, while the second is the width multipler.
        mode (str, optional): The upsampling algorithm: one of ``'nearest'`` and ``'bilinear'``.
            Default: ``'nearest'``.

    Shape:
        - Input:  :math:`(N, C, H, W)` 
        - Output: :math:`(N, C, H_{out}, W_{out})`, where

    .. math::
        \begin{aligned}
            H_{out} &=  H_{in} \times \text{scale} \\
            W_{out} &=  W_{in} \times \text{scale}
        \end{aligned}
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
        scale: Union[int, Tuple[int, int]],
        mode: Literal["nearest", "bilinear"] = "nearest",
    ) -> None:
        super().__init__()
        if mode not in ["nearest", "bilinear"]:
            raise ValueError("[upsample2d] unsupported upsampling algorithm")
        self.scale = scale
        self.mode = mode

    def __call__(self, x: mx.array) -> mx.array:
        if self.mode == "bilinear":
            return self._upsample_bilinear(x)
        else:
            return self._upsample_nearest(x)

    def _upsample_nearest(self, x: mx.array) -> mx.array:
        (batch_size, height, width, channels) = x.shape
        (
            batch_stride,
            height_stride,
            width_stride,
            channels_stride,
        ) = self._get_row_contiguous_strides(x)
        (height_scale, width_scale) = self._get_scale()
        return mx.as_strided(
            x,
            shape=(batch_size, height, height_scale, width, width_scale, channels),
            strides=(batch_stride, height_stride, 0, width_stride, 0, channels_stride),
        ).reshape((batch_size, height * height_scale, width * width_scale, channels))

    def _upsample_bilinear(self, x: mx.array) -> mx.array:
        (batch_size, height, width, channels) = x.shape
        (height_scale, width_scale) = self._get_scale()
        desired_height, desired_width = height * height_scale, width * width_scale
        img_ch_first = x.transpose((0, 3, 1, 2))
        # Compute sampling indices
        height_indices = mx.repeat(mx.arange(desired_height), desired_width)
        width_indices = mx.broadcast_to(
            mx.arange(desired_width), ((desired_height, desired_width))
        ).flatten()
        # Normalize sampling indices
        height_ratio = (height - 1.0) / (desired_height - 1.0)
        width_ratio = (width - 1.0) / (desired_width - 1.0)
        norm_height_indices = height_ratio * height_indices
        norm_width_indices = width_ratio * width_indices
        # Compute the sampling grid
        y_t = mx.floor(norm_height_indices).astype(mx.int32)
        y_b = mx.ceil(norm_height_indices).astype(mx.int32)
        x_l = mx.floor(norm_width_indices).astype(mx.int32)
        x_r = mx.ceil(norm_width_indices).astype(mx.int32)
        # Sample
        a = img_ch_first[..., y_t, x_l]
        b = img_ch_first[..., y_t, x_r]
        c = img_ch_first[..., y_b, x_l]
        d = img_ch_first[..., y_b, x_r]
        # Compute bilinear interpolation weights
        y_weight = norm_height_indices - y_t
        x_weight = norm_width_indices - x_l
        w_a = (1 - x_weight) * (1 - y_weight)
        w_b = x_weight * (1 - y_weight)
        w_c = y_weight * (1 - x_weight)
        w_d = x_weight * y_weight
        # Interpolate
        out = w_a * a + w_b * b + w_c * c + w_d * d
        out = out.reshape((batch_size, channels, desired_height, desired_width))
        # Go back to (B, H, W, C)
        out = out.transpose((0, 2, 3, 1))
        return out

    def _get_scale(self) -> Tuple[int, int]:
        if isinstance(self.scale, int):
            return (self.scale, self.scale)
        return self.scale

    def _get_row_contiguous_strides(self, a: mx.array) -> List[int]:
        return list(
            reversed(mx.cumprod(mx.array([1] + list(reversed(a.shape))))[:-1].tolist())
        )

    def _extra_repr(self) -> str:
        return f"scale={self.scale}, mode={self.mode}"
