# Copyright © 2023 Apple Inc.

from typing import List, Literal

import mlx.core as mx
from mlx.nn.layers.base import Module


class Upsample2d(Module):
    def __init__(
        self, scale: int, mode: Literal["nearest", "bilinear"] = "bilinear"
    ) -> None:
        super().__init__()
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
        scale = self.scale
        return mx.as_strided(
            x,
            shape=(batch_size, height, scale, width, scale, channels),
            strides=(batch_stride, height_stride, 0, width_stride, 0, channels_stride),
        ).reshape((batch_size, height * scale, width * scale, channels))

    def _upsample_bilinear(self, x: mx.array) -> mx.array:
        (batch_size, height, width, channels) = x.shape
        desired_height, desired_width = height * self.scale, width * self.scale
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

    def _get_row_contiguous_strides(self, a: mx.array) -> List[int]:
        return list(
            reversed(mx.cumprod(mx.array([1] + list(reversed(a.shape))))[:-1].tolist())
        )
