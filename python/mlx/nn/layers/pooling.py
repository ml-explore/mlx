# Copyright © 2023 Apple Inc.


from typing import List, Optional

import mlx.core as mx
from mlx.nn.layers.base import Module


class Pooling(Module):
    def __init__(
        self,
        kernel_size: int | List[int],
        stride: Optional[int | List[int]] = None,
        padding: int | List[int] = 0,
        mode: str = "max",
    ):
        self.kernel_size = kernel_size
        self.stride = stride if stride != None else kernel_size
        self.padding = padding
        if mode not in ["max", "mean"]:
            raise AssertionError(f"unsupported pooling mode")
        self.mode = mode

    def _get_padding(self, features_sizes: List[int]) -> List[int]:
        if isinstance(self.padding, int):
            return (
                [(0, 0)]
                + [(self.padding, self.padding)] * len(features_sizes)
                + [(0, 0)]
            )
        if len(self.padding) != len(features_sizes):
            raise ValueError(
                "the number of provided padding values must match the number of feature axes"
            )
        return [(0, 0)] + self.padding + [(0, 0)]

    def _get_strides(self, features_sizes: List[int]) -> List[int]:
        if isinstance(self.stride, int):
            return [self.stride] * len(features_sizes)
        if len(self.stride) != len(features_sizes):
            raise ValueError(
                "the number of provided strides must match the number of feature axes"
            )
        return self.stride

    def _get_kernel_size(self, features_sizes: List[int]) -> List[int]:
        if isinstance(self.kernel_size, int):
            return [self.kernel_size] * len(features_sizes)
        if len(self.kernel_size) != len(features_sizes):
            raise ValueError("kernel_size must match the number of feature axes")
        return self.kernel_size

    def _pad(self, a: mx.array) -> mx.array:
        feature_size = a.shape[1:-1]
        padding_value = {"max": float("-inf"), "mean": 0}[self.mode]
        return mx.pad(
            a, pad_width=self._get_padding(feature_size), constant_values=padding_value
        )

    def __call__(self, a: mx.array) -> mx.array:
        if a.ndim < 2:
            raise ValueError("the input must be at least two-dimensional.")

        # Select pooling operator
        pool = {"max": mx.max, "mean": mx.mean}[self.mode]
        # Pad if necessary
        a = self._pad(a)

        # a : [B, ..., C]
        batch_size = a.shape[0]
        feature_size = a.shape[1:-1]
        channels = a.shape[-1]

        batch_stride = a.strides[0]
        feature_strides = a.strides[1:-1]
        channels_stride = a.strides[-1]

        kernel_size = self._get_kernel_size(feature_size)
        strides = self._get_strides(feature_size)
        # Compute windows
        windows_shape = tuple(
            [batch_size]
            + [
                (f - k) // s + 1
                for (f, k, s) in zip(feature_size, kernel_size, strides)
            ]
            + kernel_size
            + [channels]
        )
        windows_strides = tuple(
            [batch_stride]
            + [s * f_s for (s, f_s) in zip(strides, feature_strides)]
            + feature_strides
            + [channels_stride]
        )
        windows = mx.as_strided(a, windows_shape, windows_strides)
        # Reduce over windows
        reduction_axes = (
            -1 * mx.arange(start=2, stop=2 + len(kernel_size), step=1) + windows.ndim
        ).tolist()
        return pool(windows, reduction_axes)
