# Copyright © 2023 Apple Inc.


from typing import List, Optional

import mlx.core as mx
from mlx.nn.layers.base import Module


class Pooling(Module):
    r"""Applies pooling over an input signal composed of several channels.

    In one-dimensional case, the output value of the layer with input size :math:`(N, L, C)`
    and output :math:`(N, L_{out}, C)` and :attr:`kernel_size` :math:`kL`  is:

    - If :attr:`mode` is ``'max'``
        .. math::
            out(N_i, k, C_j) = \max_{m=0, \ldots, kL - 1}
                    \text{input}(N_i, \text{stride[0]} \times k + m, C_j)

    - If :attr:`mode` is ``'mean'``
        .. math::
            out(N_i, k, C_j) = \frac{1}{kL} \sum_{m=0, \ldots, kL - 1}
                    \text{input}(N_i, \text{stride[0]} \times k + m, C_j)

    In two-dimensional case, the output value of the layer with input size :math:`(N, H, W, C)`,
    output :math:`(N, H_{out}, W_{out}, C)` and :attr:`kernel_size` :math:`(kH, kW)` is:

    - If :attr:`mode` is ``'max'``
        .. math::
            \begin{aligned}
                out(N_i, h, w, C_j) ={} & \max_{m=0, \ldots, kH-1} \max_{n=0, \ldots, kW-1} \\
                                        & \text{input}(N_i, \text{stride[0]} \times h + m,
                                                    \text{stride[1]} \times w + n, C_j)
            \end{aligned}
    - If :attr:`mode` is ``'mean'``
        .. math::
            \begin{aligned}
                out(N_i, h, w, C_j) =\frac{1} {kH \times kW} & \sum_{m=0, \ldots, kH-1} \sum_{n=0, \ldots, kW-1} \\
                                        & \text{input}(N_i, \text{stride[0]} \times h + m,
                                                    \text{stride[1]} \times w + n, C_j)
            \end{aligned}

    If :attr:`padding` is non-zero, then the input is implicitly padded. If :attr:`mode` is ``'max'``, the input is padded with negative infinity on both sides.
    If :attr:`mode` is ``'mean'``, the input is padded with 0 on both sides. Padding values are included in the calculation.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, can either be:

        - a single ``int`` -- in which case the same value is used for all feature axes
        - a ``list`` of ``int`` -- in which case, the first `int` is used for the first feature axis, the second `int` for the second feature axis, etc.

    Args:
        kernel_size (int or list(int)): The size of the pooling window.
        stride (int or list(int)): The stride of the window. Default: :attr:`kernel_size`.
        padding (int or list(int)): The padding amount on both sides of the feature axis. Default: 0.
        mode (str): Determines whether to perform max pooling or mean pooling. Either ``'max'`` or ``'mean'``. Default: ``'max'``.

    Examples::

        >>> # max pooling of square window of size=2, stride=2
        >>> m = nn.Pooling(2, stride=2, mode='max')
        >>> # mean pooling of square window of size=2, stride=2, padding=1
        >>> m = nn.Pooling(2, stride=2, padding=1, mode='mean')
        >>> output = m(input)

    """

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

    def _extra_repr(self):
        return f"{self.kernel_size}, stride={self.stride}, padding={self.padding}, mode={self.mode}"

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
        return [(0, 0)] + [(p, p) for p in self.padding] + [(0, 0)]

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
        return mx.pad(a, self._get_padding(feature_size), padding_value)

    def __call__(self, a: mx.array) -> mx.array:
        if a.ndim < 2:
            raise ValueError("the input must be at least two-dimensional.")
        # Select pooling operator
        pool = {"max": mx.max, "mean": mx.mean}[self.mode]
        # Pad if necessary
        if self.padding != 0:
            a = self._pad(a)
        # Assumes a.shape = (batch_size, ..., num_channels)
        batch_size, batch_stride = a.shape[0], a.strides[0]
        feature_size, feature_strides = a.shape[1:-1], a.strides[1:-1]
        num_channels, channels_stride = a.shape[-1], a.strides[-1]
        # Get kernel size and strides
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
            + [num_channels]
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
            -1 * mx.arange(start=2, stop=2 + len(kernel_size), step=1)
        ).tolist()
        return pool(windows, reduction_axes)
