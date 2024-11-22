# Copyright © 2023-2024 Apple Inc.

import operator
from itertools import accumulate
from typing import Optional, Tuple, Union

import mlx.core as mx
from mlx.nn.layers.base import Module


def _value_or_list(x, n, msg):
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError(msg)
        return list(x)

    if not isinstance(x, int):
        raise ValueError(msg)

    return [x] * n


def _non_overlapping_sliding_windows(x, shape, window_shape):
    # Compute the intermediate shape
    new_shape = [shape[0]]
    for s, w in zip(shape[1:], window_shape):
        new_shape.append(s // w)
        new_shape.append(w)
    new_shape.append(shape[-1])

    last_axis = len(new_shape) - 1
    axis_order = [0, *range(1, last_axis, 2), *range(2, last_axis, 2), last_axis]

    x = x.reshape(new_shape)
    x = x.transpose(axis_order)
    return x


def _sliding_windows(x, window_shape, window_strides):
    if x.ndim < 3:
        raise ValueError(
            f"To extract sliding windows at least 1 spatial dimension "
            f"(3 total) is needed but the input only has {x.ndim} dimensions."
        )

    spatial_dims = x.shape[1:-1]
    if not (len(spatial_dims) == len(window_shape) == len(window_strides)):
        raise ValueError(
            f"To extract sliding windows the window shapes and strides must have "
            f"the same number of spatial dimensions as the signal but the signal "
            f"has {len(spatial_dims)} dims and the window shape has {len(window_shape)} "
            f"and strides have {len(window_strides)}."
        )

    shape = x.shape
    if all(
        window == stride and size % window == 0
        for size, window, stride in zip(spatial_dims, window_shape, window_strides)
    ):
        return _non_overlapping_sliding_windows(x, shape, window_shape)

    strides = list(reversed(list(accumulate(reversed(shape + (1,)), operator.mul))))[1:]

    # Compute the output shape
    final_shape = [shape[0]]
    final_shape += [
        (size - window) // stride + 1
        for size, window, stride in zip(spatial_dims, window_shape, window_strides)
    ]
    final_shape += window_shape
    final_shape += [shape[-1]]

    # Compute the output strides
    final_strides = strides[:1]
    final_strides += [
        og_stride * stride for og_stride, stride in zip(strides[1:-1], window_strides)
    ]
    final_strides += strides[1:-1]
    final_strides += strides[-1:]  # should always be [1]

    return mx.as_strided(x, final_shape, final_strides)


class _Pool(Module):
    def __init__(self, pooling_function, kernel_size, stride, padding, padding_value):
        super().__init__()

        self._pooling_function = pooling_function
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._padding_value = padding_value
        self._axes = tuple(range(-len(self._kernel_size) - 1, -1, 1))

    def _extra_repr(self):
        ks = tuple(self._kernel_size)
        st = tuple(self._stride)
        pd = tuple(p[0] for p in self._padding)

        return f"kernel_size={ks}, stride={st}, padding={pd}"

    def __call__(self, x):
        if any(p[0] > 0 for p in self._padding):
            x = mx.pad(
                x,
                [(0, 0)] + self._padding + [(0, 0)],
                constant_values=self._padding_value,
            )
        x = _sliding_windows(x, self._kernel_size, self._stride)
        return self._pooling_function(x, self._axes)


class _Pool1d(_Pool):
    def __init__(
        self,
        pooling_function,
        padding_value,
        kernel_size: Union[int, Tuple[int]],
        stride: Optional[Union[int, Tuple[int]]] = None,
        padding: Union[int, Tuple[int]] = 0,
    ):
        class_name = type(self).__name__
        msg = "[{}] '{}' must be an integer or a tuple containing 1 integer"
        kernel_size = _value_or_list(
            kernel_size, 1, msg.format(class_name, "kernel_size")
        )
        if stride is not None:
            stride = _value_or_list(stride, 1, msg.format(class_name, "stride"))
        else:
            stride = kernel_size
        padding = _value_or_list(padding, 1, msg.format(class_name, "padding"))
        padding = [(p, p) for p in padding]

        super().__init__(pooling_function, kernel_size, stride, padding, padding_value)


class _Pool2d(_Pool):
    def __init__(
        self,
        pooling_function,
        padding_value,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Optional[Union[int, Tuple[int, int]]] = 0,
    ):
        class_name = type(self).__name__
        msg = "[{}] '{}' must be an integer or a tuple containing 2 integers"
        kernel_size = _value_or_list(
            kernel_size, 2, msg.format(class_name, "kernel_size")
        )
        if stride is not None:
            stride = _value_or_list(stride, 2, msg.format(class_name, "stride"))
        else:
            stride = kernel_size
        padding = _value_or_list(padding, 2, msg.format(class_name, "padding"))
        padding = [(p, p) for p in padding]

        super().__init__(pooling_function, kernel_size, stride, padding, padding_value)


class _Pool3d(_Pool):
    def __init__(
        self,
        pooling_function,
        padding_value,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Optional[Union[int, Tuple[int, int, int]]] = 0,
    ):
        class_name = type(self).__name__
        msg = "[{}] '{}' must be an integer or a tuple containing 3 integers"
        kernel_size = _value_or_list(
            kernel_size, 3, msg.format(class_name, "kernel_size")
        )
        if stride is not None:
            stride = _value_or_list(stride, 3, msg.format(class_name, "stride"))
        else:
            stride = kernel_size
        padding = _value_or_list(padding, 3, msg.format(class_name, "padding"))
        padding = [(p, p) for p in padding]

        super().__init__(pooling_function, kernel_size, stride, padding, padding_value)


class MaxPool1d(_Pool1d):
    r"""Applies 1-dimensional max pooling.

    Spatially downsamples the input by taking the maximum of a sliding window
    of size ``kernel_size`` and sliding stride ``stride``.

    Args:
        kernel_size (int or tuple(int)): The size of the pooling window kernel.
        stride (int or tuple(int), optional): The stride of the pooling window.
            Default: ``kernel_size``.
        padding (int or tuple(int), optional): How much negative infinity
            padding to apply to the input. The padding amount is applied to
            both sides of the spatial axis. Default: ``0``.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn.layers as nn
        >>> x = mx.random.normal(shape=(4, 16, 5))
        >>> pool = nn.MaxPool1d(kernel_size=2, stride=2)
        >>> pool(x)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]],
        stride: Optional[Union[int, Tuple[int]]] = None,
        padding: Union[int, Tuple[int]] = 0,
    ):
        super().__init__(mx.max, -float("inf"), kernel_size, stride, padding)


class AvgPool1d(_Pool1d):
    r"""Applies 1-dimensional average pooling.

    Spatially downsamples the input by taking the average of a sliding window
    of size ``kernel_size`` and sliding stride ``stride``.

    Args:
        kernel_size (int or tuple(int)): The size of the pooling window kernel.
        stride (int or tuple(int), optional): The stride of the pooling window.
            Default: ``kernel_size``.
        padding (int or tuple(int), optional): How much zero padding to apply to
            the input. The padding amount is applied to both sides of the spatial
            axis. Default: ``0``.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn.layers as nn
        >>> x = mx.random.normal(shape=(4, 16, 5))
        >>> pool = nn.AvgPool1d(kernel_size=2, stride=2)
        >>> pool(x)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int]],
        stride: Optional[Union[int, Tuple[int]]] = None,
        padding: Union[int, Tuple[int]] = 0,
    ):
        super().__init__(mx.mean, 0, kernel_size, stride, padding)


class MaxPool2d(_Pool2d):
    r"""Applies 2-dimensional max pooling.

    Spatially downsamples the input by taking the maximum of a sliding window
    of size ``kernel_size`` and sliding stride ``stride``.

    The parameters ``kernel_size``, ``stride``, and ``padding`` can either be:

    * a single ``int`` -- in which case the same value is used for both the
      height and width axis.
    * a ``tuple`` of two ``int`` s -- in which case, the first ``int`` is
      used for the height axis, the second ``int`` for the width axis.

    Args:
        kernel_size (int or tuple(int, int)): The size of the pooling window.
        stride (int or tuple(int, int), optional): The stride of the pooling
            window. Default: ``kernel_size``.
        padding (int or tuple(int, int), optional): How much negative infinity
            padding to apply to the input. The padding is applied on both sides
            of the height and width axis. Default: ``0``.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn.layers as nn
        >>> x = mx.random.normal(shape=(8, 32, 32, 4))
        >>> pool = nn.MaxPool2d(kernel_size=2, stride=2)
        >>> pool(x)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Optional[Union[int, Tuple[int, int]]] = 0,
    ):
        super().__init__(mx.max, -float("inf"), kernel_size, stride, padding)


class AvgPool2d(_Pool2d):
    r"""Applies 2-dimensional average pooling.

    Spatially downsamples the input by taking the average of a sliding window
    of size ``kernel_size`` and sliding stride ``stride``.

    The parameters ``kernel_size``, ``stride``, and ``padding`` can either be:

    * a single ``int`` -- in which case the same value is used for both the
      height and width axis.
    * a ``tuple`` of two ``int`` s -- in which case, the first ``int`` is
      used for the height axis, the second ``int`` for the width axis.

    Args:
        kernel_size (int or tuple(int, int)): The size of the pooling window.
        stride (int or tuple(int, int), optional): The stride of the pooling
            window. Default: ``kernel_size``.
        padding (int or tuple(int, int), optional): How much zero
            padding to apply to the input. The padding is applied on both sides
            of the height and width axis. Default: ``0``.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn.layers as nn
        >>> x = mx.random.normal(shape=(8, 32, 32, 4))
        >>> pool = nn.AvgPool2d(kernel_size=2, stride=2)
        >>> pool(x)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Optional[Union[int, Tuple[int, int]]] = 0,
    ):
        super().__init__(mx.mean, 0, kernel_size, stride, padding)


class MaxPool3d(_Pool3d):
    r"""Applies 3-dimensional max pooling.

    Spatially downsamples the input by taking the maximum of a sliding window
    of size ``kernel_size`` and sliding stride ``stride``.

    The parameters ``kernel_size``, ``stride``, and ``padding`` can either be:

    * a single ``int`` -- in which case the same value is used for the depth,
      height, and width axis.
    * a ``tuple`` of three ``int`` s -- in which case, the first ``int`` is used
      for the depth axis, the second ``int`` for the height axis, and the third
      ``int`` for the width axis.

    Args:
        kernel_size (int or tuple(int, int, int)): The size of the pooling window.
        stride (int or tuple(int, int, int), optional): The stride of the pooling
            window. Default: ``kernel_size``.
        padding (int or tuple(int, int, int), optional): How much negative infinity
            padding to apply to the input. The padding is applied on both sides
            of the depth, height and width axis. Default: ``0``.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn.layers as nn
        >>> x = mx.random.normal(shape=(8, 16, 32, 32, 4))
        >>> pool = nn.MaxPool3d(kernel_size=2, stride=2)
        >>> pool(x)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Optional[Union[int, Tuple[int, int, int]]] = 0,
    ):
        super().__init__(mx.max, -float("inf"), kernel_size, stride, padding)


class AvgPool3d(_Pool3d):
    r"""Applies 3-dimensional average pooling.

    Spatially downsamples the input by taking the average of a sliding window
    of size ``kernel_size`` and sliding stride ``stride``.

    The parameters ``kernel_size``, ``stride``, and ``padding`` can either be:

    * a single ``int`` -- in which case the same value is used for the depth,
      height, and width axis.
    * a ``tuple`` of three ``int`` s -- in which case, the first ``int`` is used
      for the depth axis, the second ``int`` for the height axis, and the third
      ``int`` for the width axis.

    Args:
        kernel_size (int or tuple(int, int, int)): The size of the pooling window.
        stride (int or tuple(int, int, int), optional): The stride of the pooling
            window. Default: ``kernel_size``.
        padding (int or tuple(int, int, int), optional): How much zero
            padding to apply to the input. The padding is applied on both sides
            of the depth, height and width axis. Default: ``0``.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn.layers as nn
        >>> x = mx.random.normal(shape=(8, 16, 32, 32, 4))
        >>> pool = nn.AvgPool3d(kernel_size=2, stride=2)
        >>> pool(x)
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Optional[Union[int, Tuple[int, int, int]]] = None,
        padding: Optional[Union[int, Tuple[int, int, int]]] = 0,
    ):
        super().__init__(mx.mean, 0, kernel_size, stride, padding)
