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


class AdaptiveAvgPool2d(Module):
    r"""Applies 2-dimensional adaptive average pooling.

    The output size is H x W, for any input size. The number of output
    features is equal to the number of input planes.

    Args:
        output_size: the target output size of the form H x W.
            Can be a tuple (H, W) or a single int for a square image.
            H and W can be either an ``int``, or ``None`` which means the size
            will be the same as that of the input.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> x = mx.random.normal(shape=(8, 32, 32, 4))
        >>> pool = nn.AdaptiveAvgPool2d((5, 7))
        >>> pool(x)
        >>> pool = nn.AdaptiveAvgPool2d(7)
        >>> pool(x)
    """

    def __init__(self, output_size: Union[int, Tuple[Optional[int], Optional[int]]]):
        super().__init__()
        self.output_size = output_size

    def __call__(self, x):
        return adaptive_avg_pool2d(x, self.output_size)


def adaptive_avg_pool2d(
    x: mx.array, output_size: Union[int, Tuple[Optional[int], Optional[int]]]
) -> mx.array:
    r"""Apply 2-dimensional adaptive average pooling.

    Args:
        x: Input array of shape (N, H, W, C) or (H, W, C).
        output_size: Target output size (H, W) or single int for square output.
            Values can be None to keep the corresponding input dimension.

    Returns:
        Output array with spatial dimensions matching output_size.
    """
    # Parse output_size
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    elif len(output_size) == 1:
        output_size = (output_size[0], output_size[0])

    # Get input dimensions
    *batch_dims, H, W, C = x.shape

    # Handle None values in output_size
    output_H = H if output_size[0] is None else output_size[0]
    output_W = W if output_size[1] is None else output_size[1]

    # If already the right size, return as is
    if H == output_H and W == output_W:
        return x

    # Calculate kernel size and stride
    kernel_H = H // output_H
    kernel_W = W // output_W
    stride_H = H // output_H
    stride_W = W // output_W

    # For exact division, use regular pooling
    if H % output_H == 0 and W % output_W == 0:
        # Reshape for pooling: (batch..., H, W, C) -> (batch..., output_H, kernel_H, output_W, kernel_W, C)
        new_shape = batch_dims + [output_H, kernel_H, output_W, kernel_W, C]
        x_reshaped = x.reshape(new_shape)

        # Average over kernel dimensions (kernel_H is at -4, kernel_W is at -2)
        result = mx.mean(
            x_reshaped, axis=[-4, -2]
        )  # Average over kernel_H and kernel_W
        return result

    # For non-exact division, use strided approach with overlap
    else:
        # Calculate actual stride to fit exactly
        stride_H = (H - kernel_H) // (output_H - 1) if output_H > 1 else 1
        stride_W = (W - kernel_W) // (output_W - 1) if output_W > 1 else 1

        # Collect all averaged values
        values = []
        for i in range(output_H):
            row_values = []
            for j in range(output_W):
                h_start = i * stride_H
                h_end = min(h_start + kernel_H, H)
                w_start = j * stride_W
                w_end = min(w_start + kernel_W, W)

                # Extract region and average
                region = x[..., h_start:h_end, w_start:w_end, :]
                avg_val = mx.mean(region, axis=(-3, -2))  # Average over H and W
                row_values.append(avg_val)
            values.append(mx.stack(row_values, axis=-2))  # Stack along W dimension

        # Stack all rows along H dimension
        result = mx.stack(values, axis=-3)
        return result


class AdaptiveAvgPool3d(Module):
    r"""Applies 3-dimensional adaptive average pooling.

    The output size is D x H x W, for any input size. The number of output
    features is equal to the number of input planes.

    Args:
        output_size: the target output size of the form D x H x W.
            Can be a tuple (D, H, W) or a single int for a cube D x D x D.
            D, H and W can be either an ``int``, or ``None`` which means the size
            will be the same as that of the input.

    Examples:
        >>> import mlx.core as mx
        >>> import mlx.nn as nn
        >>> x = mx.random.normal(shape=(8, 16, 32, 32, 4))
        >>> pool = nn.AdaptiveAvgPool3d((5, 7, 9))
        >>> pool(x)
        >>> pool = nn.AdaptiveAvgPool3d(7)
        >>> pool(x)
    """

    def __init__(
        self,
        output_size: Union[int, Tuple[Optional[int], Optional[int], Optional[int]]],
    ):
        super().__init__()
        self.output_size = output_size

    def __call__(self, x):
        return adaptive_avg_pool3d(x, self.output_size)


def adaptive_avg_pool3d(
    x: mx.array,
    output_size: Union[int, Tuple[Optional[int], Optional[int], Optional[int]]],
) -> mx.array:
    r"""Apply 3-dimensional adaptive average pooling.

    Args:
        x: Input array of shape (N, D, H, W, C) or (D, H, W, C).
        output_size: Target output size (D, H, W) or single int for cube output.
            Values can be None to keep the corresponding input dimension.

    Returns:
        Output array with spatial dimensions matching output_size.
    """
    # Parse output_size
    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)
    elif len(output_size) == 1:
        output_size = (output_size[0], output_size[0], output_size[0])
    elif len(output_size) == 2:
        output_size = (output_size[0], output_size[1], output_size[1])

    # Get input dimensions
    *batch_dims, D, H, W, C = x.shape

    # Handle None values in output_size
    output_D = D if output_size[0] is None else output_size[0]
    output_H = H if output_size[1] is None else output_size[1]
    output_W = W if output_size[2] is None else output_size[2]

    # If already the right size, return as is
    if D == output_D and H == output_H and W == output_W:
        return x

    # Calculate kernel size and stride
    kernel_D = D // output_D
    kernel_H = H // output_H
    kernel_W = W // output_W

    # For exact division, use regular pooling
    if D % output_D == 0 and H % output_H == 0 and W % output_W == 0:
        # Reshape for pooling: (batch..., D, H, W, C) -> (batch..., output_D, kernel_D, output_H, kernel_H, output_W, kernel_W, C)
        new_shape = batch_dims + [
            output_D,
            kernel_D,
            output_H,
            kernel_H,
            output_W,
            kernel_W,
            C,
        ]
        x_reshaped = x.reshape(new_shape)

        # Average over kernel dimensions (kernel_D is at -6, kernel_H is at -4, kernel_W is at -2)
        result = mx.mean(
            x_reshaped, axis=[-6, -4, -2]
        )  # Average over kernel_D, kernel_H and kernel_W
        return result

    # For non-exact division, use strided approach with overlap
    else:
        # Calculate actual stride to fit exactly
        stride_D = (D - kernel_D) // (output_D - 1) if output_D > 1 else 1
        stride_H = (H - kernel_H) // (output_H - 1) if output_H > 1 else 1
        stride_W = (W - kernel_W) // (output_W - 1) if output_W > 1 else 1

        # Collect all averaged values
        values = []
        for i in range(output_D):
            depth_values = []
            for j in range(output_H):
                row_values = []
                for k in range(output_W):
                    d_start = i * stride_D
                    d_end = min(d_start + kernel_D, D)
                    h_start = j * stride_H
                    h_end = min(h_start + kernel_H, H)
                    w_start = k * stride_W
                    w_end = min(w_start + kernel_W, W)

                    # Extract region and average
                    region = x[..., d_start:d_end, h_start:h_end, w_start:w_end, :]
                    avg_val = mx.mean(
                        region, axis=(-4, -3, -2)
                    )  # Average over D, H and W
                    row_values.append(avg_val)
                depth_values.append(
                    mx.stack(row_values, axis=-2)
                )  # Stack along W dimension
            values.append(mx.stack(depth_values, axis=-3))  # Stack along H dimension

        # Stack all depths along D dimension
        result = mx.stack(values, axis=-4)
        return result
