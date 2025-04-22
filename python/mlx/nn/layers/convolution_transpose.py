# Copyright © 2023 Apple Inc.

import math
from typing import Union

import mlx.core as mx
from mlx.nn.layers.base import Module


class ConvTranspose1d(Module):
    """Applies a 1-dimensional transposed convolution over the multi-channel input sequence.

    The channels are expected to be last i.e. the input shape should be ``NLC`` where:

    * ``N`` is the batch dimension
    * ``L`` is the sequence length
    * ``C`` is the number of input channels

    Args:
        in_channels (int): The number of input channels
        out_channels (int): The number of output channels
        kernel_size (int): The size of the convolution filters
        stride (int, optional): The stride when applying the filter.
            Default: ``1``.
        padding (int, optional): How many positions to 0-pad the input with.
            Default: ``0``.
        dilation (int, optional): The dilation of the convolution.
        output_padding(int, optional): Additional size added to one side of the
            output shape. Default: ``0``.
        bias (bool, optional): If ``True`` add a learnable bias to the output.
            Default: ``True``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        output_padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()

        scale = math.sqrt(1 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))

        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.output_padding = output_padding

    def _extra_repr(self):
        return (
            f"{self.weight.shape[-1]}, {self.weight.shape[0]}, "
            f"kernel_size={self.weight.shape[1]}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"output_padding={self.output_padding}, "
            f"bias={'bias' in self}"
        )

    def __call__(self, x):
        y = mx.conv_transpose1d(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
        )
        if "bias" in self:
            y = y + self.bias
        return y


class ConvTranspose2d(Module):
    """Applies a 2-dimensional transposed convolution over the multi-channel input image.

    The channels are expected to be last i.e. the input shape should be ``NHWC`` where:

    * ``N`` is the batch dimension
    * ``H`` is the input image height
    * ``W`` is the input image width
    * ``C`` is the number of input channels

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int or tuple): The size of the convolution filters.
        stride (int or tuple, optional): The size of the stride when
            applying the filter. Default: ``1``.
        padding (int or tuple, optional): How many positions to 0-pad
            the input with. Default: ``0``.
        dilation (int or tuple, optional): The dilation of the convolution.
        output_padding(int or tuple, optional): Additional size added to one
            side of the output shape. Default: ``0``.
        bias (bool, optional): If ``True`` add a learnable bias to the
            output. Default: ``True``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        output_padding: Union[int, tuple] = 0,
        bias: bool = True,
    ):
        super().__init__()

        kernel_size, stride, padding, output_padding = map(
            lambda x: (x, x) if isinstance(x, int) else x,
            (kernel_size, stride, padding, output_padding),
        )
        scale = math.sqrt(1 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, *kernel_size, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))

        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.output_padding = output_padding

    def _extra_repr(self):
        return (
            f"{self.weight.shape[-1]}, {self.weight.shape[0]}, "
            f"kernel_size={self.weight.shape[1:2]}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"output_padding={self.output_padding}, "
            f"bias={'bias' in self}"
        )

    def __call__(self, x):
        y = mx.conv_transpose2d(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
        )
        if "bias" in self:
            y = y + self.bias
        return y


class ConvTranspose3d(Module):
    """Applies a 3-dimensional transposed convolution over the multi-channel input image.

    The channels are expected to be last i.e. the input shape should be ``NDHWC`` where:

    * ``N`` is the batch dimension
    * ``D`` is the input image depth
    * ``H`` is the input image height
    * ``W`` is the input image width
    * ``C`` is the number of input channels

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int or tuple): The size of the convolution filters.
        stride (int or tuple, optional): The size of the stride when
            applying the filter. Default: ``1``.
        padding (int or tuple, optional): How many positions to 0-pad
            the input with. Default: ``0``.
        dilation (int or tuple, optional): The dilation of the convolution.
        output_padding(int or tuple, optional): Additional size added to one
            side of the output shape. Default: ``0``.
        bias (bool, optional): If ``True`` add a learnable bias to the
            output. Default: ``True``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        output_padding: Union[int, tuple] = 0,
        bias: bool = True,
    ):
        super().__init__()

        kernel_size, stride, padding, output_padding = map(
            lambda x: (x, x, x) if isinstance(x, int) else x,
            (kernel_size, stride, padding, output_padding),
        )
        scale = math.sqrt(
            1 / (in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2])
        )
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, *kernel_size, in_channels),
        )
        if bias:
            self.bias = mx.zeros((out_channels,))

        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.output_padding = output_padding

    def _extra_repr(self):
        return (
            f"{self.weight.shape[-1]}, {self.weight.shape[0]}, "
            f"kernel_size={self.weight.shape[1:3]}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"output_padding={self.output_padding}, "
            f"bias={'bias' in self}"
        )

    def __call__(self, x):
        y = mx.conv_transpose3d(
            x,
            self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
        )
        if "bias" in self:
            y = y + self.bias
        return y
