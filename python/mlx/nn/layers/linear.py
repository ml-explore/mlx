# Copyright © 2023 Apple Inc.

import math

import mlx.core as mx
from mlx.nn.layers.base import Module


class Identity(Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __call__(self, input):
        return input


class Linear(Module):
    r"""Applies an affine transformation to the input.

    Concretely:

    .. math::

        y = W^\top x + b

    where :math:`W` has shape ``[output_dims, input_dims]``.

    Args:
        input_dims (int): The dimensionality of the input features
        output_dims (int): The dimensionality of the output features
        bias (bool, optional): If set to ``False`` then the layer will
          not use a bias. Default ``True``.
    """

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.weight = mx.zeros((output_dims, input_dims))
        if bias:
            self.bias = mx.zeros((output_dims,))

        self.reset_parameters()

    def reset_parameters(self):
        scale = math.sqrt(1.0 / self.input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(self.output_dims, self.input_dims),
        )
        if "bias" in self:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(self.output_dims,),
            )

    def _extra_repr(self):
        return f"input_dims={self.input_dims}, output_dims={self.output_dims}, bias={'bias' in self}"

    def __call__(self, input):
        output = input @ self.weight.T
        if "bias" in self:
            output = output + self.bias
        return output


class Bilinear(Module):
    r"""Applies a bilinear transformation to the input.

    Concretely:

    .. math::

        y = input1^\top W input2 + b

    where :math:`W` has shape ``[output_dims, input1_dims, input2_dims]``.

    Args:
        input1_dims (int): The dimensionality of the input1 features
        input2_dims (int): The dimensionality of the input2 features
        output_dims (int): The dimensionality of the output features
        bias (bool, optional): If set to ``False`` then the layer will
          not use a bias. Default ``True``.
    """

    def __init__(
        self, input1_dims: int, input2_dims: int, output_dims: int, bias: bool = True
    ):
        super().__init__()
        self.input1_dims = input1_dims
        self.input2_dims = input2_dims
        self.output_dims = output_dims
        self.weight = mx.zeros((output_dims, input1_dims, input2_dims))
        if bias:
            self.bias = mx.zeros((output_dims,))

        self.reset_parameters()

    def reset_parameters(self):
        scale = math.sqrt(1.0 / self.input1_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(self.output_dims, self.input1_dims, self.input2_dims),
        )
        if "bias" in self:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(self.output_dims,),
            )

    def _extra_repr(self):
        return (
            f"input1_dims={self.input1_dims}, input2_dims={self.input2_dims}, output_dims={self.output_dims}, "
            f"bias={'bias' in self}"
        )

    def __call__(self, input1, input2):
        output = (input1 @ self.weight * input2.reshape(1, *input2.shape)).sum(-1).T
        if "bias" in self:
            output = output + self.bias
        return output
