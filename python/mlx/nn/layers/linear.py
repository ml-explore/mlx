# Copyright © 2023 Apple Inc.

import math
from typing import Any

import mlx.core as mx
from mlx.nn.layers.base import Module


class Identity(Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def __call__(self, x: mx.array) -> mx.array:
        return x


class Linear(Module):
    r"""Applies an affine transformation to the input.

    Concretely:

    .. math::

        y = x W^\top + b

    where:
    where :math:`W` has shape ``[output_dims, input_dims]`` and :math:`b` has shape ``[output_dims]``.

    The values are initialized from the uniform distribution :math:`\mathcal{U}(-{k}, {k})`,
    where :math:`k = \frac{1}{\sqrt{D_i}}` and :math:`D_i` is equal to ``input_dims``.

    Args:
        input_dims (int): The dimensionality of the input features
        output_dims (int): The dimensionality of the output features
        bias (bool, optional): If set to ``False`` then the layer will
          not use a bias. Default is ``True``.
    """

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims),
        )
        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(output_dims,),
            )

    def _extra_repr(self) -> str:
        return f"input_dims={self.weight.shape[1]}, output_dims={self.weight.shape[0]}, bias={'bias' in self}"

    def __call__(self, x: mx.array) -> mx.array:
        x = x @ self.weight.T
        if "bias" in self:
            x = x + self.bias
        return x


class Bilinear(Module):
    r"""Applies a bilinear transformation to the inputs.

    Concretely:

    .. math::

        y_i = x_1^\top W_i x_2 + b_i

    where:
    :math:`W` has shape ``[output_dims, input1_dims, input2_dims]``, :math:`b` has shape ``[output_dims ]``,
    and :math:`i` indexes the output dimension.

    The values are initialized from the uniform distribution :math:`\mathcal{U}(-{k}, {k})`,
    where :math:`k = \frac{1}{\sqrt{D_1}}` and :math:`D_1` is ``input1_dims``.

    Args:
        input1_dims (int): The dimensionality of the input1 features
        input2_dims (int): The dimensionality of the input2 features
        output_dims (int): The dimensionality of the output features
        bias (bool, optional): If set to ``False`` then the layer will
          not use a bias. Default is ``True``.
    """

    def __init__(
        self, input1_dims: int, input2_dims: int, output_dims: int, bias: bool = True
    ) -> None:
        super().__init__()
        scale = math.sqrt(1.0 / input1_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(1, output_dims, input1_dims, input2_dims),
        )
        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(output_dims,),
            )

    def _extra_repr(self) -> str:
        return (
            f"input1_dims={self.weight.shape[2]}, input2_dims={self.weight.shape[3]}, "
            f"output_dims={self.weight.shape[1]}, bias={'bias' in self}"
        )

    def __call__(self, x1: mx.array, x2: mx.array) -> mx.array:
        x1 = mx.expand_dims(x1, axis=(-2, -3))
        x2 = mx.expand_dims(x2, axis=(-2))
        y = mx.squeeze(x1 @ self.weight, -2).swapaxes(-1, -2)
        y = mx.squeeze(x2 @ y, -2)
        if "bias" in self:
            y = y + self.bias
        return y
