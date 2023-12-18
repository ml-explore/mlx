# Copyright © 2023 Apple Inc.

import math

import mlx.core as mx
from mlx.nn.layers.base import Module


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
        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims),
        )
        if bias:
            self.bias = mx.zeros((output_dims,))

    def _extra_repr(self):
        return f"input_dims={self.weight.shape[1]}, output_dims={self.weight.shape[0]}, bias={'bias' in self}"

    def __call__(self, x):
        x = x @ self.weight.T
        if "bias" in self:
            x = x + self.bias
        return x
