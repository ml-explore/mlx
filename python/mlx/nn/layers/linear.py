# Copyright © 2023 Apple Inc.

import math
from typing import Any

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.nn.layers.quantized import QuantizedLinear


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
        if "bias" in self:
            x = mx.addmm(self["bias"], x, self["weight"].T)
        else:
            x = x @ self["weight"].T
        return x

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        """Return a :obj:`QuantizedLinear` layer that approximates this layer."""
        return QuantizedLinear.from_linear(self, group_size, bits)


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
            shape=(output_dims, input2_dims, input1_dims),
        )
        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(output_dims,),
            )

    def _extra_repr(self) -> str:
        out, in2, in1 = self.weight.shape
        return (
            f"input1_dims={in1}, input2_dims={in2}, output_dims={out}, "
            f"bias={'bias' in self}"
        )

    def __call__(self, x1: mx.array, x2: mx.array) -> mx.array:
        # Normalize shapes
        out, in2, in1 = self.weight.shape
        xshape = x1.shape[:-1]
        x1 = x1.reshape(-1, in1)
        x2 = x2.reshape(-1, 1, in2)

        # Perform the bilinear transformation
        w = self.weight.reshape(out * in2, in1)
        y = x1 @ w.T
        y = y.reshape(-1, out, in2).swapaxes(-2, -1)
        y = x2 @ y
        y = y.squeeze(1)

        # Reset the shape
        y = y.reshape(*xshape, out)

        # Apply the bias
        if "bias" in self:
            y = y + self.bias

        return y
