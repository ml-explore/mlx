# Copyright © 2023 Apple Inc.

import math
from typing import Any, Tuple

import mlx.core as mx
from mlx.nn.layers.base import Module


class Linear(Module):
    """Applies an affine transformation to the input.

    Args:
        input_dims (int): The dimensionality of the input features
        output_dims (int): The dimensionality of the output features
        bias (bool): If set to False then the layer will not use a bias
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


def absmax_quantize(x: mx.array) -> Tuple[mx.array, mx.array]:
    """Quantize the input to int8 using absmax method.
    Returns:
        (mx.array, mx.array): (quantized x, scale)
    """
    assert x.dtype in (
        mx.float32,
        mx.float16,
        mx.bfloat16,
    ), f"Unsupported weight for quantization dtype: {x.dtype}"
    scale = x.abs().max(axis=1) / 127.0
    new_x = (x / scale).astype(mx.int8)
    return (
        new_x,
        scale,
    )


class AbsmaxQuantizedLinear(Module):
    """Applies an affine transformation to the input.

    Args:
        input_dims (int): The dimensionality of the input features
        output_dims (int): The dimensionality of the output features
        bias (bool): If set to False then the layer will not use a bias

    Note:
        This layer expects the weight to be a int8 mlx.array, utilized absmax_quantize
        to get the int8 weight and scale.
    """

    def __init__(self, input_dims: int, output_dims: int, bias: bool = True):
        super().__init__()
        self.weight = mx.ones(shape=(output_dims, input_dims), dtype=mx.int8)
        self.scale = mx.ones((output_dims, input_dims))
        if bias:
            self.bias = mx.zeros((output_dims,))

    def __setattr__(self, key: str, val: Any):
        if key == "weight" and isinstance(val, mx.array) and val.dtype != mx.int8:
            val, scale = absmax_quantize(val)
            self.scale = scale
        return super().__setattr__(key, val)

    def __call__(self, x: mx.array) -> mx.array:
        x = x @ (self.weight.T * self.scale)
        if "bias" in self:
            x = x + self.bias
        return x
