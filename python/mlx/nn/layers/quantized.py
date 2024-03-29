# Copyright © 2023 Apple Inc.

import math

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear
from mlx.utils import tree_flatten, tree_map


class QuantizedLinear(Module):
    """Applies an affine transformation to the input using a quantized weight matrix.

    It is the quantized equivalent of :class:`mlx.nn.Linear`. For now its
    parameters are frozen and will not be included in any gradient computation
    but this will probably change in the future.

    QuantizedLinear also provides two useful classmethods to convert linear
    layers to QuantizedLinear layers.

    - :meth:`from_linear` returns a QuantizedLinear layer that applies the same
      linear transformation up to the quantization error.
    - :meth:`quantize_module` swaps all the linear layers of the passed module
      with QuantizedLinear ones.

    Args:
        input_dims (int): The dimensionality of the input features
        output_dims (int): The dimensionality of the output features
        bias (bool, optional): If set to ``False`` then the layer will not use
            a bias. (default: True).
        group_size (int, optional): The group size to use for the quantized
            weight. See :func:`~mlx.core.quantize`. (default: 64)
        bits (int, optional): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. (default: 4)
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
    ):
        super().__init__()

        # Quantization config
        self.group_size = group_size
        self.bits = bits

        # Initialize the quantized weight
        scale = math.sqrt(1 / input_dims)
        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims),
        )
        self.weight, self.scales, self.biases = mx.quantize(weight, group_size, bits)

        # And bias if needed
        if bias:
            self.bias = mx.zeros((output_dims,))

        # Freeze this model's parameters
        self.freeze()

    def unfreeze(self, *args, **kwargs):
        """Wrap unfreeze so that we unfreeze any layers we might contain but
        our parameters will remain frozen."""
        super().unfreeze(*args, **kwargs)
        self.freeze(recurse=False)

    def _extra_repr(self):
        out_dims, in_dims = self.weight.shape
        in_dims *= 32 // self.bits
        return (
            f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self},"
            f"group_size={self.group_size}, bits={self.bits}"
        )

    def __call__(self, x):
        x = mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self["biases"],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if "bias" in self:
            x = x + self["bias"]
        return x

    @classmethod
    def from_linear(cls, linear_layer: Module, group_size: int = 64, bits: int = 4):
        """Create a QuantizedLinear layer from the parameters of a provided
        linear layer."""
        output_dims, input_dims = linear_layer.weight.shape
        ql = cls(input_dims, output_dims, False, group_size, bits)
        ql.weight, ql.scales, ql.biases = mx.quantize(
            linear_layer.weight, group_size, bits
        )
        if "bias" in linear_layer:
            ql.bias = linear_layer.bias

        return ql

    @classmethod
    def quantize_module(
        cls,
        model: Module,
        group_size: int = 64,
        bits: int = 4,
        linear_class_predicate=lambda m: isinstance(m, Linear),
    ):
        def _quantize_if_linear(m):
            if linear_class_predicate(m):
                return cls.from_linear(m, group_size, bits)
            else:
                return m

        leaves = model.leaf_modules()
        leaves = tree_map(_quantize_if_linear, leaves, is_leaf=Module.is_module)
        model.update_modules(leaves)
