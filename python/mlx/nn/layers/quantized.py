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
        bias (bool): If set to ``False`` then the layer will not use a bias.
            (default: True).
        groups (int): The group size to use for the quantized weight. See
            :func:`~mlx.core.quantize`. (default: 128)
        width (int): The bit width to use for the quantized weight. See
            :func:`~mlx.core.quantize`. (default: 4)
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        groups: int = 64,
        width: int = 4,
    ):
        super().__init__()

        # Quantization config
        self.groups = groups
        self.width = width

        # Initialize the quantized weight
        scale = math.sqrt(1 / input_dims)
        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims),
        )
        self.weight, self.scales, self.biases = mx.quantize(weight, groups, width)

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
        in_dims *= 32 // self.width
        return (
            f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self},"
            f"groups={self.groups}, width={self.width}"
        )

    def __call__(self, x):
        x = mx.quantized_matmul(
            x,
            self.weight.T,
            scales=self.scales,
            biases=self.biases,
            groups=self.groups,
            width=self.width,
        )
        if "bias" in self:
            x = x + self.bias
        return x

    @classmethod
    def from_linear(cls, linear_layer: Module, groups: int = 64, width: int = 4):
        """Create a QuantizedLinear layer from the parameters of a provided
        linear layer."""
        output_dims, input_dims = linear_layer.weight.shape
        ql = cls(input_dims, output_dims, False, groups, width)
        ql.weight, ql.scales, ql.biases = mx.quantize(
            linear_layer.weight, groups, width
        )
        if "bias" in linear_layer:
            ql.bias = linear_layer.bias

        return ql

    @classmethod
    def quantize_module(
        cls,
        model: Module,
        groups: int = 64,
        width: int = 4,
        linear_class_predicate=lambda m: isinstance(m, Linear),
    ):
        def _quantize_if_linear(m):
            if linear_class_predicate(m):
                return cls.from_linear(m, groups, width)
            else:
                return m

        leaves = model.leaf_modules()
        leaves = tree_map(_quantize_if_linear, leaves, is_leaf=Module.is_module)
        model.update_modules(leaves)
