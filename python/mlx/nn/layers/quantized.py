# Copyright © 2023-2024 Apple Inc.

import math
from typing import Callable, Optional, Union

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.utils import tree_map_with_path


def quantize(
    model: Module,
    group_size: int = 64,
    bits: int = 4,
    class_predicate: Optional[Callable[[str, Module], Union[bool, dict]]] = None,
):
    """Quantize the sub-modules of a module according to a predicate.

    By default all layers that define a ``to_quantized(group_size, bits)``
    method will be quantized. Both :obj:`Linear` and :obj:`Embedding` layers
    will be quantized. Note also, the module is updated in-place.

    Args:
        model (mlx.nn.Module): The model whose leaf modules may be quantized.
        group_size (int): The quantization group size (see
           :func:`mlx.core.quantize`). Default: ``64``.
        bits (int): The number of bits per parameter (see
           :func:`mlx.core.quantize`). Default: ``4``.
        class_predicate (Optional[Callable]): A callable which receives the
          :obj:`Module` path and :obj:`Module` itself and returns ``True`` or a
          dict of params for `to_quantized` if it should be quantized and
          ``False`` otherwise. If ``None``, then all layers that define a
          ``to_quantized(group_size, bits)`` method are quantized.
          Default: ``None``.
    """
    class_predicate = class_predicate or (lambda _, m: hasattr(m, "to_quantized"))

    def _maybe_quantize(path, m):
        if bool_or_params := class_predicate(path, m):
            if hasattr(m, "to_quantized"):
                if isinstance(bool_or_params, bool):
                    return m.to_quantized(group_size=group_size, bits=bits)
                elif isinstance(bool_or_params, dict):
                    return m.to_quantized(**bool_or_params)
                else:
                    raise ValueError(
                        "``class_predicate`` must return a bool"
                        " or a dict of parameters to pass to ``to_quantized``"
                    )
            else:
                raise ValueError(f"Unable to quantize model of type {type(m)}")
        else:
            return m

    leaves = model.leaf_modules()
    leaves = tree_map_with_path(_maybe_quantize, leaves, is_leaf=Module.is_module)
    model.update_modules(leaves)


class QuantizedEmbedding(Module):
    """The same as :obj:`Embedding` but with a  quantized weight matrix.

    :obj:`QuantizedEmbedding` also provides a :meth:`from_embedding`
    classmethod to convert embedding layers to :obj:`QuantizedEmbedding`
    layers.

    Args:
        num_embeddings (int): How many possible discrete tokens can we embed.
           Usually called the vocabulary size.
        dims (int): The dimensionality of the embeddings.
        group_size (int, optional): The group size to use for the quantized
            weight. See :func:`~mlx.core.quantize`. Default: ``64``.
        bits (int, optional): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. Default: ``4``.
    """

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        group_size: int = 64,
        bits: int = 4,
    ):
        super().__init__()

        # Quantization config
        self.group_size = group_size
        self.bits = bits

        # Initialize the quantized weight
        scale = math.sqrt(1 / dims)
        weight = mx.random.normal(shape=(num_embeddings, dims), scale=scale)
        self.weight, self.scales, self.biases = mx.quantize(weight, group_size, bits)
        self.num_embeddings = num_embeddings
        self.dims = dims

        # Freeze this model's parameters
        self.freeze()

    def __call__(self, x):
        return mx.dequantize(
            self["weight"][x],
            scales=self["scales"][x],
            biases=self["biases"][x],
            group_size=self.group_size,
            bits=self.bits,
        )

    def as_linear(self, x):
        """
        Call the quantized embedding layer as a quantized linear layer.

        Use this for example when input embedding and output projection
        weights are tied.
        """
        return mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self["biases"],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )

    def _extra_repr(self):
        return (
            f"{self.num_embeddings}, {self.dims}, "
            f"group_size={self.group_size}, bits={self.bits}"
        )

    @classmethod
    def from_embedding(
        cls, embedding_layer: Module, group_size: int = 64, bits: int = 4
    ):
        """Create a :obj:`QuantizedEmbedding` layer from an :obj:`Embedding` layer."""
        embedding_dims, dims = embedding_layer.weight.shape
        ql = cls(embedding_dims, dims, group_size, bits)
        ql.weight, ql.scales, ql.biases = mx.quantize(
            embedding_layer.weight, group_size, bits
        )
        return ql


class QuantizedLinear(Module):
    """Applies an affine transformation to the input using a quantized weight matrix.

    It is the quantized equivalent of :class:`mlx.nn.Linear`. For now its
    parameters are frozen and will not be included in any gradient computation
    but this will probably change in the future.

    :obj:`QuantizedLinear` also provides a classmethod :meth:`from_linear` to
    convert linear layers to :obj:`QuantizedLinear` layers.

    Args:
        input_dims (int): The dimensionality of the input features.
        output_dims (int): The dimensionality of the output features.
        bias (bool, optional): If set to ``False`` then the layer will not use
            a bias. Default: ``True``.
        group_size (int, optional): The group size to use for the quantized
            weight. See :func:`~mlx.core.quantize`. Default: ``64``.
        bits (int, optional): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. Default: ``4``.
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
            f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self}, "
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
        """Create a :obj:`QuantizedLinear` layer from a :obj:`Linear` layer."""
        output_dims, input_dims = linear_layer.weight.shape
        ql = cls(input_dims, output_dims, False, group_size, bits)
        ql.weight, ql.scales, ql.biases = mx.quantize(
            linear_layer.weight, group_size, bits
        )
        if "bias" in linear_layer:
            ql.bias = linear_layer.bias

        return ql
