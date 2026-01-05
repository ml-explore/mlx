# Copyright © 2023-2024 Apple Inc.

import math
from typing import Callable, Optional, Union

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.utils import tree_map_with_path


def _defaults_for_mode(mode, group_size, bits):
    mode_defaults = {
        "affine": (64, 4),
        "mxfp4": (32, 4),
        "nvfp4": (16, 4),
        "mxfp8": (32, 8),
    }
    default_group_size, default_bits = mode_defaults[mode]
    return group_size or default_group_size, bits or default_bits


def quantize(
    model: Module,
    group_size: int = None,
    bits: int = None,
    *,
    mode: str = "affine",
    class_predicate: Optional[Callable[[str, Module], Union[bool, dict]]] = None,
):
    """Quantize the sub-modules of a module according to a predicate.

    By default all layers that define a ``to_quantized(group_size, bits)``
    method will be quantized. Both :obj:`Linear` and :obj:`Embedding` layers
    will be quantized. Note also, the module is updated in-place.

    Args:
        model (mlx.nn.Module): The model whose leaf modules may be quantized.
        group_size (Optional[int]): The quantization group size (see
           :func:`mlx.core.quantize`). Default: ``None``.
        bits (Optional[int]): The number of bits per parameter (see
           :func:`mlx.core.quantize`). Default: ``None``.
        mode (str): The quantization method to use (see
           :func:`mlx.core.quantize`). Default: ``"affine"``.
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
                    return m.to_quantized(group_size=group_size, bits=bits, mode=mode)
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
        group_size (Optional[int]): The group size to use for the quantized
            weight. See :func:`~mlx.core.quantize`. Default: ``None``.
        bits (Optional[int]): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. Default: ``None``.
        mode (str): The quantization method to use (see
           :func:`mlx.core.quantize`). Default: ``"affine"``.
    """

    def __init__(
        self,
        num_embeddings: int,
        dims: int,
        group_size: int = None,
        bits: int = None,
        mode: str = "affine",
    ):
        super().__init__()

        # Quantization config
        self.group_size, self.bits = _defaults_for_mode(mode, group_size, bits)
        self.mode = mode

        # Initialize the quantized weight
        scale = math.sqrt(1 / dims)
        weight = mx.random.normal(shape=(num_embeddings, dims), scale=scale)
        self.weight, self.scales, *biases = mx.quantize(
            weight, group_size, bits, mode=mode
        )
        self.biases = biases[0] if biases else None
        self.num_embeddings = num_embeddings
        self.dims = dims

        # Freeze this model's parameters
        self.freeze()

    def __call__(self, x):
        biases = self.get("biases")
        return mx.dequantize(
            self["weight"][x],
            scales=self["scales"][x],
            biases=biases[x] if biases is not None else None,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
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
            biases=self.get("biases"),
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

    def _extra_repr(self):
        return (
            f"{self.num_embeddings}, {self.dims}, "
            f"group_size={self.group_size}, bits={self.bits}, mode={self.mode}"
        )

    @classmethod
    def from_embedding(
        cls,
        embedding_layer: Module,
        group_size: int = None,
        bits: int = None,
        mode: str = "affine",
    ):
        """Create a :obj:`QuantizedEmbedding` layer from an :obj:`Embedding` layer."""
        embedding_dims, dims = embedding_layer.weight.shape
        ql = cls(embedding_dims, dims, group_size, bits, mode=mode)
        ql.weight, ql.scales, *biases = mx.quantize(
            embedding_layer.weight,
            group_size,
            bits,
            mode=mode,
        )
        ql.biases = biases[0] if biases else None
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
        group_size (Optional[int]): The group size to use for the quantized
            weight. See :func:`~mlx.core.quantize`. Default: ``None``.
        bits (Optional[int]): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. Default: ``None``.
        mode (str): The quantization method to use (see
           :func:`mlx.core.quantize`). Default: ``"affine"``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = None,
        bits: int = None,
        mode: str = "affine",
    ):
        super().__init__()

        # Quantization config
        self.group_size, self.bits = _defaults_for_mode(mode, group_size, bits)
        self.mode = mode

        # Initialize the quantized weight
        scale = math.sqrt(1 / input_dims)
        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims),
        )
        self.weight, self.scales, *biases = mx.quantize(
            weight, group_size, bits, mode=mode
        )
        self.biases = biases[0] if biases else None

        # And bias if needed
        if bias:
            self.bias = mx.zeros((output_dims,))

        # Freeze this model's parameters
        self.freeze()

    def _extra_repr(self):
        out_dims, in_dims = self.weight.shape
        in_dims *= 32 // self.bits
        return (
            f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self}, "
            f"group_size={self.group_size}, bits={self.bits}, mode={self.mode}"
        )

    def __call__(self, x):
        x = mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self.get("biases"),
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if "bias" in self:
            x = x + self["bias"]
        return x

    @classmethod
    def from_linear(
        cls,
        linear_layer: Module,
        group_size: int = None,
        bits: int = None,
        mode: str = "affine",
    ):
        """Create a :obj:`QuantizedLinear` layer from a :obj:`Linear` layer."""
        output_dims, input_dims = linear_layer.weight.shape
        ql = cls(input_dims, output_dims, False, group_size, bits, mode=mode)
        ql.weight, ql.scales, *biases = mx.quantize(
            linear_layer.weight,
            group_size,
            bits,
            mode=mode,
        )
        ql.biases = biases[0] if biases else None

        if "bias" in linear_layer:
            ql.bias = linear_layer.bias

        return ql


class QQLinear(Module):
    """Quantizes the input and applies an affine transformation using quantized weights.

    Two use cases are supported:

    1) **Eval**:  The weights are frozen and stored in quantized form together with
       their scales (``self.weight`` is quantized and ``self.scales`` is provided).
    2) **Train**: The weights are stored in higher precision and are quantized on
         the fly during computation so that gradients with respect to the weights
         can be computed.

    To switch between the two cases, use ``layer.eval()`` and ``layer.train()`` respectively.

    Compared to the :class:`mlx.nn.QuantizedLinear` layer, this layer
    quantizes the input as well and includes weights in gradient computations.

    :obj:`QQLinear` also provides:
     -  the class method :meth:`from_linear` to convert :class:`mlx.nn.Linear`
     layers to :obj:`QQLinear` layers.

    Note: This layer does not support a bias term yet.

    Args:
        input_dims (int): The dimensionality of the input features.
        output_dims (int): The dimensionality of the output features.
        group_size (Optional[int]): The group size to use for the quantized weight.
            See :func:`~mlx.core.quantize`. Default: ``None``.
        bits (Optional[int]): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. Default: ``None``.
        mode (Optional[str]): The quantization method to use (see
            :func:`mlx.core.quantize`). Currently, only ``"nvfp4"`` and ``"mxfp8"``
            are supported. Default: ``"nvfp4"``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        group_size: int = None,
        bits: int = None,
        mode: str = "nvfp4",
    ):
        super().__init__()

        # Quantization config
        self.group_size, self.bits = _defaults_for_mode(mode, group_size, bits)
        self.mode = mode

        scale = math.sqrt(1 / input_dims)
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims),
        )
        self._quantized = False

    def _extra_repr(self):
        out_dims, in_dims = self.weight.shape
        if self.weight.dtype == mx.uint32:
            in_dims *= 32 // self.bits
        return (
            f"input_dims={in_dims}, output_dims={out_dims}, "
            f"group_size={self.group_size}, bits={self.bits}, mode={self.mode}"
        )

    def quantize(self):
        if not self._quantized:
            self.weight, self.scales = mx.quantize(
                self.weight,
                self.group_size,
                self.bits,
                mode=self.mode,
            )
            self._quantized = True

    def dequantize(self):
        if self._quantized:
            self.weight = mx.dequantize(
                self.weight,
                scales=self.scales,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
            self.__delattr__("scales")
            self._quantized = False

    def _set_training_mode(self, mode: bool):
        super()._set_training_mode(mode)

        if self._training:
            self.dequantize()
        else:
            self.quantize()

    def __call__(self, x):
        x = mx.qqmm(
            x,
            self["weight"],
            scales=self.get("scales"),
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        return x

    @classmethod
    def from_linear(
        cls,
        linear_layer: Module,
        group_size: int = None,
        bits: int = None,
        mode: str = "nvfp4",
    ):
        """Create a :obj:`QQLinear` layer from a :obj:`Linear` layer."""
        output_dims, input_dims = linear_layer.weight.shape  # (N,K)
        if linear_layer.get("bias") is not None:
            raise NotImplementedError("QQLinear does not support bias yet.")
        ql = cls(input_dims, output_dims, group_size, bits, mode=mode)
        ql.weight = linear_layer.weight
        ql.train(linear_layer.training)

        return ql
