# Copyright © 2024 Apple Inc.

import math
from functools import lru_cache
from typing import Optional

import mlx.core as mx
from mlx.nn.layers.base import Module


@lru_cache
def sum_gradients(group):
    if group.size() == 1:
        return lambda x: x

    @mx.custom_function
    def f(x):
        return x

    @f.vjp
    def f(x, dx, _):
        return mx.distributed.all_sum(dx, group=group)

    return f


class AllToShardedLinear(Module):
    """Each member of the group applies part of the affine transformation such
    that the result is sharded across the group.

    The gradients are automatically aggregated from each member of the group.

    Args:
        input_dims (int): The dimensionality of the input features
        output_dims (int): The dimensionality of the output features
        bias (bool, optional): If set to ``False`` the the layer will not use a
            bias. Default is ``True``.
        group (mx.distributed.Group, optional): The sharding will happen across
            this group. If not set then the global group is used. Default is
            ``None``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group: Optional[mx.distributed.Group] = None,
    ):
        super().__init__()

        # Initialize the parameters
        scale = math.sqrt(1.0 / input_dims)
        self.group = group or mx.distributed.init()
        N = self.group.size()

        if (output_dims % N) != 0:
            raise ValueError(
                f"Cannot shard the output of size {output_dims} across {N} devices."
            )

        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims // N, input_dims),
        )
        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(output_dims // N,),
            )

    def _extra_repr(self) -> str:
        out_dims, in_dims = self.weight.shape
        N = self.group.size()
        out_dims *= N
        return f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self}"

    def __call__(self, x: mx.array) -> mx.array:
        # Aggregate the gradients coming from each shard
        if self.group.size() > 1:
            x = sum_gradients(self.group)(x)

        # Compute the affine projection
        if "bias" in self:
            x = mx.addmm(self["bias"], x, self["weight"].T)
        else:
            x = x @ self["weight"].T
        return x

    @classmethod
    def from_linear(
        cls, linear_layer: Module, group: Optional[mx.distributed.Group] = None
    ):
        group = group or mx.distributed.init()
        N = group.size()
        r = group.rank()
        output_dims, input_dims = linear_layer.weight.shape
        step = output_dims // N

        sl = cls(input_dims, output_dims, False, group)
        # The multiplication with 1.0 forces a copy, perhaps change to
        # something better when available.
        sl.weight = linear_layer.weight[r * step : (r + 1) * step] * 1
        if "bias" in linear_layer:
            sl.bias = linear_layer.bias[r * step : (r + 1) * step] * 1

        return sl


class ShardedToAllLinear(Module):
    """Each member of the group applies part of the affine transformation and
    then aggregates the results.

    All nodes will have the same exact result after this layer.

    :class:`ShardedToAllLinear` provides a classmethod :meth:`from_linear` to
    convert linear layers to sharded :obj:`ShardedToAllLinear` layers.

    Args:
        input_dims (int): The dimensionality of the input features
        output_dims (int): The dimensionality of the output features
        bias (bool, optional): If set to ``False`` the the layer will not use a
            bias. Default is ``True``.
        group (mx.distributed.Group, optional): The sharding will happen across
            this group. If not set then the global group is used. Default is
            ``None``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group: Optional[mx.distributed.Group] = None,
    ):
        super().__init__()

        # Initialize the parameters
        scale = math.sqrt(1.0 / input_dims)
        self.group = group or mx.distributed.init()
        N = self.group.size()

        if (input_dims % N) != 0:
            raise ValueError(
                f"The input of size {input_dims} cannot be sharded across {N} devices."
            )

        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims // N),
        )
        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(output_dims,),
            )

    def _extra_repr(self) -> str:
        N = self.group.size()
        out_dims, in_dims = self.weight.shape
        in_dims *= N
        return f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self}"

    def __call__(self, x: mx.array) -> mx.array:
        if self.group.size() > 1:
            # Perform the local projection and aggregate the results
            x = x @ self["weight"].T
            x = mx.distributed.all_sum(x, group=self.group)

            # Add the bias if we have one
            if "bias" in self:
                x = x + self["bias"]
        else:
            # Normal linear layer as we are not in a distributed setting.
            if "bias" in self:
                x = mx.addmm(self["bias"], x, self["weight"].T)
            else:
                x = x @ self["weight"].T
        return x

    @classmethod
    def from_linear(
        cls, linear_layer: Module, group: Optional[mx.distributed.Group] = None
    ):
        group = group or mx.distributed.init()
        N = group.size()
        r = group.rank()
        output_dims, input_dims = linear_layer.weight.shape
        step = input_dims // N

        sl = cls(input_dims, output_dims, False, group)
        # The multiplication with 1.0 forces a copy, perhaps change to
        # something better when available.
        sl.weight = linear_layer.weight[:, r * step : (r + 1) * step] * 1
        if "bias" in linear_layer:
            sl.bias = linear_layer.bias

        return sl


class QuantizedAllToShardedLinear(Module):
    """Each member of the group applies part of the affine transformation with
    a quantized matrix such that the result is sharded across the group.

    It is the quantized equivalent of :class:`mlx.nn.AllToShardedLinear`.
    Similar to :class:`mlx.nn.QuantizedLinear` its parameters are frozen and
    will not be included in any gradient computation.

    Args:
        input_dims (int): The dimensionality of the input features.
        output_dims (int): The dimensionality of the output features.
        bias (bool, optional): If set to ``False`` then the layer will not use
            a bias. Default: ``True``.
        group_size (int, optional): The group size to use for the quantized
            weight. See :func:`~mlx.core.quantize`. Default: ``64``.
        bits (int, optional): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. Default: ``4``.
        group (mx.distributed.Group, optional): The sharding will happen across
            this group. If not set then the global group is used. Default is
            ``None``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        group: Optional[mx.distributed.Group] = None,
    ):
        super().__init__()

        # Quantization config
        self.group_size = group_size
        self.bits = bits

        # Initialize the quantized weight
        scale = math.sqrt(1.0 / input_dims)
        self.group = group or mx.distributed.init()
        N = self.group.size()

        if (output_dims % N) != 0:
            raise ValueError(
                f"Cannot shard the output of size {output_dims} across {N} devices."
            )

        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims // N, input_dims),
        )
        self.weight, self.scales, self.biases = mx.quantize(weight, group_size, bits)

        # And bias if needed
        if bias:
            self.bias = mx.zeros((output_dims // N,))

        # Freeze this model's parameters
        self.freeze()

    def unfreeze(self, *args, **kwargs):
        """Wrap unfreeze so that we unfreeze any layers we might contain but
        our parameters will remain frozen."""
        super().unfreeze(*args, **kwargs)
        self.freeze(recurse=False)

    def _extra_repr(self) -> str:
        out_dims, in_dims = self.weight.shape
        in_dims *= 32 // self.bits
        out_dims *= self.group.size()
        return (
            f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self}, "
            f"group_size={self.group_size}, bits={self.bits}"
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Aggregate the gradients coming from each shard
        if self.group.size() > 1:
            x = sum_gradients(self.group)(x)

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
    def from_quantized_linear(
        cls,
        quantized_linear_layer: Module,
        group: Optional[mx.distributed.Group] = None,
    ):
        group = group or mx.distributed.init()
        N = group.size()
        r = group.rank()
        output_dims, input_dims = quantized_linear_layer.weight.shape
        input_dims *= 32 // quantized_linear_layer.bits
        step = output_dims // N

        sl = cls(
            input_dims,
            output_dims,
            False,
            group_size=quantized_linear_layer.group_size,
            bits=quantized_linear_layer.bits,
            group=group,
        )
        sl.weight = quantized_linear_layer.weight[r * step : (r + 1) * step] * 1
        sl.scales = quantized_linear_layer.scales[r * step : (r + 1) * step] * 1
        sl.biases = quantized_linear_layer.biases[r * step : (r + 1) * step] * 1
        if "bias" in quantized_linear_layer:
            sl.bias = quantized_linear_layer.bias[r * step : (r + 1) * step] * 1

        return sl


class QuantizedShardedToAllLinear(Module):
    """Each member of the group applies part of the affine transformation using
    the quantized matrix and then aggregates the results.

    All nodes will have the same exact result after this layer.

    It is the quantized equivalent of :class:`mlx.nn.ShardedToAllLinear`.
    Similar to :class:`mlx.nn.QuantizedLinear` its parameters are frozen and
    will not be included in any gradient computation.

    Args:
        input_dims (int): The dimensionality of the input features.
        output_dims (int): The dimensionality of the output features.
        bias (bool, optional): If set to ``False`` then the layer will not use
            a bias. Default: ``True``.
        group_size (int, optional): The group size to use for the quantized
            weight. See :func:`~mlx.core.quantize`. Default: ``64``.
        bits (int, optional): The bit width to use for the quantized weight.
            See :func:`~mlx.core.quantize`. Default: ``4``.
        group (mx.distributed.Group, optional): The sharding will happen across
            this group. If not set then the global group is used. Default is
            ``None``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        group: Optional[mx.distributed.Group] = None,
    ):
        super().__init__()

        # Quantization config
        self.group_size = group_size
        self.bits = bits

        # Initialize the quantized weight
        scale = math.sqrt(1.0 / input_dims)
        self.group = group or mx.distributed.init()
        N = self.group.size()

        if (input_dims % N) != 0:
            raise ValueError(
                f"The input of size {input_dims} cannot be sharded across {N} devices."
            )

        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, input_dims // N),
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

    def _extra_repr(self) -> str:
        out_dims, in_dims = self.weight.shape
        in_dims *= (32 // self.bits) * self.group.size()
        return (
            f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self}, "
            f"group_size={self.group_size}, bits={self.bits}"
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.quantized_matmul(
            x,
            self["weight"],
            scales=self["scales"],
            biases=self["biases"],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if self.group.size() > 1:
            x = mx.distributed.all_sum(x, group=self.group)
        if "bias" in self:
            x = x + self["bias"]
        return x

    @classmethod
    def from_quantized_linear(
        cls,
        quantized_linear_layer: Module,
        group: Optional[mx.distributed.Group] = None,
    ):
        group = group or mx.distributed.init()
        N = group.size()
        r = group.rank()
        output_dims, input_dims = quantized_linear_layer.weight.shape
        step = input_dims // N
        step_grouped = quantized_linear_layer.scales.shape[1] // N
        input_dims *= (32 // quantized_linear_layer.bits) * N

        sl = cls(
            input_dims,
            output_dims,
            False,
            group_size=quantized_linear_layer.group_size,
            bits=quantized_linear_layer.bits,
            group=group,
        )
        sl.weight = quantized_linear_layer.weight[:, r * step : (r + 1) * step] * 1
        sl.scales = (
            quantized_linear_layer.scales[:, r * step_grouped : (r + 1) * step_grouped]
            * 1
        )
        sl.biases = (
            quantized_linear_layer.biases[:, r * step_grouped : (r + 1) * step_grouped]
            * 1
        )
        if "bias" in quantized_linear_layer:
            sl.bias = quantized_linear_layer.bias

        return sl
