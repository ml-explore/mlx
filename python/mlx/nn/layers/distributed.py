# Copyright © 2024 Apple Inc.

import math
from functools import lru_cache
from typing import Callable, Optional, Union

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear
from mlx.nn.layers.quantized import QuantizedLinear
from mlx.utils import tree_map_with_path


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


def _split(weight, segments, axis):
    """Equivalent to mx.split but allows for fractional segments."""
    if isinstance(segments, int) or isinstance(segments[0], int):
        return mx.split(weight, segments, axis=axis)

    N = weight.shape[axis]
    indices = [int(s * N) for s in segments]
    return mx.split(weight, indices, axis=axis)


def _shard(
    parameters: dict,
    sharding_predicate: Callable,
    group: Optional[mx.distributed.Group] = None,
):
    """Returns a new parameter tree with the weights sharded according to the
    sharding_predicate.

    The sharding predicate should return the sharding axis and optionally also
    the segments that comprise the weight.
    """
    group = group or mx.distributed.init()
    N = group.size()
    r = group.rank()

    def _shard_fn(path, weight):
        if not isinstance(weight, mx.array):
            return weight

        s = sharding_predicate(path, weight)
        if s is None:
            return weight

        axis = None
        segments = 1
        if isinstance(s, int):
            axis = s
        elif isinstance(s, tuple):
            axis, segments = s
        else:
            raise ValueError(
                "The sharding function should return int or tuple[int, list]"
            )

        return mx.contiguous(
            mx.concatenate(
                [_split(part, N, axis)[r] for part in _split(weight, segments, axis)],
                axis=axis,
            )
        )

    return tree_map_with_path(_shard_fn, parameters)


def _all_to_sharded(segments):
    """Simple predicate to shard fully connected layers such that a common
    representation becomes a sharded representation."""

    def _shard_fn(path, weight):
        return max(weight.ndim - 2, 0), segments

    return _shard_fn


def _sharded_to_all(segments):
    """Simple predicate to shard fully connected layers such that a sharded
    representation becomes a common representation."""

    def _shard_fn(path, weight):
        if path.endswith("bias"):
            return None
        return -1, segments

    return _shard_fn


def _check_sharding(sharding):
    if sharding not in ("all-to-sharded", "sharded-to-all"):
        raise ValueError(
            (
                f"Sharding type {sharding=} not supported, "
                "choose one of 'all-to-sharded' or 'sharded-to-all'"
            )
        )


def shard_inplace(
    module: Module,
    sharding: Union[str, Callable],
    *,
    segments: Union[int, list] = 1,
    group: Optional[mx.distributed.Group] = None,
):
    """Shard a module in-place by updating its parameter dictionary with the
    sharded parameter dictionary.

    The ``sharding`` argument can be any callable that given the path and the
    weight returns the sharding axis and optionally also the segments that
    comprise the unsharded weight. For instance if the weight is a fused QKV
    matrix the segments should be 3.

    .. note::
        The module doesn't change so in order for distributed communication to
        happen the module needs to natively support it and for it to be enabled.

    Args:
        module (mlx.nn.Module): The parameters of this module will be sharded
            in-place.
        sharding (str or callable): One of "all-to-sharded" and
            "sharded-to-all" or a callable that returns the sharding axis and
            segments.
        segments (int or list): The segments to use if ``sharding`` is a
            string. Default: ``1``.
        group (mlx.core.distributed.Group): The distributed group to shard
            across. If not set, the global group will be used. Default: ``None``.
    """
    if isinstance(sharding, str):
        _check_sharding(sharding)
        sharding = (
            _all_to_sharded(segments)
            if sharding == "all-to-sharded"
            else _sharded_to_all(segments)
        )
    module.update(_shard(module.parameters(), sharding, group))


def shard_linear(
    module: Module,
    sharding: str,
    *,
    segments: Union[int, list] = 1,
    group: Optional[mx.distributed.Group] = None,
):
    """Create a new linear layer that has its parameters sharded and also
    performs distributed communication either in the forward or backward
    pass.

    .. note::
        Contrary to ``shard_inplace``, the original layer is not changed but a
        new layer is returned.

    Args:
        module (mlx.nn.Module): The linear layer to be sharded.
        sharding (str): One of "all-to-sharded" and
            "sharded-to-all" that defines the type of sharding to perform.
        segments (int or list): The segments to use. Default: ``1``.
        group (mlx.core.distributed.Group): The distributed group to shard
            across. If not set, the global group will be used. Default: ``None``.
    """
    _check_sharding(sharding)
    fns = {
        ("all-to-sharded", True): AllToShardedLinear.from_linear,
        ("all-to-sharded", False): QuantizedAllToShardedLinear.from_quantized_linear,
        ("sharded-to-all", True): ShardedToAllLinear.from_linear,
        ("sharded-to-all", False): QuantizedShardedToAllLinear.from_quantized_linear,
    }
    return fns[sharding, isinstance(module, Linear)](
        module, segments=segments, group=group
    )


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
        x = sum_gradients(self.group)(x)

        # Compute the affine projection
        if "bias" in self:
            x = mx.addmm(self["bias"], x, self["weight"].T)
        else:
            x = x @ self["weight"].T
        return x

    @classmethod
    def from_linear(
        cls,
        linear_layer: Module,
        *,
        segments: Union[int, list] = 1,
        group: Optional[mx.distributed.Group] = None,
    ):
        group = group or mx.distributed.init()
        output_dims, input_dims = linear_layer.weight.shape

        sl = cls(input_dims, output_dims, hasattr(linear_layer, "bias"), group)
        sl.update(_shard(linear_layer.parameters(), _all_to_sharded(segments), group))

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
        x = x @ self["weight"].T

        x = mx.distributed.all_sum(x, group=self.group)

        if "bias" in self:
            x = x + self["bias"]

        return x

    @classmethod
    def from_linear(
        cls,
        linear_layer: Module,
        *,
        segments: Union[int, list] = 1,
        group: Optional[mx.distributed.Group] = None,
    ):
        group = group or mx.distributed.init()
        output_dims, input_dims = linear_layer.weight.shape

        sl = cls(input_dims, output_dims, hasattr(linear_layer, "bias"), group)
        sl.update(_shard(linear_layer.parameters(), _sharded_to_all(segments), group))

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
        *,
        segments: Union[int, list] = 1,
        group: Optional[mx.distributed.Group] = None,
    ):
        group = group or mx.distributed.init()
        output_dims, input_dims = quantized_linear_layer.weight.shape
        input_dims *= 32 // quantized_linear_layer.bits

        sl = cls(
            input_dims,
            output_dims,
            hasattr(quantized_linear_layer, "bias"),
            group_size=quantized_linear_layer.group_size,
            bits=quantized_linear_layer.bits,
            group=group,
        )
        sl.update(
            _shard(
                quantized_linear_layer.parameters(),
                _all_to_sharded(segments),
                group,
            )
        )

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
        x = mx.distributed.all_sum(x, group=self.group)
        if "bias" in self:
            x = x + self["bias"]
        return x

    @classmethod
    def from_quantized_linear(
        cls,
        quantized_linear_layer: Module,
        *,
        segments: Union[int, list] = 1,
        group: Optional[mx.distributed.Group] = None,
    ):
        group = group or mx.distributed.init()
        output_dims, input_dims = quantized_linear_layer.weight.shape
        input_dims *= 32 // quantized_linear_layer.bits

        sl = cls(
            input_dims,
            output_dims,
            hasattr(quantized_linear_layer, "bias"),
            group_size=quantized_linear_layer.group_size,
            bits=quantized_linear_layer.bits,
            group=group,
        )
        sl.update(
            _shard(
                quantized_linear_layer.parameters(),
                _sharded_to_all(segments),
                group,
            )
        )

        return sl
