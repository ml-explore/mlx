# Copyright © 2024 Apple Inc.

import math
from functools import lru_cache, reduce
from typing import Callable, Optional, Union

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear
from mlx.nn.layers.quantized import QuantizedLinear
from mlx.utils import tree_flatten, tree_map_with_path, tree_unflatten


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


def _rank_sizes(dim, N, block=1):
    """Split ``dim`` as evenly as possible across ``N`` ranks in multiples of
    ``block``, giving the remainder to the first ranks."""
    base, extra = divmod(dim // block, N)
    return [(base + (r < extra)) * block for r in range(N)]


def _quantized_output_sizes(dim, N, group_size):
    """Split the output rows of a quantized layer.

    Rows can be split anywhere, but splitting at multiples of ``group_size``
    when possible matches the default split of a paired sharded-to-all layer.
    """
    if dim % group_size == 0 and dim // group_size >= N:
        return _rank_sizes(dim, N, group_size)
    return _rank_sizes(dim, N)


def _resolve_sizes(dim, N, sizes, name, block=1):
    """Validate the size of each rank's shard of ``dim``, splitting it evenly
    in multiples of ``block`` if ``sizes`` is not given."""
    if sizes is None:
        sizes = _rank_sizes(dim, N, block)
    if len(sizes) != N or sum(sizes) != dim:
        raise ValueError(f"Expected {N} sizes that sum to {dim} but got {sizes}.")
    if min(sizes) <= 0:
        raise ValueError(f"Cannot shard the {name} of size {dim} across {N} devices.")
    if any(s % block for s in sizes):
        raise ValueError(f"The sizes {sizes} must be multiples of {block}.")
    return list(sizes)


def _layer_sizes(dim, N, segments, sizes, default, name):
    """Return the per-rank sizes of a layer and the sizes to shard its
    parameters with. Layers with more than one segment are split evenly."""
    if segments == 1:
        sizes = default if sizes is None else sizes
        return sizes, sizes
    if sizes is not None:
        raise ValueError("Explicit sizes are only supported with segments=1.")
    if dim % N != 0:
        raise ValueError(f"Cannot shard the {name} of size {dim} across {N} devices.")
    return [dim // N] * N, None


def _split_sizes(weight, sizes, axis):
    """Split ``weight`` along ``axis`` into parts proportional to ``sizes``.

    ``sizes`` may sum to more than the length of ``axis``, as is the case for
    packed quantized weights and their scales, but every boundary must land on
    an integer index.
    """
    dim = weight.shape[axis]
    total = sum(sizes)
    indices = []
    boundary = 0
    for s in sizes[:-1]:
        boundary += s
        index, remainder = divmod(boundary * dim, total)
        if remainder != 0:
            raise ValueError(
                f"Cannot split an axis of size {dim} according to sizes {sizes}."
            )
        indices.append(index)
    return mx.split(weight, indices, axis=axis)


def _quantized_sizes(parameters, sharding_predicate, N, quantized_paths):
    """Return the per-rank sizes of each quantized module's sharded axis.

    A quantized weight is packed along its last axis and its scales and biases
    are grouped along the same axis, so all three are split from one list of
    sizes in unpacked elements to keep them at matching boundaries. The sizes
    are multiples of ``group_size`` so that a paired all-to-sharded and
    sharded-to-all layer split their common dimension the same way.
    """
    rank_sizes = {}
    for path, weight in tree_flatten(parameters):
        module, _, name = path.rpartition(".")
        if name != "weight" or module not in quantized_paths:
            continue
        shard_spec = sharding_predicate(path, weight)
        if shard_spec is None:
            continue
        if isinstance(shard_spec, tuple):
            axis, segments = shard_spec
        else:
            axis, segments = shard_spec, 1
        if segments != 1:
            continue
        group_size, bits = quantized_paths[module]
        if axis % weight.ndim == weight.ndim - 1:
            dim = (weight.shape[axis] * 32) // bits
            sizes = _rank_sizes(dim, N, group_size)
        else:
            dim = weight.shape[axis]
            sizes = _quantized_output_sizes(dim, N, group_size)
        if min(sizes) <= 0:
            raise ValueError(
                f"Cannot shard the quantized {module or 'module'} of size "
                f"{dim} across {N} devices."
            )
        rank_sizes[module] = sizes
    return rank_sizes


def _shard(
    parameters: dict,
    sharding_predicate: Callable,
    group: Optional[mx.distributed.Group] = None,
    sizes: Optional[list] = None,
    quantized_paths: Optional[dict] = None,
):
    """Returns a new parameter tree with the weights sharded according to the
    sharding_predicate.

    The sharding predicate should return the sharding axis and optionally also
    the segments that comprise the weight. If ``sizes`` is provided, each rank
    gets the corresponding proportion of the sharded axis instead of an equal
    part. ``sizes`` requires a single segment. ``quantized_paths`` maps the
    path of a quantized module to its ``(group_size, bits)`` so that its
    packed weight and grouped metadata stay at matching boundaries.
    """
    group = group or mx.distributed.init()
    N = group.size()
    r = group.rank()
    quantized_sizes = (
        _quantized_sizes(parameters, sharding_predicate, N, quantized_paths)
        if sizes is None and quantized_paths
        else {}
    )

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

        rank_sizes = sizes
        if rank_sizes is None:
            rank_sizes = quantized_sizes.get(path.rpartition(".")[0])
        if rank_sizes is None and segments == 1:
            # Split as evenly as possible rather than requiring the axis to
            # divide by the number of devices.
            rank_sizes = _rank_sizes(weight.shape[axis], N)
            if min(rank_sizes) <= 0:
                raise ValueError(
                    f"Cannot shard {path!r} of size {weight.shape[axis]} "
                    f"across {N} devices."
                )
        if rank_sizes is not None:
            return mx.contiguous(_split_sizes(weight, rank_sizes, axis)[r])

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
        if path.endswith("bias"):
            return -1, segments
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
    # Detect quantized modules, including third party ones, so that their
    # packed weights and grouped metadata are split at the same boundaries.
    quantized_paths = {
        path: (child.group_size, child.bits)
        for path, child in module.named_modules()
        if isinstance(getattr(child, "group_size", None), int)
        and isinstance(getattr(child, "bits", None), int)
        and "scales" in child
    }
    module.update(
        _shard(module.parameters(), sharding, group, quantized_paths=quantized_paths)
    )


def shard_linear(
    module: Module,
    sharding: str,
    *,
    segments: Union[int, list] = 1,
    group: Optional[mx.distributed.Group] = None,
    sizes: Optional[list] = None,
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
        sizes (list, optional): The size of each rank's shard of the sharded
            dimension. If not set, the dimension is split as evenly as
            possible with the remainder going to the first ranks. Uneven
            splits require ``segments=1``. Default: ``None``.
    """
    _check_sharding(sharding)
    fns = {
        ("all-to-sharded", True): AllToShardedLinear.from_linear,
        ("all-to-sharded", False): QuantizedAllToShardedLinear.from_quantized_linear,
        ("sharded-to-all", True): ShardedToAllLinear.from_linear,
        ("sharded-to-all", False): QuantizedShardedToAllLinear.from_quantized_linear,
    }
    return fns[sharding, isinstance(module, Linear)](
        module, segments=segments, group=group, sizes=sizes
    )


class AllToShardedLinear(Module):
    """Each member of the group applies part of the affine transformation such
    that the result is sharded across the group.

    The gradients are automatically aggregated from each member of the group.

    Args:
        input_dims (int): The dimensionality of the input features
        output_dims (int): The dimensionality of the output features
        bias (bool, optional): If set to ``False`` the layer will not use a
            bias. Default is ``True``.
        group (mx.distributed.Group, optional): The sharding will happen across
            this group. If not set then the global group is used. Default is
            ``None``.
        sizes (list, optional): The size of each rank's shard of the output
            features. If not set, they are split as evenly as possible.
            Default: ``None``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group: Optional[mx.distributed.Group] = None,
        sizes: Optional[list] = None,
    ):
        super().__init__()

        # Initialize the parameters
        scale = math.sqrt(1.0 / input_dims)
        self.group = group or mx.distributed.init()
        N = self.group.size()

        sizes = _resolve_sizes(output_dims, N, sizes, "output")
        local_output_dims = sizes[self.group.rank()]
        self._output_dims = output_dims

        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(local_output_dims, input_dims),
        )
        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(local_output_dims,),
            )

    def _extra_repr(self) -> str:
        in_dims = self.weight.shape[1]
        out_dims = self._output_dims
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
        sizes: Optional[list] = None,
    ):
        group = group or mx.distributed.init()
        N = group.size()
        output_dims, input_dims = linear_layer.weight.shape
        sizes, shard_sizes = _layer_sizes(
            output_dims, N, segments, sizes, _rank_sizes(output_dims, N), "output"
        )

        sl = cls(input_dims, output_dims, hasattr(linear_layer, "bias"), group, sizes)
        sl.update(
            _shard(
                linear_layer.parameters(), _all_to_sharded(segments), group, shard_sizes
            )
        )

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
        bias (bool, optional): If set to ``False`` the layer will not use a
            bias. Default is ``True``.
        group (mx.distributed.Group, optional): The sharding will happen across
            this group. If not set then the global group is used. Default is
            ``None``.
        sizes (list, optional): The size of each rank's shard of the input
            features. If not set, they are split as evenly as possible.
            Default: ``None``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group: Optional[mx.distributed.Group] = None,
        sizes: Optional[list] = None,
    ):
        super().__init__()

        # Initialize the parameters
        scale = math.sqrt(1.0 / input_dims)
        self.group = group or mx.distributed.init()
        N = self.group.size()

        sizes = _resolve_sizes(input_dims, N, sizes, "input")
        local_input_dims = sizes[self.group.rank()]
        self._input_dims = input_dims

        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, local_input_dims),
        )
        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(output_dims,),
            )

    def _extra_repr(self) -> str:
        out_dims = self.weight.shape[0]
        in_dims = self._input_dims
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
        sizes: Optional[list] = None,
    ):
        group = group or mx.distributed.init()
        N = group.size()
        output_dims, input_dims = linear_layer.weight.shape
        sizes, shard_sizes = _layer_sizes(
            input_dims, N, segments, sizes, _rank_sizes(input_dims, N), "input"
        )

        sl = cls(input_dims, output_dims, hasattr(linear_layer, "bias"), group, sizes)
        sl.update(
            _shard(
                linear_layer.parameters(), _sharded_to_all(segments), group, shard_sizes
            )
        )

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
        mode (str, optional): The quantization method to use (see
            :func:`~mlx.core.quantize`). Default: ``"affine"``.
        group (mx.distributed.Group, optional): The sharding will happen across
            this group. If not set then the global group is used. Default is
            ``None``.
        sizes (list, optional): The size of each rank's shard of the output
            features. If not set, they are split as evenly as possible,
            preferring multiples of ``group_size``. Default: ``None``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        group: Optional[mx.distributed.Group] = None,
        sizes: Optional[list] = None,
    ):
        super().__init__()

        # Quantization config
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Initialize the quantized weight
        scale = math.sqrt(1.0 / input_dims)
        self.group = group or mx.distributed.init()
        N = self.group.size()

        if sizes is None:
            sizes = _quantized_output_sizes(output_dims, N, group_size)
        sizes = _resolve_sizes(output_dims, N, sizes, "output")
        local_output_dims = sizes[self.group.rank()]
        self._output_dims = output_dims

        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(local_output_dims, input_dims),
        )
        self.weight, self.scales, *biases = mx.quantize(
            weight, group_size, bits, mode=mode
        )
        self.biases = biases[0] if biases else None

        # And bias if needed
        if bias:
            self.bias = mx.zeros((local_output_dims,))

        # Freeze this model's parameters
        self.freeze()

    def unfreeze(self, *args, **kwargs):
        """Wrap unfreeze so that we unfreeze any layers we might contain but
        our parameters will remain frozen."""
        super().unfreeze(*args, **kwargs)
        self.freeze(recurse=False)

    def _extra_repr(self) -> str:
        out_dims, in_dims = self.weight.shape
        in_dims = (in_dims * 32) // self.bits
        out_dims = self._output_dims
        return (
            f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self}, "
            f"group_size={self.group_size}, bits={self.bits}, mode={self.mode}"
        )

    def __call__(self, x: mx.array) -> mx.array:
        # Aggregate the gradients coming from each shard
        x = sum_gradients(self.group)(x)

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
    def from_quantized_linear(
        cls,
        quantized_linear_layer: Module,
        *,
        segments: Union[int, list] = 1,
        group: Optional[mx.distributed.Group] = None,
        sizes: Optional[list] = None,
    ):
        group = group or mx.distributed.init()
        N = group.size()
        output_dims, input_dims = quantized_linear_layer.weight.shape
        input_dims = (input_dims * 32) // quantized_linear_layer.bits
        group_size = quantized_linear_layer.group_size
        sizes, shard_sizes = _layer_sizes(
            output_dims,
            N,
            segments,
            sizes,
            _quantized_output_sizes(output_dims, N, group_size),
            "output",
        )

        sl = cls(
            input_dims,
            output_dims,
            hasattr(quantized_linear_layer, "bias"),
            group_size=group_size,
            bits=quantized_linear_layer.bits,
            mode=getattr(quantized_linear_layer, "mode", "affine"),
            group=group,
            sizes=sizes,
        )
        sl.update(
            _shard(
                quantized_linear_layer.parameters(),
                _all_to_sharded(segments),
                group,
                shard_sizes,
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
        mode (str, optional): The quantization method to use (see
            :func:`~mlx.core.quantize`). Default: ``"affine"``.
        group (mx.distributed.Group, optional): The sharding will happen across
            this group. If not set then the global group is used. Default is
            ``None``.
        sizes (list, optional): The size of each rank's shard of the input
            features. Each size must be a multiple of ``group_size``. If not
            set, they are split as evenly as possible. Default: ``None``.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        bias: bool = True,
        group_size: int = 64,
        bits: int = 4,
        mode: str = "affine",
        group: Optional[mx.distributed.Group] = None,
        sizes: Optional[list] = None,
    ):
        super().__init__()

        # Quantization config
        self.group_size = group_size
        self.bits = bits
        self.mode = mode

        # Initialize the quantized weight
        scale = math.sqrt(1.0 / input_dims)
        self.group = group or mx.distributed.init()
        N = self.group.size()

        sizes = _resolve_sizes(input_dims, N, sizes, "input", group_size)
        local_input_dims = sizes[self.group.rank()]
        self._input_dims = input_dims

        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, local_input_dims),
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

    def unfreeze(self, *args, **kwargs):
        """Wrap unfreeze so that we unfreeze any layers we might contain but
        our parameters will remain frozen."""
        super().unfreeze(*args, **kwargs)
        self.freeze(recurse=False)

    def _extra_repr(self) -> str:
        out_dims = self.weight.shape[0]
        in_dims = self._input_dims
        return (
            f"input_dims={in_dims}, output_dims={out_dims}, bias={'bias' in self}, "
            f"group_size={self.group_size}, bits={self.bits}, mode={self.mode}"
        )

    def __call__(self, x: mx.array) -> mx.array:
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
        sizes: Optional[list] = None,
    ):
        group = group or mx.distributed.init()
        N = group.size()
        output_dims, input_dims = quantized_linear_layer.weight.shape
        input_dims = (input_dims * 32) // quantized_linear_layer.bits
        group_size = quantized_linear_layer.group_size
        sizes, shard_sizes = _layer_sizes(
            input_dims,
            N,
            segments,
            sizes,
            _rank_sizes(input_dims, N, group_size),
            "input",
        )

        sl = cls(
            input_dims,
            output_dims,
            hasattr(quantized_linear_layer, "bias"),
            group_size=group_size,
            bits=quantized_linear_layer.bits,
            mode=getattr(quantized_linear_layer, "mode", "affine"),
            group=group,
            sizes=sizes,
        )
        sl.update(
            _shard(
                quantized_linear_layer.parameters(),
                _sharded_to_all(segments),
                group,
                shard_sizes,
            )
        )

        return sl


def _make_gather_fn(group, full_shapes, shard_sizes, compute_dtype):
    N = group.size()
    indices = reduce(lambda acc, w: acc + [acc[-1] + w], shard_sizes, [0])
    split_indices = indices[1:-1]
    shard_shapes = [(shape[0] // N,) + tuple(shape[1:]) for shape in full_shapes]

    def _maybe_cast(x, dtype):
        if dtype is None or x.dtype == dtype:
            return x
        return x.astype(dtype)

    @mx.custom_function
    def gather(shards):
        shard = mx.concatenate(
            [_maybe_cast(s.reshape(1, -1), compute_dtype) for s in shards], axis=1
        )
        full = mx.distributed.all_gather(shard, group=group)
        parts = mx.split(full, split_indices, axis=1)
        return [p.reshape(shape) for p, shape in zip(parts, full_shapes)]

    @gather.vjp
    def gather_vjp(shards, cotangents, _):
        local_full = mx.concatenate([c.reshape(N, -1) for c in cotangents], axis=1)
        local_shard = mx.distributed.sum_scatter(local_full, group=group) / N
        parts = mx.split(local_shard, split_indices, axis=1)
        return [
            _maybe_cast(p.reshape(shape), s.dtype)
            for p, shape, s in zip(parts, shard_shapes, shards)
        ]

    return gather


def _maybe_shard(m, k, v):
    if isinstance(v, FullyShardedModule):
        return False
    return Module.valid_parameter_filter(m, k, v)


class FullyShardedModule(Module):
    """Wrap a module so each member of the group holds only a shard of its
    parameters.

    The full parameters are gathered for the forward pass and the gradients
    are reduce-scattered in the backward pass, so during training
    each member of the group stores and updates only its own shard.

    Every parameter is sharded along axis 0, so each parameter's size along
    that axis must be divisible by the size of ``group``.

    Use :func:`~mlx.nn.layers.distributed.fully_shard` to wrap a module.

    Args:
        module (mlx.nn.Module): The module whose parameters will be sharded.
        group (mlx.core.distributed.Group, optional): The group to shard
            across. If not set, the global group is used. Default: ``None``.
        compute_dtype (mlx.core.Dtype, optional): If set, the gathered
            parameters are cast to this dtype for the forward pass.
            Default: ``None``.
    """

    def __init__(
        self,
        module: Module,
        group: Optional[mx.distributed.Group] = None,
        compute_dtype: Optional[mx.Dtype] = None,
    ):
        super().__init__()
        group = group or mx.distributed.init()
        N = group.size()

        shard_params = module.filter_and_map(_maybe_shard)
        flat = tree_flatten(shard_params)
        for path, a in flat:
            if a.ndim == 0:
                raise ValueError(
                    f"Cannot shard parameter '{path}' because it is a scalar."
                )
            if a.shape[0] % N != 0:
                raise ValueError(
                    f"Cannot shard parameter '{path}' with shape {a.shape} "
                    f"across {N} devices: axis 0 must be divisible by {N}."
                )

        super(Module, self).__setattr__("_paths", [k for k, _ in flat])
        full_shapes = [a.shape for _, a in flat]
        shard_sizes = [a.size // N for _, a in flat]

        module.update(_shard(shard_params, lambda p, w: 0, group))

        self.module = module
        self._gather_fn = _make_gather_fn(
            group, full_shapes, shard_sizes, compute_dtype
        )

    def _extra_repr(self) -> str:
        return f"num_sharded_params={len(self._paths)}"

    def _gathered_call(self, fn, *args, **kwargs):
        shard_tree = self.module.filter_and_map(_maybe_shard)
        shards = [a for _, a in tree_flatten(shard_tree)]
        fulls = self._gather_fn(shards)
        self.module.update(tree_unflatten(list(zip(self._paths, fulls))))
        try:
            return fn(*args, **kwargs)
        finally:
            self.module.update(shard_tree)

    def __call__(self, *args, **kwargs):
        return self._gathered_call(self.module, *args, **kwargs)

    def as_linear(self, *args, **kwargs):
        return self._gathered_call(self.module.as_linear, *args, **kwargs)


def fully_shard(
    module: Module,
    *,
    group: Optional[mx.distributed.Group] = None,
    compute_dtype: Optional[mx.Dtype] = None,
) -> Module:
    """Wrap ``module`` in a :class:`FullyShardedModule`.

    Args:
        module (mlx.nn.Module): The module to wrap.
        group (mlx.core.distributed.Group, optional): The group to shard
            across. If not set, the global group is used. Default: ``None``.
        compute_dtype (mlx.core.Dtype, optional): If set, the gathered
            parameters are cast to this dtype for the forward pass.
            Default: ``None``.

    Returns:
        The wrapped :class:`FullyShardedModule`, or ``module`` unchanged.
    """
    group = group or mx.distributed.init()
    if group.size() == 1:
        return module
    if isinstance(module, FullyShardedModule):
        return module

    wrapped = FullyShardedModule(module, group=group, compute_dtype=compute_dtype)
    return wrapped if wrapped._paths else module
