# Copyright © 2024 Apple Inc.

import math
from functools import lru_cache, reduce
from typing import Callable, Optional, Union

import mlx.core as mx
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear
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
    """Split ``dim`` as evenly as possible across ``N`` ranks.

    Give remainder blocks to the first ranks. All sizes are multiples of
    ``block``.
    """
    if dim % block != 0:
        raise ValueError(
            f"Cannot shard dimension of size {dim} across {N} devices: "
            f"the size is not a multiple of the quantization group size "
            f"({block}), so it cannot be split into quantize-able chunks."
        )

    n_blocks = dim // block
    base_blocks = n_blocks // N
    extra_blocks = n_blocks - base_blocks * N
    return [(base_blocks + (1 if r < extra_blocks else 0)) * block for r in range(N)]


def _quantized_output_sizes(dim, N, group_size):
    """Use quantization-group boundaries when a paired input shard can too."""
    block = group_size if dim % group_size == 0 and dim // group_size >= N else 1
    return _rank_sizes(dim, N, block=block)


def _resolve_sizes(dim, n_segments, N, sizes, block=1):
    """Resolve total and per-segment rank sizes for equal-sized segments.

    Return ``(cls_sizes, shard_sizes)``, where ``cls_sizes`` contains each
    rank's total and ``shard_sizes`` contains its share of one segment.
    """
    if dim % n_segments != 0:
        raise ValueError(
            f"Cannot split dimension of size {dim} into {n_segments} equal "
            f"segments."
        )
    seg_dim = dim // n_segments
    if sizes is None:
        shard_sizes = _rank_sizes(seg_dim, N, block=block)
    else:
        if len(sizes) != N:
            raise ValueError(
                f"Explicit per-rank sizes {sizes} has {len(sizes)} entries, "
                f"but the group has {N} ranks -- these must match."
            )
        if sum(sizes) != dim:
            raise ValueError(f"Explicit per-rank sizes {sizes} do not sum to {dim}.")
        if any(s % n_segments != 0 for s in sizes):
            raise ValueError(
                f"Explicit per-rank sizes {sizes} must each be evenly "
                f"divisible by the segment count {n_segments} -- each "
                f"segment needs an equal, exact share of every rank's "
                f"allocation."
            )
        shard_sizes = [s // n_segments for s in sizes]
        if block != 1 and any(s % block != 0 for s in shard_sizes):
            raise ValueError(
                f"Explicit per-rank sizes {sizes}, divided evenly across "
                f"{n_segments} segments, give each segment's share as "
                f"{shard_sizes} -- these must each be a multiple of the "
                f"quantization group_size ({block})."
            )
    cls_sizes = [s * n_segments for s in shard_sizes]
    return cls_sizes, shard_sizes


def _check_no_zero_shares(sizes):
    """Reject zero-width quantized shards on every rank."""
    if any(s == 0 for s in sizes):
        raise ValueError(
            f"Resolved per-rank sizes {sizes} give rank(s) "
            f"{[r for r, s in enumerate(sizes) if s == 0]} a zero-width "
            f"shard -- for a quantized layer this can silently produce "
            f"wrong (not just zero) results from mx.quantized_matmul, not "
            f"merely a wasted rank. Use fewer ranks, fewer segments, or a "
            f"smaller quantization block size."
        )


def _n_segments(segments, reason="of an explicit `sizes` split"):
    if isinstance(segments, int) and not isinstance(segments, bool):
        return segments
    raise ValueError(
        f"segments={segments!r}: only a plain int segment count is "
        f"supported here (a list of fractional or index-based segment "
        f"boundaries isn't, since unequal segments have no single "
        f"well-defined per-segment share {reason})."
    )


def _segment_sizes_from_spec(dim, segments):
    """Return the segment sizes produced by ``_split`` for ``dim``."""
    if isinstance(segments, int) or isinstance(segments[0], int):
        indices_or_sections = segments
    else:
        indices_or_sections = [int(s * dim) for s in segments]
    return [int(part.size) for part in mx.split(mx.arange(dim), indices_or_sections)]


def _split_uneven(weight, N, axis, sizes=None):
    """Split ``weight`` along ``axis`` into ``N`` pieces.

    ``sizes`` may use a logical dimension whose length differs from the array
    axis, as with packed quantized weights and grouped scales. In that case,
    scale each boundary to the array axis and require an exact integer index.
    """
    dim = weight.shape[axis]
    if sizes is None:
        local_sizes = _rank_sizes(dim, N)
    else:
        if len(sizes) != N:
            raise ValueError(
                f"sizes {sizes} has {len(sizes)} entries, but N={N} ranks "
                f"were requested -- these must match, or ranks beyond "
                f"len(sizes) would silently get no shard at all."
            )
        total = sum(sizes)
        if dim == total:
            local_sizes = list(sizes)
        elif total != 0 and all(
            (sum(sizes[: i + 1]) * dim) % total == 0 for i in range(len(sizes) - 1)
        ):
            # Scale logical boundaries instead of sizes to support fractional
            # packing ratios such as 3-bit weights.
            ratio_num, ratio_den = dim, total
            local_sizes = []
            acc_real = 0
            acc_local = 0
            for s in sizes[:-1]:
                acc_real += s
                nxt = (acc_real * ratio_num) // ratio_den
                local_sizes.append(nxt - acc_local)
                acc_local = nxt
            local_sizes.append(dim - acc_local)
        else:
            raise ValueError(
                f"Explicit per-rank sizes {sizes} (summing to {total}) are "
                f"not compatible with this array's axis size {dim}: at "
                f"least one cumulative boundary, scaled by the ratio "
                f"{dim}/{total}, doesn't land on a whole number."
            )
    indices = []
    acc = 0
    for s in local_sizes[:-1]:
        acc += s
        indices.append(acc)
    return mx.split(weight, indices, axis=axis)


def _normalize_axis(axis, ndim):
    if (
        not isinstance(axis, int)
        or isinstance(axis, bool)
        or axis < -ndim
        or axis >= ndim
    ):
        raise ValueError(f"Invalid sharding axis {axis} for an array with {ndim} axes.")
    return axis % ndim


def _shard(
    parameters: dict,
    sharding_predicate: Callable,
    group: Optional[mx.distributed.Group] = None,
    sizes: Optional[list] = None,
    quantized_paths=None,
):
    """Return parameters sharded according to ``sharding_predicate``.

    The sharding predicate should return the sharding axis and optionally also
    the segments that comprise the weight. ``sizes`` sets logical per-rank
    sizes. ``quantized_paths`` keeps packed weights and grouped metadata at
    matching logical boundaries.
    """
    group = group or mx.distributed.init()
    N = group.size()
    r = group.rank()

    def _group_of(path):
        return path.rsplit(".", 1)[0] if "." in path else ""

    def _leaf_name(path):
        return path.rsplit(".", 1)[-1]

    quantized_paths = {} if quantized_paths is None else dict(quantized_paths)
    predicate_results = {}
    sibling_sizes = {}
    if sizes is None and quantized_paths:
        for path, leaf in tree_flatten(parameters):
            if (
                not isinstance(leaf, mx.array)
                or _leaf_name(path) != "weight"
                or _group_of(path) not in quantized_paths
            ):
                continue
            shard_spec = sharding_predicate(path, leaf)
            predicate_results[path] = shard_spec
            if shard_spec is None:
                continue
            axis = shard_spec if isinstance(shard_spec, int) else shard_spec[0]
            axis = _normalize_axis(axis, leaf.ndim)
            segments = 1 if isinstance(shard_spec, int) else shard_spec[1]
            group_size, bits = quantized_paths[_group_of(path)]
            # Quantization packs and groups only the last weight axis.
            is_packed_axis = axis == leaf.ndim - 1
            weight_seg_sizes = [
                part.shape[axis] for part in _split(leaf, segments, axis)
            ]
            if is_packed_axis:
                real_seg_sizes = [sz * 32 // bits for sz in weight_seg_sizes]
                weight_seg_rank_sizes = [
                    _rank_sizes(sz, N, block=group_size) for sz in real_seg_sizes
                ]
            else:
                weight_seg_rank_sizes = [
                    _quantized_output_sizes(sz, N, group_size)
                    for sz in weight_seg_sizes
                ]
            # Quantized matmul receives the concatenated segments.
            total_rank_sizes = [
                sum(seg[rk] for seg in weight_seg_rank_sizes) for rk in range(N)
            ]
            _check_no_zero_shares(total_rank_sizes)
            sibling_sizes[_group_of(path)] = (
                axis,
                weight_seg_sizes,
                weight_seg_rank_sizes,
            )

    def _shard_fn(path, weight):
        if not isinstance(weight, mx.array):
            return weight

        if path in predicate_results:
            shard_spec = predicate_results[path]
        else:
            shard_spec = sharding_predicate(path, weight)
        if shard_spec is None:
            return weight

        axis = None
        segments = 1
        if isinstance(shard_spec, int):
            axis = shard_spec
        elif isinstance(shard_spec, tuple):
            axis, segments = shard_spec
        else:
            raise ValueError(
                "The sharding function should return int or tuple[int, list]"
            )
        axis = _normalize_axis(axis, weight.ndim)

        quantized_ref = None
        if sizes is None and _group_of(path) in quantized_paths:
            quantized_ref = sibling_sizes.get(_group_of(path))
            if quantized_ref is not None and quantized_ref[0] != axis:
                quantized_ref = None
            if quantized_ref is None and _leaf_name(path) in ("scales", "biases"):
                raise ValueError(
                    f"Cannot shard quantized parameter {path!r}: its "
                    f"'weight' sibling was not sharded consistently on "
                    f"axis {axis} (the sharding predicate returned `None` "
                    f"for it, or a different axis) -- there is no safe "
                    f"reference split to keep this leaf's boundaries "
                    f"consistent with weight's."
                )

        segment_sizes = None
        if sizes and isinstance(sizes[0], (list, tuple)):
            segment_sizes = sizes
            logical_sizes = [sum(s) for s in segment_sizes]
            parts = _split_uneven(weight, len(logical_sizes), axis, sizes=logical_sizes)
        elif quantized_ref is not None:
            # Reuse the weight boundaries for its packed/grouped metadata.
            parts = _split_uneven(
                weight, len(quantized_ref[1]), axis, sizes=quantized_ref[1]
            )
        else:
            parts = _split(weight, segments, axis)

        per_segment_sizes = quantized_ref[2] if quantized_ref is not None else None
        return mx.contiguous(
            mx.concatenate(
                [
                    _split_uneven(
                        part,
                        N,
                        axis,
                        (
                            per_segment_sizes[i]
                            if per_segment_sizes
                            else segment_sizes[i] if segment_sizes else sizes
                        ),
                    )[r]
                    for i, part in enumerate(parts)
                ],
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
    # Detect compatible third-party quantized layers by their parameters.
    quantized_paths = {
        path: (child.group_size, child.bits)
        for path, child in module.named_modules()
        if isinstance(getattr(child, "group_size", None), int)
        and isinstance(getattr(child, "bits", None), int)
        and "scales" in child
    }
    module.update(
        _shard(
            module.parameters(),
            sharding,
            group,
            quantized_paths=quantized_paths,
        )
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
        segments (int or list): The segments to split independently before
            sharding. Explicit ``sizes`` require an integer segment count.
            Default: ``1``.
        group (mlx.core.distributed.Group): The distributed group to shard
            across. If not set, the global group will be used. Default: ``None``.
        sizes (list, optional): Explicit sizes for each rank. The sizes must
            sum to the sharded dimension. Quantized input sizes must be
            multiples of ``group_size``. Default: ``None``.

    .. note::
        Paired quantized layers with different ``group_size`` values should
        use explicit matching ``sizes``.
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
        sizes (list, optional): Explicit per-rank sizes (one entry per rank in
            ``group``, summing to the full sharded dimension) to use instead
            of the automatic remainder-aware split. Default: ``None``.
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
        r = self.group.rank()

        # Each rank gets a possibly uneven slice of the output features.
        if sizes is not None:
            if len(sizes) != N:
                raise ValueError(
                    f"Explicit per-rank sizes {sizes} has {len(sizes)} "
                    f"entries, but the group has {N} ranks -- these must "
                    f"match."
                )
            if any(s < 0 for s in sizes):
                raise ValueError(
                    f"Explicit per-rank sizes {sizes} must all be " f"non-negative."
                )
            if sum(sizes) != output_dims:
                raise ValueError(
                    f"Explicit per-rank sizes {sizes} do not sum to "
                    f"output_dims={output_dims}."
                )
            my_output_dims = sizes[r]
        else:
            my_output_dims = _rank_sizes(output_dims, N)[r]
        self._total_output_dims = output_dims

        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(my_output_dims, input_dims),
        )
        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(my_output_dims,),
            )

    def _extra_repr(self) -> str:
        out_dims, in_dims = self.weight.shape
        out_dims = self._total_output_dims
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
        if sizes is None and not isinstance(segments, int):
            # Split unequal segments independently across ranks.
            seg_sizes = _segment_sizes_from_spec(output_dims, segments)
            cls_sizes = [sum(_rank_sizes(s, N)[r] for s in seg_sizes) for r in range(N)]
            shard_sizes = None
        else:
            n_segments = _n_segments(segments)
            if n_segments == 1:
                cls_sizes = sizes if sizes is not None else _rank_sizes(output_dims, N)
                shard_sizes = cls_sizes
            else:
                cls_sizes, shard_sizes = _resolve_sizes(
                    output_dims, n_segments, N, sizes
                )

        sl = cls(
            input_dims, output_dims, hasattr(linear_layer, "bias"), group, cls_sizes
        )
        sl.update(
            _shard(
                linear_layer.parameters(),
                _all_to_sharded(segments),
                group,
                sizes=shard_sizes,
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
        sizes (list, optional): Explicit per-rank sizes (one entry per rank in
            ``group``, summing to the full sharded dimension) to use instead
            of the automatic remainder-aware split. Default: ``None``.
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
        r = self.group.rank()

        # Each rank gets a possibly uneven slice of the input features.
        if sizes is not None:
            if len(sizes) != N:
                raise ValueError(
                    f"Explicit per-rank sizes {sizes} has {len(sizes)} "
                    f"entries, but the group has {N} ranks -- these must "
                    f"match."
                )
            if any(s < 0 for s in sizes):
                raise ValueError(
                    f"Explicit per-rank sizes {sizes} must all be " f"non-negative."
                )
            if sum(sizes) != input_dims:
                raise ValueError(
                    f"Explicit per-rank sizes {sizes} do not sum to "
                    f"input_dims={input_dims}."
                )
            my_input_dims = sizes[r]
        else:
            my_input_dims = _rank_sizes(input_dims, N)[r]
        self._total_input_dims = input_dims

        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, my_input_dims),
        )
        if bias:
            self.bias = mx.random.uniform(
                low=-scale,
                high=scale,
                shape=(output_dims,),
            )

    def _extra_repr(self) -> str:
        out_dims, in_dims = self.weight.shape
        in_dims = self._total_input_dims
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
        if sizes is None and not isinstance(segments, int):
            # Split unequal segments independently across ranks.
            seg_sizes = _segment_sizes_from_spec(input_dims, segments)
            cls_sizes = [sum(_rank_sizes(s, N)[r] for s in seg_sizes) for r in range(N)]
            shard_sizes = None
        else:
            n_segments = _n_segments(segments)
            if n_segments == 1:
                cls_sizes = sizes if sizes is not None else _rank_sizes(input_dims, N)
                shard_sizes = cls_sizes
            else:
                cls_sizes, shard_sizes = _resolve_sizes(
                    input_dims, n_segments, N, sizes
                )

        sl = cls(
            input_dims, output_dims, hasattr(linear_layer, "bias"), group, cls_sizes
        )
        sl.update(
            _shard(
                linear_layer.parameters(),
                _sharded_to_all(segments),
                group,
                sizes=shard_sizes,
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
        sizes (list, optional): Explicit per-rank sizes (one entry per rank in
            ``group``, summing to the full sharded dimension) to use instead
            of the automatic remainder-aware split. Default: ``None``.
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
        r = self.group.rank()

        # Prefer group boundaries to match a paired input-sharded layer.
        if sizes is not None:
            if len(sizes) != N:
                raise ValueError(
                    f"Explicit per-rank sizes {sizes} has {len(sizes)} "
                    f"entries, but the group has {N} ranks -- these must "
                    f"match."
                )
            if any(s < 0 for s in sizes):
                raise ValueError(
                    f"Explicit per-rank sizes {sizes} must all be " f"non-negative."
                )
            if sum(sizes) != output_dims:
                raise ValueError(
                    f"Explicit per-rank sizes {sizes} do not sum to "
                    f"output_dims={output_dims}."
                )
            my_output_dims = sizes[r]
        else:
            sizes = _quantized_output_sizes(output_dims, N, group_size)
            my_output_dims = sizes[r]
        _check_no_zero_shares(sizes)
        self._total_output_dims = output_dims

        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(my_output_dims, input_dims),
        )
        self.weight, self.scales, *biases = mx.quantize(
            weight, group_size, bits, mode=mode
        )
        self.biases = biases[0] if biases else None

        # And bias if needed
        if bias:
            self.bias = mx.zeros((my_output_dims,))

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
        out_dims = self._total_output_dims
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
        # Use the same sizes for allocation and parameter slicing.
        if sizes is None and not isinstance(segments, int):
            seg_sizes = _segment_sizes_from_spec(output_dims, segments)
            shard_sizes = [
                _quantized_output_sizes(s, N, quantized_linear_layer.group_size)
                for s in seg_sizes
            ]
            cls_sizes = [sum(s[r] for s in shard_sizes) for r in range(N)]
        else:
            n_segments = _n_segments(segments)
            if n_segments == 1:
                cls_sizes = (
                    sizes
                    if sizes is not None
                    else _quantized_output_sizes(
                        output_dims, N, quantized_linear_layer.group_size
                    )
                )
                shard_sizes = cls_sizes
            else:
                # Output rows need not use group boundaries when there are
                # fewer groups than ranks.
                seg_dim = output_dims // n_segments
                group_size = quantized_linear_layer.group_size
                block = (
                    group_size
                    if sizes is None
                    and seg_dim % group_size == 0
                    and seg_dim // group_size >= N
                    else 1
                )
                cls_sizes, shard_sizes = _resolve_sizes(
                    output_dims, n_segments, N, sizes, block=block
                )

        sl = cls(
            input_dims,
            output_dims,
            hasattr(quantized_linear_layer, "bias"),
            group_size=quantized_linear_layer.group_size,
            bits=quantized_linear_layer.bits,
            mode=getattr(quantized_linear_layer, "mode", "affine"),
            group=group,
            sizes=cls_sizes,
        )
        sl.update(
            _shard(
                quantized_linear_layer.parameters(),
                _all_to_sharded(segments),
                group,
                sizes=shard_sizes,
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
        sizes (list, optional): Explicit per-rank sizes. Each size must be a
            multiple of ``group_size``. Default: ``None``.
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
        r = self.group.rank()

        # Keep each input shard aligned to quantization groups.
        if sizes is not None:
            if len(sizes) != N:
                raise ValueError(
                    f"Explicit per-rank sizes {sizes} has {len(sizes)} "
                    f"entries, but the group has {N} ranks -- these must "
                    f"match."
                )
            if any(s < 0 for s in sizes):
                raise ValueError(
                    f"Explicit per-rank sizes {sizes} must all be " f"non-negative."
                )
            if sum(sizes) != input_dims:
                raise ValueError(
                    f"Explicit per-rank sizes {sizes} do not sum to "
                    f"input_dims={input_dims}."
                )
            if any(s % group_size != 0 for s in sizes):
                raise ValueError(
                    f"Explicit per-rank sizes {sizes} must each be a "
                    f"multiple of group_size={group_size}."
                )
            my_input_dims = sizes[r]
        else:
            sizes = _rank_sizes(input_dims, N, block=group_size)
            my_input_dims = sizes[r]
        _check_no_zero_shares(sizes)
        self._total_input_dims = input_dims

        weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dims, my_input_dims),
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
        out_dims, in_dims = self.weight.shape
        in_dims = self._total_input_dims
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
        # Use the same logical sizes for packed weights and grouped metadata.
        group_size = quantized_linear_layer.group_size
        if sizes is None and not isinstance(segments, int):
            seg_sizes = _segment_sizes_from_spec(input_dims, segments)
            shard_sizes = [_rank_sizes(s, N, block=group_size) for s in seg_sizes]
            cls_sizes = [sum(s[r] for s in shard_sizes) for r in range(N)]
        else:
            n_segments = _n_segments(segments)
            if n_segments == 1:
                cls_sizes = (
                    sizes
                    if sizes is not None
                    else _rank_sizes(input_dims, N, block=group_size)
                )
                shard_sizes = cls_sizes
            else:
                cls_sizes, shard_sizes = _resolve_sizes(
                    input_dims, n_segments, N, sizes, block=group_size
                )

        sl = cls(
            input_dims,
            output_dims,
            hasattr(quantized_linear_layer, "bias"),
            group_size=quantized_linear_layer.group_size,
            bits=quantized_linear_layer.bits,
            mode=getattr(quantized_linear_layer, "mode", "affine"),
            group=group,
            sizes=cls_sizes,
        )
        sl.update(
            _shard(
                quantized_linear_layer.parameters(),
                _sharded_to_all(segments),
                group,
                sizes=shard_sizes,
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
