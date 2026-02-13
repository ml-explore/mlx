# Copyright © 2023-2024 Apple Inc.

from functools import wraps
from typing import Any, Callable, List, Optional

import mlx.core as mx

from ..utils import tree_flatten, tree_map, tree_reduce, tree_unflatten
from .layers.base import Module


def value_and_grad(model: Module, fn: Callable):
    """Transform the passed function ``fn`` to a function that computes the
    gradients of ``fn`` wrt the model's trainable parameters and also its
    value.

    Args:
        model (mlx.nn.Module): The model whose trainable parameters to compute
                               gradients for
        fn (Callable): The scalar function to compute gradients for

    Returns:
        A callable that returns the value of ``fn`` and the gradients wrt the
        trainable parameters of ``model``
    """

    def inner_fn(params, *args, **kwargs):
        model.update(params)
        return fn(*args, **kwargs)

    value_grad_fn = mx.value_and_grad(inner_fn)

    @wraps(fn)
    def wrapped_value_grad_fn(*args, **kwargs):
        value, grad = value_grad_fn(model.trainable_parameters(), *args, **kwargs)
        return value, grad

    return wrapped_value_grad_fn


def checkpoint(module: Module, fn: Optional[Callable] = None):
    """Transform the passed callable to one that performs gradient
    checkpointing with respect to the trainable parameters of the module (and
    the callable's inputs).

    Args:
        module (mlx.nn.Module): The module for whose parameters we will be
            performing gradient checkpointing.
        fn (Callable, optional): The function to checkpoint. If not provided it
            defaults to the provided module.

    Returns:
        A callable that saves the inputs and outputs during the forward pass
        and recomputes all intermediate states during the backward pass.
    """
    if fn is None:
        # Capturing module instead of module.__call__ allows someone to
        # monkey-patch __call__ later on and the correct method will be used
        fn = module

    def inner_fn(params, *args, **kwargs):
        module.update(params)
        return fn(*args, **kwargs)

    checkpointed_fn = mx.checkpoint(inner_fn)

    @wraps(fn)
    def wrapped_checkpointed_fn(*args, **kwargs):
        return checkpointed_fn(module.trainable_parameters(), *args, **kwargs)

    return wrapped_checkpointed_fn


def _group_arrays_by_size(
    sizes: List[int],
    itemsize: int,
    threshold: int,
) -> List[List[int]]:
    groups = []
    current_group = []
    current_size = 0
    for i in range(len(sizes)):
        current_group.append(i)
        current_size += sizes[i] * itemsize
        if current_size >= threshold:
            groups.append(current_group)
            current_group = []
            current_size = 0
    if current_group:
        groups.append(current_group)
    return groups


def _get_itemsize(
    communication_type: Optional[mx.Dtype],
    default_dtype: mx.Dtype,
) -> int:
    return (
        communication_type.size
        if communication_type is not None
        else default_dtype.size
    )


def _compute_split_indices(group: List[int], elem_sizes: List[int]) -> List[int]:
    indices = [0]
    for i in group:
        indices.append(indices[-1] + elem_sizes[i])
    return indices[1:-1]


def _make_comm_fn(
    communication_type: Optional[mx.Dtype],
    op_fn: Callable,
) -> Callable:
    def comm_fn(x):
        dt = x.dtype
        x = x.astype(communication_type) if communication_type is not None else x
        return op_fn(x).astype(dt)

    return comm_fn


def _extract_info(flat: List) -> tuple:
    keys = [k for k, _ in flat]
    shapes = [v.shape for _, v in flat]
    sizes = [v.size for _, v in flat]
    dtypes = [v.dtype for _, v in flat]
    return keys, shapes, sizes, dtypes


def _grouped_comm_op(
    arrays: Any,
    group: mx.distributed.Group,
    size_threshold: int,
    communication_type: Optional[mx.Dtype],
    comm_fn: Callable,
    tree_map_fn: Callable,
    reshape_for_concat: Callable,
    get_split_sizes: Callable,
    get_target_shapes: Callable,
    concat_axis: int = 0,
    split_axis: int = 0,
    flatten_before_split: bool = False,
) -> Any:
    N = group.size()

    if N == 1:
        return arrays

    if size_threshold <= 0:
        return tree_map(tree_map_fn, arrays)

    flat = tree_flatten(arrays)

    if not flat:
        return arrays

    keys, shapes, sizes, dtypes = _extract_info(flat)

    if not all(dt == dtypes[0] for dt in dtypes):
        return _grouped_comm_op(
            arrays,
            group,
            0,
            communication_type,
            comm_fn,
            tree_map_fn,
            reshape_for_concat,
            get_split_sizes,
            get_target_shapes,
            concat_axis,
            split_axis,
            flatten_before_split,
        )

    split_sizes = get_split_sizes(sizes, N)
    target_shapes = get_target_shapes(shapes, N)

    itemsize = _get_itemsize(communication_type, dtypes[0])
    groups = _group_arrays_by_size(sizes, itemsize, size_threshold)

    new_flat = []
    for grp in groups:
        split_indices = _compute_split_indices(grp, split_sizes)

        big = mx.concatenate(
            [reshape_for_concat(flat[i][1], N) for i in grp],
            axis=concat_axis,
        )

        result = comm_fn(big)

        if flatten_before_split:
            result = result.reshape(-1)
            parts = mx.split(result, split_indices)
        else:
            parts = mx.split(result, split_indices, axis=split_axis)

        for idx_in_grp, i in enumerate(grp):
            new_flat.append((keys[i], parts[idx_in_grp].reshape(target_shapes[i])))

    return tree_unflatten(new_flat)


def average_gradients(
    gradients: Any,
    group: Optional[mx.distributed.Group] = None,
    all_reduce_size: int = 32 * 1024**2,
    communication_type: Optional[mx.Dtype] = None,
    communication_stream: Optional[mx.Stream] = None,
):
    """Average the gradients across the distributed processes in the passed group.

    This helper enables concatenating several gradients of small arrays to one
    big all reduce call for better networking performance.

    Args:
        gradients (Any): The Python tree containing the gradients (it should
            have the same structure across processes)
        group (Optional[mlx.core.distributed.Group]): The group of processes to
            average the gradients. If set to ``None`` the global group is used.
            Default: ``None``.
        all_reduce_size (int): Group arrays until their size in bytes exceeds
            this number. Perform one communication step per group of arrays. If
            less or equal to 0 array grouping is disabled. Default: ``32MiB``.
        communication_type (Optional[mlx.core.Dtype]): If provided cast to this
            type before performing the communication. Typically cast to a
            smaller float to reduce the communication size. Default: ``None``.
        communication_stream (Optional[mlx.core.Stream]): The stream to usse
            for the communication. If unspecified the default communication
            stream is used which can vary by back-end. Default: ``None``.
    """
    group = group or mx.distributed.init()
    N = group.size()

    _average = _make_comm_fn(
        communication_type,
        lambda x: mx.distributed.all_sum(x, stream=communication_stream) / N,
    )

    return _grouped_comm_op(
        arrays=gradients,
        group=group,
        size_threshold=all_reduce_size,
        communication_type=communication_type,
        comm_fn=_average,
        tree_map_fn=_average,
        reshape_for_concat=lambda v, N: v.reshape(-1),
        get_split_sizes=lambda sizes, N: sizes,
        get_target_shapes=lambda shapes, N: shapes,
    )


def reduce_scatter_gradients(
    gradients: Any,
    group: Optional[mx.distributed.Group] = None,
    reduce_scatter_size: int = 32 * 1024**2,
    communication_type: Optional[mx.Dtype] = None,
    communication_stream: Optional[mx.Stream] = None,
):
    """Reduce-scatter gradients across distributed processes.

    Similar to average_gradients but uses reduce_scatter instead of all_reduce,
    so each rank ends up with 1/N of the averaged gradients.

    Args:
        gradients (Any): The Python tree containing the gradients (it should
            have the same structure across processes)
        group (Optional[mlx.core.distributed.Group]): The group of processes to
            reduce-scatter the gradients. If set to ``None`` the global group is used.
            Default: ``None``.
        reduce_scatter_size (int): Group arrays until their size in bytes exceeds
            this number. Perform one communication step per group of arrays. If
            less or equal to 0 array grouping is disabled. Default: ``32MiB``.
        communication_type (Optional[mlx.core.Dtype]): If provided cast to this
            type before performing the communication. Typically cast to a
            smaller float to reduce the communication size. Default: ``None``.
        communication_stream (Optional[mlx.core.Stream]): The stream to use
            for the communication. If unspecified the default communication
            stream is used which can vary by back-end. Default: ``None``.
    """
    group = group or mx.distributed.init()
    N = group.size()

    _reduce_scatter = _make_comm_fn(
        communication_type,
        lambda x: mx.distributed.sum_scatter(
            x, group=group, stream=communication_stream
        )
        / N,
    )

    return _grouped_comm_op(
        arrays=gradients,
        group=group,
        size_threshold=reduce_scatter_size,
        communication_type=communication_type,
        comm_fn=_reduce_scatter,
        tree_map_fn=lambda g: _reduce_scatter(g.reshape(N, -1)).reshape(
            g.shape[0] // N, *g.shape[1:]
        ),
        reshape_for_concat=lambda v, N: v.reshape(N, -1),
        get_split_sizes=lambda sizes, N: [s // N for s in sizes],
        get_target_shapes=lambda shapes, N: [(s[0] // N, *s[1:]) for s in shapes],
        concat_axis=1,
        flatten_before_split=True,
    )


def all_gather_parameters(
    parameters_slice: Any,
    group: Optional[mx.distributed.Group] = None,
    all_gather_size: int = 32 * 1024**2,
    communication_type: Optional[mx.Dtype] = None,
    communication_stream: Optional[mx.Stream] = None,
):
    """All-gather parameters across distributed processes.

    Each rank has 1/N of the parameters, this gathers them to reconstruct
    the full parameters on each rank.

    Args:
        parameters_slice (Any): The Python tree containing the parameter slices
            (it should have the same structure across processes)
        group (Optional[mlx.core.distributed.Group]): The group of processes to
            all-gather the parameters. If set to ``None`` the global group is used.
            Default: ``None``.
        all_gather_size (int): Group arrays until their size in bytes exceeds
            this number. Perform one communication step per group of arrays. If
            less or equal to 0 array grouping is disabled. Default: ``32MiB``.
        communication_type (Optional[mlx.core.Dtype]): If provided cast to this
            type before performing the communication. Typically cast to a
            smaller float to reduce the communication size. Default: ``None``.
        communication_stream (Optional[mlx.core.Stream]): The stream to use
            for the communication. If unspecified the default communication
            stream is used which can vary by back-end. Default: ``None``.
    """
    group = group or mx.distributed.init()
    N = group.size()

    _all_gather = _make_comm_fn(
        communication_type,
        lambda x: mx.distributed.all_gather(
            x, group=group, stream=communication_stream
        ),
    )

    return _grouped_comm_op(
        arrays=parameters_slice,
        group=group,
        size_threshold=all_gather_size,
        communication_type=communication_type,
        comm_fn=lambda x: _all_gather(x.reshape(1, -1)),
        tree_map_fn=lambda p: _all_gather(p.reshape(1, -1)).reshape(
            p.shape[0] * N, *p.shape[1:]
        ),
        reshape_for_concat=lambda v, N: v.reshape(-1),
        get_split_sizes=lambda sizes, N: sizes,
        get_target_shapes=lambda shapes, N: [(s[0] * N, *s[1:]) for s in shapes],
        split_axis=1,
    )


def clip_grads_fsdp(grads_slice, max_norm):
    local_norm_sq = tree_reduce(lambda acc, g: acc + g.square().sum(), grads_slice, 0.0)
    global_norm_sq = mx.distributed.all_sum(local_norm_sq)
    grad_norm = mx.sqrt(global_norm_sq)
    normalizer = mx.minimum(max_norm / (grad_norm + 1e-6), 1.0)
    grads_slice = tree_map(lambda g: g * normalizer, grads_slice)

    return grads_slice, grad_norm
