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


def _make_comm_fn(
    communication_type: Optional[mx.Dtype],
    op_fn: Callable,
) -> Callable:

    def comm_fn(x):
        dt = x.dtype
        x = x.astype(communication_type) if communication_type is not None else x
        return op_fn(x).astype(dt)

    return comm_fn


def _comm_op(
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
    split_axis: Optional[int] = 0,
) -> Any:
    N = group.size()

    if N == 1:
        return arrays

    if size_threshold <= 0:
        return tree_map(tree_map_fn, arrays)

    flat = tree_flatten(arrays)

    if not flat:
        return arrays

    keys = [k for k, _ in flat]
    shapes = [v.shape for _, v in flat]
    sizes = [v.size for _, v in flat]
    dtypes = [v.dtype for _, v in flat]

    if not all(dt == dtypes[0] for dt in dtypes):
        return _comm_op(
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
        )

    split_sizes = get_split_sizes(sizes, N)
    target_shapes = get_target_shapes(shapes, N)

    itemsize = (
        communication_type.size if communication_type is not None else dtypes[0].size
    )
    groups = _group_arrays_by_size(sizes, itemsize, size_threshold)

    new_flat = []
    for group in groups:
        split_indices = [0]
        for i in group:
            split_indices.append(split_indices[-1] + split_sizes[i])
        split_indices = split_indices[1:-1]

        big = mx.concatenate(
            [reshape_for_concat(flat[i][1], N) for i in group],
            axis=concat_axis,
        )

        result = comm_fn(big)

        if split_axis is None:
            result = result.reshape(-1)
            parts = mx.split(result, split_indices)
        else:
            parts = mx.split(result, split_indices, axis=split_axis)

        for idx_in_group, i in enumerate(group):
            new_flat.append((keys[i], parts[idx_in_group].reshape(target_shapes[i])))

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

    return _comm_op(
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

    return _comm_op(
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
        split_axis=None,
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

    return _comm_op(
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


def _clip_grad_norm_fsdp(grads_slice, max_norm):
    """Compute global gradient norm and clip gradients.

    Args:
        grads_slice: Gradient slices (each rank has 1/N of full gradients)
        max_norm: Maximum gradient norm

    Returns:
        Tuple of (clipped_grads_slice, grad_norm)
    """
    local_norm_sq = tree_reduce(lambda acc, g: acc + g.square().sum(), grads_slice, 0.0)
    global_norm_sq = mx.distributed.all_sum(local_norm_sq)
    grad_norm = mx.sqrt(global_norm_sq)
    normalizer = mx.minimum(max_norm / (grad_norm + 1e-6), 1.0)
    clipped = tree_map(lambda g: g * normalizer, grads_slice)
    return clipped, grad_norm


def fsdp_update_params(
    parameters,
    gradients,
    optimizer,
    group: Optional[mx.distributed.Group] = None,
    communication_size: int = 32 * 1024**2,
    communication_type: Optional[mx.Dtype] = None,
    communication_stream: Optional[mx.Stream] = None,
    max_norm: Optional[float] = None,
):
    """Update parameters using FSDP (Fully Sharded Data Parallel) strategy.

    This function performs the following steps:

    1. Sum-scatter gradients across ranks (each rank gets 1/N of averaged gradients)
    2. Optionally clip gradients by global norm
    3. Apply optimizer update on local parameter slice
    4. All-gather updated parameters to reconstruct full parameters on each rank

    Note: There is currently no option to shard model parameters during the
    forward/backward passes. Each rank holds a full parameter copy. This is equivalent
    to PyTorch's ``torch.distributed.fsdp.fully_shard(model, reshard_after_forward=False)``.

    All parameters must have their first dimension divisible by the world size.

    Args:
        parameters (Any): The Python tree containing the full parameters (same
            structure across processes). Each parameter's first dimension must
            be divisible by the world size.
        gradients (Any): The Python tree containing the full gradients (same
            structure as parameters)
        optimizer: Optimizer with an apply_gradients method
        group (Optional[mlx.core.distributed.Group]): The group of processes for
            communication. If set to ``None`` the global group is used.
            Default: ``None``.
        communication_size (int): Group arrays until their size in bytes exceeds
            this number. Perform one communication step per group of arrays. If
            less or equal to 0 array grouping is disabled. Default: ``32MiB``.
        communication_type (Optional[mlx.core.Dtype]): If provided cast to this
            type before performing the communication. Typically cast to a
            smaller float to reduce the communication size. Default: ``None``.
        communication_stream (Optional[mlx.core.Stream]): The stream to use
            for the communication. If unspecified the default communication
            stream is used which can vary by back-end. Default: ``None``.
        max_norm (Optional[float]): If provided, clip gradients to this
            maximum global norm before applying optimizer update. Default: ``None``.

    Returns:
        The updated parameters tree. If ``max_norm`` is provided, returns
        a tuple ``(parameters, grad_norm)`` where ``grad_norm`` is the global
        gradient norm before clipping.

    Example:

        >>> optimizer = optim.SGD(learning_rate=0.01)
        >>> # Without gradient clipping
        >>> updated_params = fsdp_update_params(params, grads, optimizer)
        >>>
        >>> # With gradient clipping
        >>> updated_params, grad_norm = fsdp_update_params(
        ...     params, grads, optimizer, max_norm=1.0
        ... )
    """
    group = group or mx.distributed.init()
    rank = group.rank()
    world_size = group.size()

    # Sum-scatter gradients so each rank gets 1/N of the averaged gradients
    grads_slice = reduce_scatter_gradients(
        gradients,
        group=group,
        reduce_scatter_size=communication_size,
        communication_type=communication_type,
        communication_stream=communication_stream,
    )

    # Optionally clip gradients
    grad_norm = None
    if max_norm is not None:
        grads_slice, grad_norm = _clip_grad_norm_fsdp(grads_slice, max_norm)

    # Extract local parameter slice (each rank owns 1/N of parameters)
    params_slice = tree_map(
        lambda x: x.reshape(world_size, x.shape[0] // world_size, *x.shape[1:])[rank],
        parameters,
    )

    # Apply optimizer update on local parameter slice
    params_slice = optimizer.apply_gradients(grads_slice, params_slice)

    # All-gather updated parameters from all ranks
    params = all_gather_parameters(
        params_slice,
        group=group,
        all_gather_size=communication_size,
        communication_type=communication_type,
        communication_stream=communication_stream,
    )

    return (params, grad_norm) if max_norm is not None else params
