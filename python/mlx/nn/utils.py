# Copyright © 2023-2024 Apple Inc.

from functools import reduce, wraps
from typing import Any, Callable, Optional

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


def _extract_info(flat):
    keys = [k for k, _ in flat]
    shapes = [g.shape for _, g in flat]
    sizes = [g.size for _, g in flat]
    dtypes = [g.dtype for _, g in flat]
    return keys, shapes, sizes, dtypes


def _group_by_size(keys, sizes, itemsize, communication_size):
    grad_groups = []
    grad_group = []
    grad_group_size = 0
    for i in range(len(keys)):
        grad_group.append(i)
        grad_group_size += sizes[i] * itemsize
        if grad_group_size >= communication_size:
            grad_groups.append(grad_group)
            grad_group = []
            grad_group_size = 0
    if grad_group:
        grad_groups.append(grad_group)
        grad_group = []
    return grad_groups


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
        communication_stream (Optional[mlx.core.Stream]): The stream to use
            for the communication. If unspecified the default communication
            stream is used which can vary by back-end. Default: ``None``.
    """
    group = group or mx.distributed.init()
    N = group.size()

    if N == 1:
        return gradients

    def _average(x):
        dt = x.dtype
        x = x.astype(communication_type) if communication_type is not None else x
        return mx.distributed.all_sum(x, stream=communication_stream).astype(dt) / N

    if all_reduce_size <= 0:
        return tree_map(_average, gradients)

    else:
        flat_grads = tree_flatten(gradients)
        if len(flat_grads) == 0:
            return gradients

        # Extract some info for the gradient
        keys, shapes, sizes, dtypes = _extract_info(flat_grads)

        # We can't group them if they have mixed types
        if not all(dt == dtypes[0] for dt in dtypes):
            return average_gradients(gradients, group, 0, communication_type)
        itemsize = (
            communication_type.size
            if communication_type is not None
            else dtypes[0].size
        )

        # Gather the gradients in groups that are just above or equal to all_reduce_size
        grad_groups = _group_by_size(keys, sizes, itemsize, all_reduce_size)

        # Concatenate-reduce-split
        new_flat_grads = []
        for grad_group in grad_groups:
            indices = reduce(lambda x, y: x + [x[-1] + sizes[y]], grad_group, [0])
            big_grad = mx.concatenate(
                [flat_grads[i][1].reshape(-1) for i in grad_group]
            )
            big_grad = _average(big_grad)
            big_grad = mx.split(big_grad, indices[1:-1])
            new_flat_grads.extend(
                (keys[j], big_grad[i].reshape(shapes[j]))
                for i, j in enumerate(grad_group)
            )

        return tree_unflatten(new_flat_grads)


def _clip_grads_fsdp(grads_slice, max_norm):
    local_norm_sq = tree_reduce(lambda acc, g: acc + g.square().sum(), grads_slice, 0.0)
    global_norm_sq = mx.distributed.all_sum(local_norm_sq)
    grad_norm = mx.sqrt(global_norm_sq)
    normalizer = mx.minimum(max_norm / (grad_norm + 1e-6), 1.0)
    grads_slice = tree_map(lambda g: g * normalizer, grads_slice)

    return grads_slice, grad_norm


def fsdp_apply_gradients(
    gradients,
    parameters,
    optimizer,
    group=None,
    communication_size=32 * 1024**2,
    communication_type=None,
    communication_stream=None,
    max_norm=None,
):
    """Perform a distributed optimizer step by sharding gradients and optimizer states across ranks.

    This helper function performs the following steps:
    1. Reduce-scatter the gradients across ranks so each rank gets a shard of the averaged gradients.
    2. Optionally clip the sharded gradients by global norm.
    3. Apply the optimizer update on the local parameter slice using the sharded gradients.
    4. All-gather the updated parameter slices from all ranks to reconstruct the full parameters tree.

    This is similar to PyTorch's FSDP with `reshard_after_forward=False`.

    Args:
        gradients (Any): The Python tree containing the full gradients (it should
            have the same structure as ``parameters``). Each gradient's first
            dimension must be divisible by the world size.
        parameters (Any): The Python tree containing the full parameters (it should
            have the same structure across processes). Each parameter's first
            dimension must be divisible by the world size.
        optimizer: Optimizer with an ``apply_gradients`` method.
        group (Optional[mlx.core.distributed.Group]): The group of processes for
            communication. If ``None``, the global group is used.
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
            maximum global norm before applying the optimizer update.
            Default: ``None``.

    Returns:
        If ``max_norm`` is ``None``, returns the updated full-parameter tree.
        Otherwise returns ``(parameters, grad_norm)``, where ``grad_norm`` is
        the global gradient norm before clipping.

    Example:

        >>> optimizer = optim.SGD(learning_rate=0.01)
        >>> # Without gradient clipping
        >>> updated_params = fsdp_apply_gradients(params, grads, optimizer)
        >>> model.update(updated_params)
        >>>
        >>> # With gradient clipping
        >>> updated_params, grad_norm = fsdp_apply_gradients(
        ...     params, grads, optimizer, max_norm=1.0
        ... )
        >>> model.update(updated_params)
    """
    group = group or mx.distributed.init()
    N = group.size()
    rank = group.rank()

    if N == 1:
        if max_norm is not None:
            gradients, grad_norm = _clip_grads_fsdp(gradients, max_norm)
            return optimizer.apply_gradients(gradients, parameters), grad_norm
        return optimizer.apply_gradients(gradients, parameters)

    flat_grads = tree_flatten(gradients)
    flat_params = tree_flatten(parameters)

    def _sum_scatter(x):
        dt = x.dtype
        x = x.astype(communication_type) if communication_type is not None else x
        return (
            mx.distributed.sum_scatter(
                x, group=group, stream=communication_stream
            ).astype(dt)
            / N
        )

    def _all_gather(x):
        dt = x.dtype
        x = x.astype(communication_type) if communication_type is not None else x
        return mx.distributed.all_gather(
            x, group=group, stream=communication_stream
        ).astype(dt)

    keys, shapes, sizes, dtypes = _extract_info(flat_grads)
    itemsize = dtypes[0].size

    groups = _group_by_size(keys, sizes, itemsize, communication_size)

    # reduce-scatter gradients, shard parameters
    grad_slices = {}
    param_slices = {}
    for group_idx, arr_group in enumerate(groups):
        big_grad = mx.concatenate(
            [flat_grads[i][1].reshape(N, -1) for i in arr_group], axis=1
        )
        grad_slices[group_idx] = _sum_scatter(big_grad)
        big_param = mx.concatenate(
            [flat_params[i][1].reshape(N, -1) for i in arr_group], axis=1
        )
        param_slices[group_idx] = big_param[rank]

    # clip gradients if needed
    grad_norm = None
    if max_norm is not None:
        grad_slices, grad_norm = _clip_grads_fsdp(grad_slices, max_norm)

    # optimizer step
    updated_param_slices = optimizer.apply_gradients(grad_slices, param_slices)

    # all-gather and reconstruct
    new_flat = []
    for group_idx, arr_group in enumerate(groups):
        big_gathered = _all_gather(updated_param_slices[group_idx].reshape(1, -1))

        split_sizes = [sizes[i] // N for i in arr_group]
        split_indices = []
        acc = 0
        for s in split_sizes:
            acc += s
            split_indices.append(acc)

        parts = mx.split(big_gathered, split_indices[:-1], axis=1)
        for idx_in_group, i in enumerate(arr_group):
            new_flat.append((keys[i], parts[idx_in_group].reshape(shapes[i])))

    result = tree_unflatten(new_flat)
    if max_norm is not None:
        return result, grad_norm
    return result
