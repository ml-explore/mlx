# Copyright © 2023-2024 Apple Inc.

from functools import reduce, wraps
from typing import Any, Callable, Optional

import mlx.core as mx

from ..utils import tree_flatten, tree_map, tree_unflatten
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


def average_gradients(
    gradients: Any,
    group: Optional[mx.distributed.Group] = None,
    all_reduce_size: int = 32 * 1024**2,
    communication_type: Optional[mx.Dtype] = None,
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
    """
    group = group or mx.distributed.init()
    N = group.size()

    if N == 1:
        return gradients

    def _average(x):
        dt = x.dtype
        x = x.astype(communication_type) if communication_type is not None else x
        return mx.distributed.all_sum(x, stream=mx.cpu).astype(dt) / N

    if all_reduce_size <= 0:
        return tree_map(_average, gradients)

    else:
        flat_grads = tree_flatten(gradients)
        if len(flat_grads) == 0:
            return gradients

        # Extract some info for the gradient
        keys = [k for k, _ in flat_grads]
        shapes = [v.shape for _, v in flat_grads]
        sizes = [v.size for _, v in flat_grads]
        dtypes = [v.dtype for _, v in flat_grads]

        # We can't group them if they have mixed types
        if not all(dt == dtypes[0] for dt in dtypes):
            return average_gradients(gradients, group, 0, communication_type)
        itemsize = (
            communication_type.size
            if communication_type is not None
            else dtypes[0].size
        )

        # Gather the gradients in groups that are just above or equal to all_reduce_size
        grad_groups = []
        grad_group = []
        grad_group_size = 0
        for i in range(len(keys)):
            grad_group.append(i)
            grad_group_size += sizes[i] * itemsize
            if grad_group_size >= all_reduce_size:
                grad_groups.append(grad_group)
                grad_group = []
                grad_group_size = 0
        if grad_group:
            grad_groups.append(grad_group)
            grad_group = []

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
