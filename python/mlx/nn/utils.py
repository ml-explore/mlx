# Copyright © 2023-2024 Apple Inc.

from functools import reduce, wraps
from typing import Any, Callable, Optional

import mlx.core as mx

from ..utils import tree_flatten, tree_map, tree_reduce, tree_unflatten
from .layers.base import Module
from .layers.distributed import _shard


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
        communication_stream (Optional[mlx.core.Stream]): The stream to use
            for the communication. If unspecified the default communication
            stream is used which can vary by back-end. Default: ``None``.
    """
    group = group or mx.distributed.init()
    N = group.size()

    if N == 1:
        return gradients

    if all_reduce_size <= 0:
        return tree_map(
            lambda x: mx.distributed.all_sum(
                x,
                group=group,
                stream=communication_stream,
            )
            / N,
            gradients,
        )

    else:
        flat_grads = tree_flatten(gradients)
        if len(flat_grads) == 0:
            return gradients

        # Extract some info for the gradient
        keys, shapes, sizes, dtypes = _extract_info(flat_grads)

        # We can't group them if they have mixed types
        if not all(dt == dtypes[0] for dt in dtypes):
            return average_gradients(gradients, group, 0)
        # Gather the gradients in groups that are just above or equal to all_reduce_size
        grad_groups = _group_by_size(keys, sizes, dtypes[0].size, all_reduce_size)

        # Concatenate-reduce-split
        new_flat_grads = []
        for grad_group in grad_groups:
            indices = reduce(lambda x, y: x + [x[-1] + sizes[y]], grad_group, [0])
            big_grad = mx.concatenate(
                [flat_grads[i][1].reshape(-1) for i in grad_group]
            )
            big_grad = (
                mx.distributed.all_sum(
                    big_grad, stream=communication_stream, group=group
                )
                / N
            )
            big_grad = mx.split(big_grad, indices[1:-1])
            new_flat_grads.extend(
                (keys[j], big_grad[i].reshape(shapes[j]))
                for i, j in enumerate(grad_group)
            )

        return tree_unflatten(new_flat_grads)


def clip_grads_fsdp(grads_slice, max_norm, group=None):
    local_norm_sq = tree_reduce(lambda acc, g: acc + g.square().sum(), grads_slice, 0.0)
    global_norm_sq = mx.distributed.all_sum(local_norm_sq, group=group)
    grad_norm = mx.sqrt(global_norm_sq)
    normalizer = mx.minimum(max_norm / (grad_norm + 1e-6), 1.0)
    grads_slice = tree_map(lambda g: g * normalizer, grads_slice)

    return grads_slice, grad_norm


def _make_gather_fn(group, full_shapes, shard_sizes, cast_dtype):
    S = group.size()
    indices = reduce(lambda acc, w: acc + [acc[-1] + w], shard_sizes, [0])
    split_indices = indices[1:-1]
    shard_shapes = [(shape[0] // S,) + tuple(shape[1:]) for shape in full_shapes]

    def _maybe_cast(x, dtype):
        if dtype is None or x.dtype == dtype:
            return x
        return x.astype(dtype)

    @mx.custom_function
    def gather(shards):
        big_shard = mx.concatenate(
            [_maybe_cast(s.reshape(1, -1), cast_dtype) for s in shards], axis=1
        )
        big_full = mx.distributed.all_gather(big_shard, group=group)
        parts = mx.split(big_full, split_indices, axis=1)
        return [p.reshape(shape) for p, shape in zip(parts, full_shapes)]

    @gather.vjp
    def gather_vjp(shards, cotangents, _):
        big_cot_full = mx.concatenate([c.reshape(S, -1) for c in cotangents], axis=1)
        big_cot_shard = mx.distributed.sum_scatter(big_cot_full, group=group) / S
        parts = mx.split(big_cot_shard, split_indices, axis=1)
        return [p.reshape(shape) for p, shape in zip(parts, shard_shapes)]

    return gather


def _maybe_shard(m, k, v):
    if isinstance(v, FullyShardedModule):
        return False
    return Module.valid_parameter_filter(m, k, v)


class FullyShardedModule(Module):
    def __init__(self, module, group, cast_dtype):
        super().__init__()
        group = group or mx.distributed.init()
        N = group.size()

        shard_params = module.filter_and_map(_maybe_shard)
        flat = tree_flatten(shard_params)
        for path, a in flat:
            if a.ndim == 0:
                raise ValueError(
                    f"FSDP: parameter {path} is a 0-D scalar and cannot be sharded."
                )
            if a.shape[0] % N != 0:
                raise ValueError(
                    f"FSDP: parameter {path} has shape {a.shape}; axis 0 must "
                    f"be divisible by the FSDP group size {N}."
                )

        self._paths = [k for k, _ in flat]
        full_shapes = [a.shape for _, a in flat]
        shard_sizes = [a.size // N for _, a in flat]

        module.update(_shard(shard_params, lambda p, w: 0, group))

        self.module = module
        self._gather_fn = _make_gather_fn(group, full_shapes, shard_sizes, cast_dtype)

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
    group: Optional["mx.distributed.Group"] = None,
    cast_dtype: Optional[mx.Dtype] = None,
) -> Module:
    group = group or mx.distributed.init()
    if group.size() == 1:
        return module
    if isinstance(module, FullyShardedModule):
        return module

    wrapped = FullyShardedModule(module, group, cast_dtype)
    return wrapped if wrapped._paths else module
