# Copyright © 2023-2024 Apple Inc.

from functools import wraps
from typing import Callable

import mlx.core as mx

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


def checkpoint(module: Module):
    """Transform the passed module to one that performs gradient
    checkpointing.

    The checkpointing is with respect to the module's trainable parameters and
    inputs of the module's ``__call__`` function.

    Args:
        module (mlx.nn.Module): The module for whose parameters we will be
            performing gradient checkpointing.

    Returns:
        The module that saves the inputs and outputs during the forward pass
        and recomputes all intermediate states during the backward pass.
    """

    fn = module.__call__

    def inner_fn(params, *args, **kwargs):
        module.update(params)
        return fn(*args, **kwargs)

    checkpointed_fn = mx.checkpoint(inner_fn)

    @wraps(fn)
    def wrapped_checkpointed_fn(*args, **kwargs):
        return checkpointed_fn(module.trainable_parameters(), *args, **kwargs)

    class _(type(module)):
        def __call__(self, *arg, **kwarg):
            return wrapped_checkpointed_fn(*arg, **kwarg)

    module.__class__ = _
    return module
