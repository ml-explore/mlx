# Copyright © 2023 Apple Inc.

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

    def wrapped_value_grad_fn(*args, **kwargs):
        value, grad = value_grad_fn(model.trainable_parameters(), *args, **kwargs)
        return value, grad

    return wrapped_value_grad_fn


def maybe_checkpoint(fn: Callable):
    """Transform the passed function ``fn`` to a function that optionally
    performs gradient checkpointing.

    If the function's first argument is a module (always true when it is a
    method) then the checkpointing considers the module's parameters.

    To check if checkpointing needs to be applied, the transformed function
    looks at the ``checkpoint`` keyword argument or the ``checkpoint``
    attribute in case it is a module call.

    Args:
        fn (Callable): The function to be checkpointed

    Returns:
        A function that maybe applies gradient checkpointing when calling
        ``fn``
    """

    @wraps(fn)
    def checkpointable_fn(*args, **kwargs):
        checkpoint = False
        module = None
        if isinstance(args[0], Module):
            module = args[0]
            if hasattr(module, "checkpoint"):
                checkpoint = module.checkpoint
            else:
                checkpoint = kwargs.pop("checkpoint", False)
        else:
            checkpoint = kwargs.pop("checkpoint", False)

        if not checkpoint:
            return fn(*args, **kwargs)

        if module is not None:

            def pure_module_call(params, *args, **kwargs):
                module.update(params)
                return fn(module, *args, **kwargs)

            return mx.checkpoint(
                pure_module_call, module.parameters(), *args[1:], **kwargs
            )
        else:
            return mx.checkpoint(fn, *args, **kwargs)

    return checkpointable_fn
