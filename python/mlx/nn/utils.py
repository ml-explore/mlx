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
    """Transform the passed module to one that performs gradient checkpointing.

    The checkpointing is with respect to the module's trainable parameters and
    inputs of the module's ``__call__`` function.

    Args:
        module (mlx.nn.Module): The module for which we will perform gradient
          checkpointing.

    Returns:
        A new module that saves the inputs and outputs during the forward pass
        and recomputes all intermediate states during the backward pass.
    """

    t = type(module)
    cp_name = f"__checkpointed_{id(t)}__"
    cp_class = globals().get(cp_name, None)
    if cp_class is None:

        def init(self):
            pass

        def call(self, *args, **kwargs):
            return self.__checkpointed_call__(
                self.trainable_parameters(), *args, **kwargs
            )

        cp_class = type(t.__name__, (t,), {})
        cp_class.__init__ = init
        cp_class.__call__ = call
        globals()[cp_name] = cp_class

    cp_module = cp_class()
    cp_module.__dict__.update(module.__dict__)
    super(Module, cp_module).update(module.state)

    def inner_fn(params, *args, **kwargs):
        module.update(params)
        return module(*args, **kwargs)

    cp_module.__checkpointed_call__ = mx.checkpoint(inner_fn)
    return cp_module
