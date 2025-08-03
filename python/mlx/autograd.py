# Copyright © 2023-2024 Apple Inc.

"""Gradient computation control utilities for MLX."""

from typing import Any, Callable, TypeVar

import mlx.core as mx

F = TypeVar("F", bound=Callable[..., Any])

# Export the core functions
is_grad_enabled = mx.is_grad_enabled
set_grad_enabled = mx.set_grad_enabled

__all__ = [
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "is_grad_enabled",
]


class _NoParamDecoratorContextManager:
    """Base class for context managers that can also be used as decorators."""

    def __call__(self, func: F) -> F:
        """Decorator usage."""

        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


class no_grad(_NoParamDecoratorContextManager):
    r"""Context manager that disables gradient calculation.

    Disabling gradient calculation is useful for inference, when you are sure
    that you will not call backward passes. It will reduce memory consumption
    for computations that would otherwise compute gradients.

    In this mode, gradient computation will be disabled for all operations.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    Example::
        >>> import mlx.core as mx
        >>> x = mx.array([1.], requires_grad=True)  # MLX doesn't have requires_grad, but for illustration
        >>> with mx.no_grad():
        ...     y = x * 2
        >>> # y won't have gradients computed for it
        >>> @mx.no_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> z = doubler(x)
        >>> # z won't have gradients computed for it
    """

    def __init__(self) -> None:
        self.prev = False

    def __enter__(self) -> None:
        self.prev = mx.is_grad_enabled()
        mx.set_grad_enabled(False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        mx.set_grad_enabled(self.prev)


class enable_grad(_NoParamDecoratorContextManager):
    r"""Context manager that enables gradient calculation.

    Enables gradient calculation, if it has been disabled via :class:`~no_grad`
    or :class:`~set_grad_enabled`.

    This context manager is thread local; it will not affect computation
    in other threads.

    Also functions as a decorator.

    Example::
        >>> import mlx.core as mx
        >>> x = mx.array([1.])
        >>> with mx.no_grad():
        ...     with mx.enable_grad():
        ...         y = x * 2
        >>> # y will have gradients computed for it
        >>> @mx.enable_grad()
        ... def doubler(x):
        ...     return x * 2
        >>> with mx.no_grad():
        ...     z = doubler(x)
        >>> # z will have gradients computed for it
    """

    def __init__(self) -> None:
        self.prev = False

    def __enter__(self) -> None:
        self.prev = mx.is_grad_enabled()
        mx.set_grad_enabled(True)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        mx.set_grad_enabled(self.prev)


class set_grad_enabled:
    r"""Context manager that sets gradient calculation on or off.

    ``set_grad_enabled`` will enable or disable grads based on its argument :attr:`mode`.
    It can be used as a context manager or as a function.

    This context manager is thread local; it will not affect computation
    in other threads.

    Args:
        mode (bool): Flag whether to enable grad (``True``), or disable
                     (``False``). This can be used to conditionally enable
                     gradients.

    Example::
        >>> import mlx.core as mx
        >>> x = mx.array([1.])
        >>> is_train = False
        >>> with mx.set_grad_enabled(is_train):
        ...     y = x * 2
        >>> # y won't have gradients computed
        >>> _ = mx.set_grad_enabled(True)
        >>> y = x * 2
        >>> # y will have gradients computed
        >>> _ = mx.set_grad_enabled(False)
        >>> y = x * 2
        >>> # y won't have gradients computed
    """

    def __init__(self, mode: bool) -> None:
        self.prev = mx.is_grad_enabled()
        self.mode = mode
        mx.set_grad_enabled(mode)

    def __call__(self, func: F) -> F:
        """Decorator usage."""
        mx.set_grad_enabled(self.prev)

        def wrapper(*args, **kwargs):
            with set_grad_enabled(self.mode):
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self) -> None:
        mx.set_grad_enabled(self.mode)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        mx.set_grad_enabled(self.prev)

    def __str__(self) -> str:
        return f"set_grad_enabled(mode={self.mode})"

    def __repr__(self) -> str:
        return str(self)

    def clone(self) -> "set_grad_enabled":
        """Create a copy of this class."""
        return self.__class__(self.mode)
