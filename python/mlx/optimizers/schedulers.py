# Copyright © 2023-2024 Apple Inc.

import math
from typing import Callable

import mlx.core as mx


def exponential_decay(init: float, decay_rate: float):
    r"""Make an exponential decay scheduler.

    Args:
        init (float): Initial value.
        decay_rate (float): Multiplicative factor to decay by.

    Example:
        >>> lr_schedule = optim.exponential_decay(1e-1, 0.9)
        >>> optimizer = optim.SGD(learning_rate=lr_schedule)
        >>> optimizer.learning_rate
        array(0.1, dtype=float32)
        >>>
        >>> for _ in range(5): optimizer.update({}, {})
        ...
        >>> optimizer.learning_rate
        array(0.06561, dtype=float32)
    """

    def schedule(step):
        return init * decay_rate**step

    return schedule


def step_decay(init: float, decay_rate: float, step_size: int):
    r"""Make a step decay scheduler.

    Args:
        init (float): Initial value.
        decay_rate (float): Multiplicative factor to decay by.
        step_size (int): Decay every ``step_size`` steps.

    Example:

        >>> lr_schedule = optim.step_decay(1e-1, 0.9, 10)
        >>> optimizer = optim.SGD(learning_rate=lr_schedule)
        >>> optimizer.learning_rate
        array(0.1, dtype=float32)
        >>>
        >>> for _ in range(21): optimizer.update({}, {})
        ...
        >>> optimizer.learning_rate
        array(0.081, dtype=float32)
    """

    def schedule(step):
        return init * (decay_rate ** (step // step_size))

    return schedule


def cosine_decay(init: float, decay_steps: int):
    r"""Make a cosine decay scheduler.

    Args:
        init (float): Initial value.
        decay_steps (int): Number of steps to decay over. The decayed
            value is constant for steps beyond ``decay_steps``.

    Example:

        >>> lr_schedule = optim.cosine_decay(1e-1, 1000)
        >>> optimizer = optim.SGD(learning_rate=lr_schedule)
        >>> optimizer.learning_rate
        array(0.1, dtype=float32)
        >>>
        >>> for _ in range(5): optimizer.update({}, {})
        ...
        >>> optimizer.learning_rate
        array(0.0999961, dtype=float32)
    """

    def scheduler(step):
        s = mx.minimum(step, decay_steps)
        decay = 0.5 * (1.0 + mx.cos((math.pi / decay_steps) * s))
        return init * decay

    return scheduler


def linear_warmup(
    schedule_fn, length: int, init: float = 0.0, finish: float = 0.0
) -> Callable:
    """
    >>> lr_schedule = optim.cosine_with_warmup(100, finish_lr=1e-1)
    >>> optimizer = optim.Adam(learning_rate=lr_schedule)
    >>> optimizer.learning_rate
    array(0.0, dtype=float32)
    >>> for _ in range(100): optimizer.update({}, {})
    ...
    >>> optimizer.learning_rate
    array(0.1, dtype=float32)

    :param schedule_fn: other schedule function, such as cosine_decay,step_decay, etc.
    :param length: warmup length
    :param init: start value (0 by default)
    :param finish: final value
    :return: callable takes a step and returns the schedulable
    """

    def schedule_step_fn(step):
        if step <= length:
            return step * ((finish - init) / length) + init
        else:
            offset_idx = step.item() - length
            return schedule_fn(offset_idx)

    return schedule_step_fn
