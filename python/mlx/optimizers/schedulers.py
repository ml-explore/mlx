# Copyright © 2023-2024 Apple Inc.

import math

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
