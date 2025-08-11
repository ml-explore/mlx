# Copyright © 2023-2024 Apple Inc.

import math
from typing import Callable, List

import mlx.core as mx


def exponential_decay(init: float, decay_rate: float) -> Callable:
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


def step_decay(init: float, decay_rate: float, step_size: int) -> Callable:
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


def cosine_decay(init: float, decay_steps: int, end: float = 0.0) -> Callable:
    r"""Make a cosine decay scheduler.

    Args:
        init (float): Initial value.
        decay_steps (int): Number of steps to decay over. The decayed
            value is constant for steps beyond ``decay_steps``.
        end (float, optional): Final value to decay to. Default: ``0``.

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

    def schedule(step):
        s = mx.minimum(step, decay_steps)
        decay = 0.5 * (1.0 + mx.cos((math.pi / decay_steps) * s))
        return end + decay * (init - end)

    return schedule


def join_schedules(schedules: List[Callable], boundaries: List[int]) -> Callable:
    r"""Join multiple schedules to create a new schedule.

    Args:
        schedules (list(Callable)): A list of schedules. Schedule :math:`i+1`
          receives a step count indicating the number of steps since
          the :math:`i`-th boundary.
        boundaries (list(int)): A list of integers of length ``len(schedules) - 1``
          that indicates when to transition between schedules.

    Example:
        >>> linear = optim.linear_schedule(0, 1e-1, steps=10)
        >>> cosine = optim.cosine_decay(1e-1, 200)
        >>> lr_schedule = optim.join_schedules([linear, cosine], [10])
        >>> optimizer = optim.Adam(learning_rate=lr_schedule)
        >>> optimizer.learning_rate
        array(0.0, dtype=float32)
        >>> for _ in range(12): optimizer.update({}, {})
        ...
        >>> optimizer.learning_rate
        array(0.0999938, dtype=float32)
    """
    if len(schedules) == 0:
        raise ValueError("Must provide at least 1 schedule to join.")

    if len(schedules) != len(boundaries) + 1:
        raise ValueError(
            f"Received {len(boundaries)} boundaries but "
            f"expected {len(schedules) - 1}."
        )

    def schedule(step):
        output = schedules[0](step)
        for boundary, schedule in zip(boundaries, schedules[1:]):
            output = mx.where(step < boundary, output, schedule(step - boundary))
        return output

    return schedule


def linear_schedule(init: float, end: float, steps: int) -> Callable:
    r"""Make a linear scheduler.

    Args:
        init (float): Initial value.
        end (float): Final value.
        steps (int): Number of steps to apply the schedule over. The value is
          ``end`` for any steps beyond ``steps``.

    Example:

        >>> lr_schedule = optim.linear_schedule(0, 1e-1, 100)
        >>> optimizer = optim.Adam(learning_rate=lr_schedule)
        >>> optimizer.learning_rate
        array(0.0, dtype=float32)
        >>> for _ in range(101): optimizer.update({}, {})
        ...
        >>> optimizer.learning_rate
        array(0.1, dtype=float32)
    """
    if steps < 1:
        raise ValueError(f"steps must be greater than 0, but got {steps}.")

    def schedule(step):
        step = mx.minimum(step, steps)
        return step * ((end - init) / steps) + init

    return schedule


def cosine_annealing_warm_restarts(
    init: float, T_0: int, T_mult: int = 1, eta_min: float = 0.0
) -> Callable:
    r"""Make a cosine annealing scheduler with warm restarts.

    The learning rate anneals using a cosine schedule and resets periodically.

    Args:
        init (float): Initial learning rate.
        T_0 (int): Number of steps for the first restart.
        T_mult (int, optional): Factor to increase the period after each restart. Default: ``1``.
        eta_min (float, optional): Minimum learning rate. Default: ``0.0``.

    Example:
        >>> lr_schedule = optim.cosine_annealing_warm_restarts(1e-1, T_0=10)
        >>> optimizer = optim.SGD(learning_rate=lr_schedule)
        >>> optimizer.learning_rate
        array(0.1, dtype=float32)
        >>>
        >>> for _ in range(11): optimizer.update({}, {})
        ...
        >>> optimizer.learning_rate
        array(0.1, dtype=float32)
    """
    if T_0 < 1:
        raise ValueError(f"T_0 must be at least 1, got {T_0}")
    if T_mult < 1:
        raise ValueError(f"T_mult must be at least 1, got {T_mult}")
    if eta_min < 0:
        raise ValueError(f"eta_min must be non-negative, got {eta_min}")
    if init < eta_min:
        raise ValueError(f"init must be >= eta_min, got init={init}, eta_min={eta_min}")

    def schedule(step):
        if isinstance(step, mx.array):
            step_val = step.item() if step.size == 1 else step
        else:
            step_val = step

        if T_mult == 1:
            T_cur = step_val % T_0
            T_i = T_0
        else:
            if step_val >= T_0:
                n = int(math.log((step_val / T_0 * (T_mult - 1) + 1), T_mult))
                T_cur = step_val - T_0 * (T_mult**n - 1) / (T_mult - 1)
                T_i = T_0 * T_mult**n
            else:
                T_i = T_0
                T_cur = step_val

        cos_inner = math.pi * T_cur / T_i
        return eta_min + (init - eta_min) * 0.5 * (1.0 + mx.cos(cos_inner))

    return schedule
