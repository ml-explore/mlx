# Copyright © 2023-2024 Apple Inc.

import math
from typing import Callable, List, Optional

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


def cyclic_lr(
    base_lr: float,
    max_lr: float,
    step_size_up: int = 2000,
    step_size_down: Optional[int] = None,
    mode: str = "triangular",
    gamma: float = 1.0,
) -> Callable:
    r"""Make a cyclic learning rate scheduler.

    Args:
        base_lr (float): Lower learning rate boundary.
        max_lr (float): Upper learning rate boundary.
        step_size_up (int): Number of steps in the increasing half of a cycle. Default: ``2000``.
        step_size_down (int, optional): Number of steps in the decreasing half.
            If ``None``, equals ``step_size_up``. Default: ``None``.
        mode (str): One of ``"triangular"``, ``"triangular2"``, ``"exp_range"``. Default: ``"triangular"``.
        gamma (float): Scaling factor for ``"exp_range"`` mode. Default: ``1.0``.

    Example:
        >>> lr_schedule = optim.cyclic_lr(0.001, 0.1, step_size_up=10)
        >>> optimizer = optim.SGD(learning_rate=lr_schedule)
        >>> optimizer.learning_rate
        array(0.001, dtype=float32)
        >>>
        >>> for _ in range(5): optimizer.update({}, {})
        ...
        >>> optimizer.learning_rate
        array(0.0505, dtype=float32)
    """
    step_size_down = step_size_down if step_size_down is not None else step_size_up
    total_size = step_size_up + step_size_down
    step_ratio = step_size_up / total_size

    def schedule(step):
        if isinstance(step, mx.array):
            step_val = step.item() if step.size == 1 else step
        else:
            step_val = step

        cycle = math.floor(1 + step_val / total_size)
        x = 1.0 + step_val / total_size - cycle

        if x <= step_ratio:
            scale_factor = x / step_ratio
        else:
            scale_factor = (x - 1) / (step_ratio - 1)

        if mode == "triangular":
            scale_fn_val = 1.0
        elif mode == "triangular2":
            scale_fn_val = 1 / (2.0 ** (cycle - 1))
        else:  # exp_range
            scale_fn_val = gamma ** (cycle - 1)

        base_height = (max_lr - base_lr) * scale_factor
        return base_lr + base_height * scale_fn_val

    return schedule
