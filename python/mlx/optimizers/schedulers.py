# Copyright © 2023-2024 Apple Inc.

import math
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

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


class ScheduleJoiner:
    r"""Make a schedule joiner

    Instanciated with a list of schedules and an increasing list of 0-based steps, each of which indicates
    when the next schedule will be used.  The instanciated object is a callable schedule that is a
    concatenation of the schedules at the given step boundaries

    So, the first item in the schedule list will be used until the first boundary, then the next schedule
    will be used until the next boundary, and so on.

    Args:
        schedules (list of schedules): The schedules to join
        boundaries (list of integers): The list of steps that mark the use of the next schedule (in order)

    Example:
        >>> warmup_schedule = optim.linear_warmup(100, finish=1e-1)
        >>> cosine_schedule = optim.cosine_decay(1e-1, 200)
        >>> lr_schedule = ScheduleJoiner([warmup_schedule, cosine_schedule], [101])
        >>> optimizer = optim.Adam(learning_rate=lr_schedule)
        >>> optimizer.learning_rate
        array(0.0, dtype=float32)
        >>> for _ in range(101): optimizer.update({}, {})
        ...
        >>> optimizer.learning_rate
        array(1e-5, dtype=float32)

    """

    def __init__(self, schedules: List[Callable], boundaries: List[int]):
        self.schedules = schedules
        self.boundaries = boundaries

    def __call__(self, step: int):
        if step < self.boundaries[0] or not self.boundaries:
            current_schedule = self.schedules[0]
            updated_step = step
        else:
            curr_sched_idx = -1
            for idx, boundary in filter(
                lambda i: step <= i[-1], enumerate(self.boundaries)
            ):
                if step == boundary:
                    curr_sched_idx = idx + 1
                    current_schedule = self.schedules[curr_sched_idx]
                    updated_step = 0
                    break
                elif boundary > step:
                    curr_sched_idx = idx
                    current_schedule = self.schedules[curr_sched_idx]
                    updated_step = boundary - step
                    break
            if curr_sched_idx == -1:
                curr_sched_idx = len(self.boundaries)
                current_schedule = self.schedules[curr_sched_idx]
                updated_step = step - self.boundaries[-1]
        return current_schedule(updated_step)


def linear_warmup(length: int, finish: float, init: float = 0.0) -> Callable:
    r"""Make a linear warmup scheduler.

    Args:
        length (int): Length of warmup.
        finish (float): Value at the end of the warmup.
        init (float): Initial value.

    Example:

        >>> lr_schedule = optim.linear_warmup(100, 1e-1)
        >>> optimizer = optim.Adam(learning_rate=lr_schedule)
        >>> optimizer.learning_rate
        array(0.0, dtype=float32)
        >>> for _ in range(100): optimizer.update({}, {})
        ...
        >>> optimizer.learning_rate
        array(1e-5, dtype=float32)
    """

    def step_fn(step):
        return step * ((finish - init) / length) + init

    return step_fn
