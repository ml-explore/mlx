# Copyright © 2023-2024 Apple Inc.

import math
from typing import Callable, List, Optional, Tuple

import mlx.core as mx
from mlx.optimizers import Optimizer
from mlx.utils import tree_map


class LRScheduler:
    r"""Base class for learning rate schedulers.

    Args:
        optimizer (Optimizer): The optimizer for which to adjust the learning rate.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lr = optimizer.learning_rate
        self.step(last_epoch)

    def get_lr(self) -> float:
        raise NotImplementedError

    def step(self, epoch: int = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.optimizer.learning_rate = self.get_lr()


class StepLR(LRScheduler):
    r"""Decays the learning rate by a factor of gamma every step_size epochs.

    Args:
        optimizer (Optimizer): The optimizer for which to adjust the learning rate.
        step_size (int): Period of learning rate decay.
        gamma (float, optional): Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        super().__init__(optimizer, last_epoch)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self) -> float:
        return self.base_lr * self.gamma ** (self.last_epoch // self.step_size)


class ExponentialLR(LRScheduler):
    r"""Decays the learning rate exponentially.

    Args:
        optimizer (Optimizer): The optimizer for which to adjust the learning rate.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(self, optimizer: Optimizer, gamma: float, last_epoch: int = -1):
        super().__init__(optimizer, last_epoch)
        self.gamma = gamma

    def get_lr(self) -> float:
        return self.base_lr * self.gamma**self.last_epoch


class MultiStepLR(LRScheduler):
    r"""Decays the learning rate by a factor of gamma at specified milestones.

    Args:
        optimizer (Optimizer): The optimizer for which to adjust the learning rate.
        milestones (List[int]): List of epoch indices. Must be increasing.
        gamma (float, optional): Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        super().__init__(optimizer, last_epoch)
        self.milestones = sorted(milestones)
        self.gamma = gamma

    def get_lr(self) -> float:
        factor = self.gamma ** sum(
            self.last_epoch >= milestone for milestone in self.milestones
        )
        return self.base_lr * factor


class LambdaLR(LRScheduler):
    r"""Decays the learning rate using a user-defined lambda function.

    Args:
        optimizer (Optimizer): The optimizer for which to adjust the learning rate.
        lr_lambda (Callable): A function defining the decay factor.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Callable,
        last_epoch: int = -1,
    ):
        super().__init__(optimizer, last_epoch)
        self.lr_lambda = lr_lambda

    def get_lr(self) -> float:
        return self.base_lr * self.lr_lambda(self.last_epoch)


class PolynomialLR(LRScheduler):
    r"""Decays the learning rate in a polynomial manner.

    Args:
        optimizer (Optimizer): The optimizer for which to adjust the learning rate.
        max_decay_steps (int): The maximum number of decay steps.
        end_lr (float): The end learning rate.
        power (float): The power of the polynomial decay.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_decay_steps: int,
        end_lr: float,
        power: float,
        last_epoch: int = -1,
    ):
        super().__init__(optimizer, last_epoch)
        self.max_decay_steps = max_decay_steps
        self.end_lr = end_lr
        self.power = power

    def get_lr(self) -> float:
        decay_steps = min(self.last_epoch, self.max_decay_steps)
        decay_factor = (1 - decay_steps / self.max_decay_steps) ** self.power
        return (self.base_lr - self.end_lr) * decay_factor + self.end_lr


class CosineAnnealingLR(LRScheduler):
    r"""Decays the learning rate using a cosine annealing schedule.

    Args:
        optimizer (Optimizer): The optimizer for which to adjust the learning rate.
        T_max (int): The maximum number of iterations.
        eta_min (float, optional): The minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(
        self, optimizer: Optimizer, T_max: int, eta_min: float = 0, last_epoch: int = -1
    ):
        super().__init__(optimizer, last_epoch)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self) -> float:
        if self.last_epoch == 0:
            return self.base_lr
        return (
            self.eta_min
            + (self.base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
        )


class SequentialLR(LRScheduler):
    r"""Applies a sequence of learning rate schedulers based on milestones.

    Args:
        schedulers (List[LRScheduler]): List of learning rate schedulers.
        milestones (List[int]): List of epoch indices to switch to the next scheduler.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
    """

    def __init__(
        self, schedulers: List[LRScheduler], milestones: List[int], last_epoch: int = -1
    ):
        super().__init__(self.current_scheduler.optimizer, last_epoch)
        self.schedulers = schedulers
        self.milestones = milestones
        self.current_scheduler = self.schedulers[0]

    def step(self, epoch: int = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        if epoch in self.milestones:
            self.current_scheduler = self.schedulers[self.milestones.index(epoch)]
        self.current_scheduler.step(epoch)
