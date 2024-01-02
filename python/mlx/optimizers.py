# Copyright © 2023 Apple Inc.

import math
from typing import Callable, List

import mlx.core as mx
from mlx.utils import tree_map


class OptimizerState(dict):
    """The optimizer state implements a recursively defined
    :class:`collections.defaultdict`, namely a missing key in an optimizer
    state is an :class:`OptimizerState`.

    .. note::
       :meth:`OptimizerState.get` in contrast to a normal dictionary also sets
       the key to the ``default`` value if the ``key`` was not present in the
       dictionary.
    """

    def __getitem__(self, key):
        if key not in self:
            self[key] = OptimizerState()
        return super().__getitem__(key)

    def get(self, key, default):
        """If ``key`` doesn't exist set its value to ``default`` and then return it."""
        if key not in self:
            self[key] = default
        return super().__getitem__(key)


class Optimizer:
    """The base class for all optimizers. It allows us to implement an
    optimizer on a per-parameter basis and apply it to a parameter tree.

    Attributes:
        state (OptimizerState): It holds the optimizer's state dictionary.
    """

    def __init__(self):
        self.state = OptimizerState()

    def update(self, model: "mlx.nn.Module", gradients: dict):
        """Apply the gradients to the parameters of the model and update the
        model with the new parameters.

        Args:
            model (mlx.nn.Module): An mlx module to be updated.
            gradients (dict): A Python tree of gradients, most likely computed
                              via :func:`mlx.nn.value_and_grad`.
        """
        model.update(self.apply_gradients(gradients, model))

    def apply_gradients(self, gradients: dict, model: dict):
        """Apply the gradients to the parameters and return the updated parameters.

        Can be used to update a model via
        ``model.update(opt.apply_gradients(grads, model))`` which is precisely
        how :meth:`Optimizer.update` is implemented.

        Args:
            gradients (dict): A Python tree of gradients.
            model (dict): A Python tree of parameters. It can be a superset of
                          the gradients. In that case the returned python tree
                          will be of the same structure as the gradients.
        """
        return tree_map(self.apply_single, gradients, model, self.state)

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: OptimizerState
    ):
        """To be extended by the children classes to implement each optimizer's
        update."""
        raise NotImplementedError()


class SGD(Optimizer):
    r"""Stochastic gradient descent optimizer.

    Updates a parameter :math:`w` with a gradient :math:`g` as follows

    .. math::

        v_{t+1} &= \mu v_t + (1 - \tau) g_t \\
        w_{t+1} &= w_t - \lambda v_{t+1}

    Args:
        learning_rate (float): The learning rate :math:`\lambda`.
        momentum (float, optional): The momentum strength :math:`\mu`. Default: ``0``
        weight_decay (float, optional): The weight decay (L2 penalty). Default: ``0``
        dampening (float, optional): Dampening for momentum :math:`\tau`. Default: ``0``
        nesterov (bool, optional): Enables Nesterov momentum. Default: ``False``
    """

    def __init__(
        self,
        learning_rate: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False,
    ):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening."
            )
        super().__init__()

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dampening = dampening
        self.nesterov = nesterov

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: OptimizerState
    ):
        """Performs the SGD parameter update and stores :math:`v` in the
        optimizer state."""
        if self.momentum <= 0:
            return parameter - self.learning_rate * gradient

        v = state.get("v", mx.zeros_like(gradient))

        if self.weight_decay != 0:
            gradient += self.weight_decay * parameter

        v = self.momentum * v
        if self.dampening > 0:
            v += (1 - self.dampening) * gradient
        else:
            v += gradient

        if self.nesterov:
            update = gradient + self.momentum * v
        else:
            update = v
        state["v"] = v
        return parameter - self.learning_rate * update


class RMSprop(Optimizer):
    r"""Implementation of the RMSprop optimizer [1].

    [1]: Tieleman, T. and Hinton, G. 2012. Lecture 6.5-rmsprop, coursera: Neural networks for machine learning

    .. math::

        v_{t+1} &= \alpha v_t + (1 - \alpha) g_t^2 \\
        w_{t+1} &= w_t - \lambda \frac{g_t}{\sqrt{v_{t+1}} + \epsilon}

    Args:
        learning_rate (float): The learning rate :math:`\lambda`.
        alpha (float, optional): The smoothing constant :math:`\alpha`.
          Default: ``0.99``
        eps (float, optional): The term :math:`\epsilon` added to the denominator
          to improve numerical stability. Default: ``1e-8``
    """

    def __init__(self, learning_rate: float, alpha: float = 0.99, eps: float = 1e-8):
        super().__init__()

        self.learning_rate = learning_rate
        self.alpha = alpha
        self.eps = eps

        if self.alpha < 0.0:
            raise ValueError(
                f"RMSprop alpha should be >=0, {self.alpha} was provided instead"
            )
        if self.eps < 0.0:
            raise ValueError(
                f"RMSprop epsilon should be >0, {self.eps} was provided instead"
            )

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: OptimizerState
    ):
        """Performs the RMSprop parameter update and stores :math:`v` in the optimizer state."""
        lr = self.learning_rate
        alpha = self.alpha
        eps = self.eps

        v = state.get("v", mx.zeros_like(gradient))
        v = alpha * v + (1 - alpha) * mx.square(gradient)
        state["v"] = v

        return parameter - lr * gradient / (mx.sqrt(v) + eps)


class Adagrad(Optimizer):
    r"""Implementation of the Adagrad optimizer [1].

    Our Adagrad implementation follows the original paper. In detail,

    [1]: Duchi, J., Hazan, E. and Singer, Y., 2011. Adaptive subgradient methods
    for online learning and stochastic optimization. JMLR 2011.

    .. math::

        v_{t+1} &= v_t + g_t^2 \\
        w_{t+1} &= w_t - \lambda \frac{g_t}{\sqrt{v_{t+1}} + \epsilon}

    Args:
        learning_rate (float): The learning rate :math:`\lambda`.
        eps (float, optional): The term :math:`\epsilon` added to the
          denominator to improve numerical stability. Default: ``1e-8``
    """

    def __init__(self, learning_rate: float, eps: float = 1e-8):
        super().__init__()

        self.learning_rate = learning_rate
        self.eps = eps

        if self.eps < 0.0:
            raise ValueError(
                f"Adagrad epsilon should be >0, {self.eps} was provided instead"
            )

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: OptimizerState
    ):
        """Performs the Adagrad parameter update and stores :math:`v` in the
        optimizer state."""
        lr = self.learning_rate
        eps = self.eps

        v = state.get("v", mx.zeros_like(gradient))
        v = v + mx.square(gradient)
        state["v"] = v

        return parameter - lr * gradient / (mx.sqrt(v) + eps)


class AdaDelta(Optimizer):
    r"""Implementation of the AdaDelta optimizer with learning rate[1].

    Our AdaDelta implementation follows the original paper. In detail,

    [1]: Zeiler, M.D., 2012. ADADELTA: an adaptive learning rate method. arXiv preprint arXiv:1212.5701.

    .. math::

        v_{t+1} &= \rho v_t + (1 - \rho) g_t^2 \\
        \Delta w_{t+1} &= \frac{\sqrt{u_t + \epsilon}}{\sqrt{v_{t+1} + \epsilon}} g_t \\
        u_{t+1} &= \rho u_t + (1 - \rho) \Delta w_{t+1}^2 \\
        w_{t+1} &= w_t - \lambda \Delta w_{t+1}

    Args:
        learning_rate (float): The learning rate :math:`\lambda`.
        rho (float, optional): The coefficient :math:`\rho` used for computing a
            running average of squared gradients. Default: ``0.9``
        eps (float, optional): The term :math:`\epsilon` added to the denominator to improve
          numerical stability. Ddefault: `1e-8`
    """

    def __init__(self, learning_rate: float, rho: float = 0.9, eps: float = 1e-6):
        super().__init__()

        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = eps
        if self.rho < 0.0:
            raise ValueError(
                f"AdaDelta rho should be >=0, {self.rho} was provided instead"
            )
        if self.eps < 0.0:
            raise ValueError(
                f"AdaDelta epsilon should be >0, {self.eps} was provided instead"
            )

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: OptimizerState
    ):
        """Performs the AdaDelta parameter update and stores :math:`v` and
        :math:`u` in the optimizer state."""
        lr = self.learning_rate
        rho = self.rho
        eps = self.eps

        v = state.get("v", mx.zeros_like(gradient))
        u = state.get("s", mx.zeros_like(gradient))

        v = rho * v + (1 - rho) * mx.square(gradient)
        d = mx.sqrt(u + eps) / mx.sqrt(v + eps) * gradient
        u = rho * u + (1 - rho) * mx.square(d)

        state["v"] = v
        state["u"] = u

        return parameter - lr * d


class Adam(Optimizer):
    r"""Implementation of the Adam optimizer [1].

    Our Adam implementation follows the original paper and omits the bias
    correction in the first and second moment estimates. In detail,

    [1]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic
    optimization. ICLR 2015.

    .. math::

        m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
        v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
        w_{t+1} &= w_t - \lambda \frac{m_{t+1}}{\sqrt{v_{t+1} + \epsilon}}

    Args:
        learning_rate (float): The learning rate :math:`\lambda`.
        betas (Tuple[float, float], optional): The coefficients
          :math:`(\beta_1, \beta_2)` used for computing running averages of the
          gradient and its square. Default: ``(0.9, 0.999)``
        eps (float, optional): The term :math:`\epsilon` added to the
          denominator to improve numerical stability. Default: ``1e-8``
    """

    def __init__(
        self, learning_rate: float, betas: List[float] = [0.9, 0.999], eps: float = 1e-8
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.betas = betas
        self.eps = eps

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: OptimizerState
    ):
        """Performs the Adam parameter update and stores :math:`v` and
        :math:`m` in the optimizer state."""
        lr = self.learning_rate
        b1, b2 = self.betas
        eps = self.eps

        m = state.get("m", gradient)
        v = state.get("v", mx.square(gradient))
        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * mx.square(gradient)
        state["m"] = m
        state["v"] = v

        return parameter - lr * m / (mx.sqrt(v) + eps)


class AdamW(Adam):
    r"""Implementation of the AdamW optimizer [1].

    Following the above convention, in contrast with [1], we do not use bias
    correction in the first and second moments for AdamW. We update the weights
    with a weight_decay (:math:`\lambda`) value:

    [1]: Loshchilov, I. and Hutter, F., 2019. Decoupled weight decay
    regularization. ICLR 2019.

    .. math::

        m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
        v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
        w_{t+1} &= w_t - \alpha (\frac{m_{t+1}}{\sqrt{v_{t+1} + \epsilon}} + \lambda w_t)

    Args:
        learning_rate (float): The learning rate :math:`\alpha`.
        betas (Tuple[float, float], optional): The coefficients
          :math:`(\beta_1, \beta_2)` used for computing running averages of the
          gradient and its square. Default: ``(0.9, 0.999)``
        eps (float, optional): The term :math:`\epsilon` added to the
          denominator to improve numerical stability. Default: ``1e-8``
        weight_decay (float, optional): The weight decay :math:`\lambda`.
          Default: ``0``.
    """

    def __init__(
        self,
        learning_rate: float,
        betas: List[float] = [0.9, 0.999],
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        super().__init__(learning_rate=learning_rate, betas=betas, eps=eps)
        self.weight_decay = weight_decay

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: OptimizerState
    ):
        """Performs the AdamW parameter update by modifying the parameters
        passed into Adam.
        """

        return super().apply_single(
            gradient, parameter * (1 - self.learning_rate * self.weight_decay), state
        )


class Adamax(Adam):
    r"""Implementation of the Adamax optimizer. It is a variant of Adam based
    on the infinity norm [1].

    Our Adam implementation follows the original paper and omits the bias
    correction in the first and second moment estimates. In detail,

    [1]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic
    optimization. ICLR 2015.

    .. math::

        m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
        v_{t+1} &= \max(\beta_2 v_t, |g_t|) \\
        w_{t+1} &= w_t - \lambda \frac{m_{t+1}}{v_{t+1} + \epsilon}

    Args:
        learning_rate (float): The learning rate :math:`\lambda`.
        betas (Tuple[float, float], optional): The coefficients
          :math:`(\beta_1, \beta_2)` used for computing running averages of the
          gradient and its square. Default: ``(0.9, 0.999)``
        eps (float, optional): The term :math:`\epsilon` added to the
          denominator to improve numerical stability. Default: ``1e-8``
    """

    def __init__(
        self, learning_rate: float, betas: List[float] = [0.9, 0.999], eps: float = 1e-8
    ):
        super().__init__(learning_rate, betas, eps)

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: OptimizerState
    ):
        """Performs the Adamax parameter update and stores :math:`v` and
        :math:`m` in the optimizer state."""
        lr = self.learning_rate
        b1, b2 = self.betas
        eps = self.eps

        m = state.get("m", mx.zeros_like(gradient))
        v = state.get("v", mx.zeros_like(gradient))

        m = b1 * m + (1 - b1) * gradient
        v = mx.maximum(b2 * v, mx.abs(gradient))
        state["m"] = m
        state["v"] = v

        return parameter - lr * m / (v + eps)


class Lion(Optimizer):
    r"""Implementation of the Lion optimizer [1].

    Since updates are computed through the sign operation, they tend to
    have larger norm than for other optimizers such as SGD and Adam.
    We recommend a learning rate that is 3-10x smaller than AdamW and a
    weight decay 3-10x larger than AdamW to maintain the strength
    (lr * wd). Our Lion implementation follows the original paper. In
    detail,

    [1]: Chen, X. Symbolic Discovery of Optimization Algorithms. arXiv
    preprint arXiv:2302.06675.

    .. math::

        c_{t + 1} &= \beta_1 m_t + (1 - \beta_1) g_t
        m_{t + 1} &= \beta_2 m_t + (1 - \beta_2) g_t
        w_{t + 1} &= w_t - \eta (\text{sign}(c_t) + \lambda w_t)

    Args:
        learning_rate (float): The learning rate :math:`\eta`.
        betas (Tuple[float, float], optional): The coefficients
          :math:`(\beta_1, \beta_2)` used for computing the gradient
          momentum and update direction. Default: ``(0.9, 0.99)``
        weight_decay (float, optional): The weight decay :math:`\lambda`. Default: ``0.0``
    """

    def __init__(
        self,
        learning_rate: float,
        betas: List[float] = [0.9, 0.99],
        weight_decay: float = 0.0,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: OptimizerState
    ):
        """Performs the Lion parameter update and stores :math:`m`
        in the optimizer state."""
        lr = self.learning_rate
        b1, b2 = self.betas
        weight_decay = self.weight_decay

        m = state.get("m", gradient)
        c = b1 * m + (1 - b1) * gradient
        state["m"] = b2 * m + (1 - b2) * gradient
        if weight_decay > 0:
            parameter = (1 - lr * weight_decay) * parameter
        return parameter - lr * mx.sign(c)


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
        lr_lambda (Callable): A function or a list of functions defining the decay factor.
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
