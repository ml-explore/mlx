# Copyright © 2023 Apple Inc.

import math
from typing import List

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

        v_{t+1} &= \mu v_t + g_t \\
        w_{t+1} &= w_t - \lambda v_{t+1}

    Args:
        learning_rate (float): The learning :math:`\lambda` for the update
        momentum (float, optional): The momentum strength :math:`\mu` (default: 0)
        weight_decay (float, optional): The weight decay (L2 penalty) (default: 0)
        dampening (float, optional): Dampening for momentum :math:`\tau` (default: 0)
        nesterov (bool, optional): Enables Nesterov momentum (default: False)
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


class Adam(Optimizer):
    r"""Implementation of the Adam optimizer [1].

    Our Adam implementation follows the original paper and omits the bias
    correction in the first and second moment estimates. In detail,

    .. math::

        m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
        v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
        w_{t+1} &= w_t - \lambda \frac{m_{t+1}}{\sqrt{v_{t+1} + \epsilon}}

    [1]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic
    optimization. ICLR 2015.
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
    with a weight_decay (λ) value:

    .. math::

        m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
        v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
        w_{t+1} &= w_t - \alpha (\frac{m_{t+1}}{\sqrt{v_{t+1} + \epsilon}} + \lambda w_t)

    [1]: Loshchilov, I. and Hutter, F., 2019. Decoupled weight decay 
    regularization. ICLR 2019.
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


class Adagrad(Optimizer):
    r"""Implementation of the Adagrad optimizer [1].

    Our Adagrad implementation follows the original paper. In detail,

    .. math::

        v_{t+1} &= v_t + g_t^2 \\
        w_{t+1} &= w_t - \lambda \frac{g_t}{\sqrt{v_{t+1} + \epsilon}}

    [1]: Duchi, J., Hazan, E. and Singer, Y., 2011. Adaptive subgradient methods
    for online learning and stochastic optimization. JMLR 2011.
    """

    def __init__(self, learning_rate: float, eps: float = 1e-8):
        super().__init__()

        self.learning_rate = learning_rate
        self.eps = eps

        if self.learning_rate < 0.0:
            raise ValueError(
                f"Adagrad learning rate should be >=0, {self.learning_rate} was provided instead"
            )
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
