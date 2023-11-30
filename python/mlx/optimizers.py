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

        v_{t+1} &= \mu v_t + (1 - \mu) g_t \\
        w_{t+1} &= w_t - \lambda v_{t+1}

    Args:
        learning_rate (float): The learning :math:`\lambda` for the update
        momentum (float): The momentum strength :math:`\mu`
    """

    def __init__(self, learning_rate: float, momentum: float = 0.0):
        super().__init__()

        self.learning_rate = learning_rate
        self.momentum = momentum

    def apply_single(
        self, gradient: mx.array, parameter: mx.array, state: OptimizerState
    ):
        """Performs the SGD parameter update and stores :math:`v` in the
        optimizer state."""
        if self.momentum <= 0:
            return parameter - self.learning_rate * gradient

        v = state.get("v", mx.zeros_like(gradient))
        v = self.momentum * v + (1 - self.momentum) * gradient
        state["v"] = v
        return parameter - self.learning_rate * v


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
