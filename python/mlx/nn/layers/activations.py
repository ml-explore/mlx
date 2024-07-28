# Copyright © 2023 Apple Inc.

import math
from functools import partial
from typing import Any

import mlx.core as mx
from mlx.nn.layers.base import Module


def _make_activation_module(f):
    def decorator(klass):
        klass.__call__ = lambda _, x: f(x)
        return klass

    return decorator


@partial(mx.compile, shapeless=True)
def sigmoid(x):
    r"""Applies the sigmoid function.

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
    """
    return mx.sigmoid(x)


@partial(mx.compile, shapeless=True)
def relu(x):
    r"""Applies the Rectified Linear Unit.

    Simply ``mx.maximum(x, 0)``.
    """
    return mx.maximum(x, 0)


@partial(mx.compile, shapeless=True)
def leaky_relu(x, negative_slope=0.01):
    r"""Applies the Leaky Rectified Linear Unit.

    Simply ``mx.maximum(negative_slope * x, x)``.
    """
    return mx.maximum(negative_slope * x, x)


@partial(mx.compile, shapeless=True)
def log_softmax(x, axis=-1):
    r"""Applies the Log Softmax function.

    Applies :math:`x + \log \sum_i e^{x_i}` element wise.
    """
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


@partial(mx.compile, shapeless=True)
def elu(x, alpha=1.0):
    r"""Applies the Exponential Linear Unit.

    Simply ``mx.where(x > 0, x, alpha * (mx.exp(x) - 1))``.
    """
    return mx.where(x > 0, x, alpha * (mx.exp(x) - 1))


@partial(mx.compile, shapeless=True)
def relu6(x):
    r"""Applies the Rectified Linear Unit 6.

    Applies :math:`\min(\max(x, 0), 6)` element wise.
    """
    return mx.minimum(mx.maximum(x, 0), 6.0)


@partial(mx.compile, shapeless=True)
def softmax(x, axis=-1):
    r"""Applies the Softmax function.

    Applies :math:`\frac{e^{x_i}}{\sum_j e^{x_j}}` element wise.
    """
    return mx.softmax(x, axis=axis)


@partial(mx.compile, shapeless=True)
def softplus(x):
    r"""Applies the Softplus function.

    Applies :math:`\log(1 + \exp(x))` element wise.
    """
    return mx.logaddexp(x, 0)


@partial(mx.compile, shapeless=True)
def softsign(x):
    r"""Applies the Softsign function.

    Applies :math:`\frac{x}{1 + |x|}` element wise.
    """
    return mx.divide(x, 1 + mx.abs(x))


@partial(mx.compile, shapeless=True)
def softshrink(x, lambd: float = 0.5):
    r"""Applies the Softshrink activation function.

    .. math::
        \text{softshrink}(x) = \begin{cases}
        x - \lambda & \text{if } x > \lambda \\
        x + \lambda & \text{if } x < -\lambda \\
        0 & \text{otherwise}
        \end{cases}
    """
    return mx.where(mx.abs(x) > lambd, x - mx.sign(x) * lambd, 0)


@partial(mx.compile, shapeless=True)
def celu(x, alpha=1.0):
    r"""Applies the Continuously Differentiable Exponential Linear Unit.

    Applies :math:`\max(0, x) + \min(0, \alpha * (\exp(x / \alpha) - 1))`
    element wise.
    """
    return mx.maximum(x, 0.0) + alpha * (mx.exp(mx.minimum(x, 0.0) / alpha) - 1)


@partial(mx.compile, shapeless=True)
def silu(x):
    r"""Applies the Sigmoid Linear Unit. Also known as Swish.

    Applies :math:`x \sigma(x)` element wise, where :math:`\sigma(\cdot)` is
    the logistic sigmoid.
    """
    return x * mx.sigmoid(x)


@partial(mx.compile, shapeless=True)
def log_sigmoid(x):
    r"""Applies the Log Sigmoid function.

    Applies :math:`\log(\sigma(x)) = -\log(1 + e^{-x})` element wise.
    """
    return -softplus(-x)


@partial(mx.compile, shapeless=True)
def gelu(x) -> mx.array:
    r"""Applies the Gaussian Error Linear Units function.

    .. math::
        \textrm{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Gaussian CDF.

    See also :func:`gelu_approx` and :func:`gelu_fast_approx` for faster
    approximations.
    """
    return x * (1 + mx.erf(x / math.sqrt(2))) / 2


@partial(mx.compile, shapeless=True)
def gelu_approx(x):
    r"""An approximation to Gaussian Error Linear Unit.

    See :func:`gelu` for the exact computation.

    This function approximates ``gelu`` with a maximum absolute error :math:`<
    0.0005` in the range :math:`[-6, 6]` using the following

    .. math::

        x = 0.5 * x * \left(1 + \text{Tanh}\left((\sqrt{2 / \pi} * \left(x + 0.044715 * x^3\right)\right)\right)

    """
    return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


@partial(mx.compile, shapeless=True)
def gelu_fast_approx(x):
    r"""A fast approximation to Gaussian Error Linear Unit.

    See :func:`gelu` for the exact computation.

    This function approximates ``gelu`` with a maximum absolute error :math:`<
    0.015` in the range :math:`[-6, 6]` using the following

    .. math::

        x = x \sigma\left(1.702 x\right)

    where :math:`\sigma(\cdot)` is the logistic sigmoid.

    References:
    - https://github.com/hendrycks/GELUs
    - https://arxiv.org/abs/1606.08415
    """
    return x * mx.sigmoid(1.702 * x)


def glu(x: mx.array, axis: int = -1) -> mx.array:
    r"""Applies the gated linear unit function.

    This function splits the ``axis`` dimension of the input into two halves
    (:math:`a` and :math:`b`) and applies :math:`a * \sigma(b)`.

    .. math::
        \textrm{GLU}(x) = a * \sigma(b)

    Args:
        axis (int): The dimension to split along. Default: ``-1``
    """
    a, b = mx.split(x, indices_or_sections=2, axis=axis)
    return a * mx.sigmoid(b)


@partial(mx.compile, shapeless=True)
def step(x: mx.array, threshold: float = 0.0):
    r"""Applies the Step Activation Function.

    This function implements a binary step activation, where the output is set
    to 1 if the input is greater than a specified threshold, and 0 otherwise.

    .. math::
        \text{step}(x) = \begin{cases}
        0 & \text{if } x < \text{threshold} \\
        1 & \text{if } x \geq \text{threshold}
        \end{cases}

    Args:
        threshold: The value to threshold at.
    """

    return mx.where(x > threshold, 1, 0)


@partial(mx.compile, shapeless=True)
def selu(x):
    r"""Applies the Scaled Exponential Linear Unit.

    .. math::
        \text{selu}(x) = \begin{cases}
        \lambda x & \text{if } x > 0 \\
        \lambda \alpha (\exp(x) - 1) & \text{if } x \leq 0
        \end{cases}

    where :math:`\lambda = 1.0507` and :math:`\alpha = 1.67326`.

    See also :func:`elu`.
    """
    return elu(x, 1.67326) * 1.0507


@partial(mx.compile, shapeless=True)
def prelu(x: mx.array, alpha: mx.array) -> mx.array:
    r"""Applies the element-wise parametric ReLU.

    .. math::
        \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

    where :math:`a` is an array.
    """
    return mx.maximum(0, x) + alpha * mx.minimum(0, x)


@partial(mx.compile, shapeless=True)
def mish(x: mx.array) -> mx.array:
    r"""Applies the Mish function, element-wise.

    Mish: A Self Regularized Non-Monotonic Neural Activation Function.

    Reference: https://arxiv.org/abs/1908.08681

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

    """
    return x * mx.tanh(softplus(x))


@partial(mx.compile, shapeless=True)
def hardswish(x):
    r"""Applies the hardswish function, element-wise.

    .. math::
        \text{Hardswish}(x) = x * \min(\max(x + 3, 0), 6) / 6
    """
    max_x_3 = mx.maximum(x + 3, 0)
    return x * mx.minimum(max_x_3, 6) / 6


@partial(mx.compile, shapeless=True)
def hard_tanh(x, min_val=-1.0, max_val=1.0):
    r"""Applies the HardTanh function.

    Applies :math:`\max(\min(x, \text{max\_val}), \text{min\_val})` element-wise.
    """
    return mx.minimum(mx.maximum(x, min_val), max_val)


@partial(mx.compile, shapeless=True)
def hard_shrink(x, lambd=0.5):
    r"""Applies the HardShrink activation function.

    .. math::
        \text{hardshrink}(x) = \begin{cases}
        x & \text{if } x > \lambda \\
        x & \text{if } x < -\lambda \\
        0 & \text{otherwise}
        \end{cases}
    """
    return mx.where(mx.abs(x) > lambd, x, 0)


@partial(mx.compile, shapeless=True)
def softmin(x, axis=-1):
    r"""Applies the Softmin function.

    Applies :math:`\frac{e^{-x_i}}{\sum_j e^{-x_j}}` element-wise.
    """
    return mx.softmax(-x, axis=axis)


def tanh(x):
    """Applies the hyperbolic tangent function.

    Simply ``mx.tanh(x)``.
    """
    return mx.tanh(x)


class GLU(Module):
    r"""Applies the gated linear unit function.

    This function splits the ``axis`` dimension of the input into two halves
    (:math:`a` and :math:`b`) and applies :math:`a * \sigma(b)`.

    .. math::
        \textrm{GLU}(x) = a * \sigma(b)

    Args:
        axis (int): The dimension to split along. Default: ``-1``
    """

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def __call__(self, x) -> Any:
        return glu(x=x, axis=self.axis)


@_make_activation_module(sigmoid)
class Sigmoid(Module):
    r"""Applies the sigmoid function, element-wise.

    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
    """


@_make_activation_module(mish)
class Mish(Module):
    r"""Applies the Mish function, element-wise.

    Reference: https://arxiv.org/abs/1908.08681

    .. math::
        \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))

    """


@_make_activation_module(relu)
class ReLU(Module):
    r"""Applies the Rectified Linear Unit.
        Simply ``mx.maximum(x, 0)``.

    See :func:`relu` for the functional equivalent.
    """


class LeakyReLU(Module):
    r"""Applies the Leaky Rectified Linear Unit.

    Simply ``mx.maximum(negative_slope * x, x)``.

    Args:
        negative_slope: Controls the angle of the negative slope. Default: ``1e-2``
    """

    def __init__(self, negative_slope=1e-2):
        super().__init__()
        self._negative_slope = negative_slope

    def __call__(self, x):
        return leaky_relu(x, self._negative_slope)


class ELU(Module):
    r"""Applies the Exponential Linear Unit.
        Simply ``mx.where(x > 0, x, alpha * (mx.exp(x) - 1))``.

    See :func:`elu` for the functional equivalent.

    Args:
        alpha: the :math:`\alpha` value for the ELU formulation. Default: ``1.0``
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self._alpha = alpha

    def __call__(self, x):
        return elu(x, self._alpha)


@_make_activation_module(relu6)
class ReLU6(Module):
    r"""Applies the Rectified Linear Unit 6.

    See :func:`relu6` for the functional equivalent.
    """


@_make_activation_module(softmax)
class Softmax(Module):
    r"""Applies the Softmax function.

    See :func:`softmax` for the functional equivalent.
    """


@_make_activation_module(softplus)
class Softplus(Module):
    r"""Applies the Softplus function.

    See :func:`softplus` for the functional equivalent.
    """


@_make_activation_module(softsign)
class Softsign(Module):
    r"""Applies the Softsign function.

    See :func:`softsign` for the functional equivalent.
    """


class Softshrink(Module):
    r"""Applies the Softshrink function.

    See :func:`softshrink` for the functional equivalent.

    Args:
        lambd: the :math:`\lambda` value for Softshrink. Default: ``0.5``
    """

    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def __call__(self, x):
        return softshrink(x, self.lambd)


class CELU(Module):
    r"""Applies the Continuously Differentiable Exponential Linear Unit.
        Applies :math:`\max(0, x) + \min(0, \alpha * (\exp(x / \alpha) - 1))`
        element wise.

    See :func:`celu` for the functional equivalent.

    Args:
        alpha: the :math:`\alpha` value for the CELU formulation. Default: ``1.0``
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self._alpha = alpha

    def __call__(self, x):
        return celu(x, self._alpha)


@_make_activation_module(silu)
class SiLU(Module):
    r"""Applies the Sigmoid Linear Unit. Also known as Swish.

    See :func:`silu` for the functional equivalent.
    """


@_make_activation_module(log_softmax)
class LogSoftmax(Module):
    r"""Applies the Log Softmax function.

    See :func:`log_softmax` for the functional equivalent.
    """


@_make_activation_module(log_sigmoid)
class LogSigmoid(Module):
    r"""Applies the Log Sigmoid function.

    See :func:`log_sigmoid` for the functional equivalent.
    """


class PReLU(Module):
    r"""Applies the element-wise parametric ReLU.
        Applies :math:`\max(0, x) + a * \min(0, x)` element wise, where :math:`a`
        is an array.

    See :func:`prelu` for the functional equivalent.

    Args:
        num_parameters: number of :math:`a` to learn. Default: ``1``
        init: the initial value of :math:`a`. Default: ``0.25``
    """

    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.weight = mx.full([num_parameters], init)

    def __call__(self, x: mx.array):
        return prelu(x, self.weight)


class GELU(Module):
    r"""Applies the Gaussian Error Linear Units.

    .. math::
        \textrm{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Gaussian CDF.

    However, if ``approx`` is set to 'precise' or 'fast' it applies

    .. math::
        \textrm{GELUApprox}(x) &= 0.5 * x * \left(1 + \text{Tanh}\left((\sqrt{2 / \pi} * \left(x + 0.044715 * x^3\right)\right)\right) \\
        \textrm{GELUFast}(x) &= x * \sigma\left(1.702 * x\right)

    respectively.

    .. note::
       For compatibility with the PyTorch API, 'tanh' can be used as an alias
       for 'precise'.

    See :func:`gelu`, :func:`gelu_approx` and :func:`gelu_fast_approx` for the
    functional equivalents and information regarding error bounds.
    

    Args:
        approx ('none' | 'precise' | 'fast'): Which approximation to gelu to use if any.
    """

    def __init__(self, approx="none"):
        super().__init__()

        if approx == "none":
            self._act = gelu
        elif approx == "precise" or approx == "tanh":
            self._act = gelu_approx
        elif approx == "fast":
            self._act = gelu_fast_approx
        else:
            raise ValueError(
                f"The approximation should be in ['none', 'precise', 'tanh', 'fast'] but '{approx}' was given"
            )

    def __call__(self, x):
        return self._act(x)


@_make_activation_module(tanh)
class Tanh(Module):
    r"""Applies the hyperbolic tangent function.

    See :func:`tanh` for the functional equivalent.
    """


@_make_activation_module(hardswish)
class Hardswish(Module):
    r"""Applies the hardswish function, element-wise.

    See :func:`hardswish` for the functional equivalent.
    """


class Step(Module):
    r"""Applies the Step Activation Function.

    This function implements a binary step activation, where the output is set
    to 1 if the input is greater than a specified threshold, and 0 otherwise.

    .. math::
        \text{step}(x) = \begin{cases}
        0 & \text{if } x < \text{threshold} \\
        1 & \text{if } x \geq \text{threshold}
        \end{cases}

    Args:
        threshold: The value to threshold at.
    """

    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold

    def __call__(self, x: mx.array):
        return step(x, self.threshold)


@_make_activation_module(selu)
class SELU(Module):
    r"""Applies the Scaled Exponential Linear Unit.

    See :func:`selu` for the functional equivalent.
    """


@_make_activation_module(hard_tanh)
class HardTanh(Module):
    r"""Applies the HardTanh function.

    See :func:`hard_tanh` for the functional equivalent.
    """


@_make_activation_module(hard_shrink)
class HardShrink(Module):
    r"""Applies the HardShrink function.

    See :func:`hard_shrink` for the functional equivalent.

    Args:
        lambd: the :math:`\lambda` value for Hardshrink. Default: ``0.5``
    """


@_make_activation_module(softmin)
class Softmin(Module):
    r"""Applies the Softmin function.

    See :func:`softmin` for the functional equivalent.
    """
