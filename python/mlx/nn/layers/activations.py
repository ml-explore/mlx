# Copyright © 2023 Apple Inc.

import math

import mlx.core as mx
from mlx.nn.layers.base import Module


def _make_activation_module(f):
    def decorator(klass):
        klass.__doc__ = f.__doc__
        klass.__call__ = lambda self, x: f(x)
        return klass

    return decorator


def relu(x):
    """Applies the Rectified Linear Unit.

    Simply ``mx.maximum(x, 0)``.
    """
    return mx.maximum(x, 0)


def silu(x):
    r"""Applies the Sigmoid Linear Unit.

    Applies :math:`x \sigma(x)` element wise, where :math:`\sigma(\cdot)` is
    the logistic sigmoid.
    """
    return x * mx.sigmoid(x)


def gelu(x):
    """Applies the Gaussian Error Linear Units function.

    .. math::
        \\textrm{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Gaussian CDF.

    See also :func:`gelu_approx` and :func:`gelu_fast_approx` for faster
    approximations.
    """
    return x * (1 + mx.erf(x / math.sqrt(2))) / 2


def gelu_approx(x):
    r"""An approximation to Gaussian Error Linear Unit.

    See :func:`gelu` for the exact computation.

    This function approximates ``gelu`` with a maximum absolute error :math:`<
    0.0003` in the range :math:`[-6, 6]` using the following

    .. math::

        x = x \sigma\left(1.60033 x \left(1 + 0.0433603 x^2\right)\right)

    where :math:`\sigma(\cdot)` is the logistic sigmoid.
    """
    return x * mx.sigmoid(1.60033 * x * (1 + 0.0433603 * x.square()))


def gelu_fast_approx(x):
    r"""A fast approximation to Gaussian Error Linear Unit.

    See :func:`gelu` for the exact computation.

    This function approximates ``gelu`` with a maximum absolute error :math:`<
    0.015` in the range :math:`[-6, 6]` using the following

    .. math::

        x = x \sigma\left(1.773 x\right)

    where :math:`\sigma(\cdot)` is the logistic sigmoid.
    """
    return x * mx.sigmoid(1.773 * x)


@_make_activation_module(relu)
class ReLU(Module):
    pass


@_make_activation_module(silu)
class SiLU(Module):
    pass


class GELU(Module):
    r"""Applies the Gaussian Error Linear Units.

    .. math::
        \textrm{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Gaussian CDF.

    However, if ``approx`` is set to 'precise' or 'fast' it applies

    .. math::
        \textrm{GELUApprox}(x) &= x * \sigma\left(1.60033 * x \left(1 + 0.0433603 * x^2\right)\right) \\
        \textrm{GELUFast}(x) &= x * \sigma\left(1.773 * x\right)

    respectively.

    See :func:`gelu`, :func:`gelu_approx` and :func:`gelu_fast_approx` for the
    functional equivalents and information regarding error bounds.

    Args:
        approx ('none' | 'precise' | 'fast'): Which approximation to gelu to use if any.
    """

    def __init__(self, approx="none"):
        super().__init__()

        if approx == "none":
            self._act = gelu
        elif approx == "precise":
            self._act = gelu_approx
        elif approx == "fast":
            self._act = gelu_fast_approx
        else:
            raise ValueError(
                f"The approximation should be in ['none', 'precise', 'fast'] but '{approx}' was given"
            )

    def __call__(self, x):
        return self._act(x)
