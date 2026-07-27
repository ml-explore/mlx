# Copyright © 2024 Apple Inc.

"""Special mathematical functions implemented purely with ``mlx.core`` ops.

This module provides MLX-native equivalents of a subset of
:mod:`scipy.special`. Every function is written using only element-wise
``mlx.core`` operations, so the results are lazy, differentiable, and run on
whatever device (CPU / GPU) the input array lives on — no custom Metal or C++
kernels required.

The implementations use classic closed-form approximations whose error bounds
are well characterised:

* ``erf`` / ``erfc`` — Abramowitz & Stegun 7.1.26 (rational approximation,
  ``|error| <= 1.5e-7``).
* ``i0`` — Abramowitz & Stegun 9.8.1 / 9.8.2 (polynomial approximations,
  ``|error| <= 1.9e-7``).
* ``gammaln`` — Lanczos approximation (``g = 7``, 9 coefficients), with the
  reflection formula for arguments ``< 1/2``.
* ``digamma`` — upward recurrence to the asymptotic (Stirling) regime, with the
  reflection formula for non-positive arguments.

References:
    M. Abramowitz and I. A. Stegun, *Handbook of Mathematical Functions*,
    National Bureau of Standards, 1964.
    C. Lanczos, "A Precision Approximation of the Gamma Function",
    *J. SIAM Numer. Anal.* 1 (1964) 86-96.
"""

import math

import mlx.core as mx

__all__ = ["erf", "erfc", "i0", "gammaln", "digamma"]

# Abramowitz & Stegun 7.1.26 coefficients (error function).
_ERF_P = 0.3275911
_ERF_A = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)

# Abramowitz & Stegun 9.8.1 coefficients (I0, |x| <= 3.75, argument t = (x/3.75)**2).
_I0_SMALL = (
    1.0,
    3.5156229,
    3.0899424,
    1.2067492,
    0.2659732,
    0.0360768,
    0.0045813,
)
# Abramowitz & Stegun 9.8.2 coefficients (I0, |x| >= 3.75, argument t = 3.75/x).
_I0_LARGE = (
    0.39894228,
    0.01328592,
    0.00225319,
    -0.00157565,
    0.00916281,
    -0.02057706,
    0.02635537,
    -0.01647633,
    0.00392377,
)

# Lanczos coefficients for g = 7 (log-gamma).
_LANCZOS_G = 7.0
_LANCZOS_C = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def _as_float(x: mx.array) -> mx.array:
    """Return ``x`` as a floating-point array, promoting integer inputs."""
    x = mx.array(x)
    if x.dtype in (mx.float16, mx.bfloat16, mx.float32, mx.float64):
        return x
    return x.astype(mx.float32)


def _horner(t: mx.array, coeffs) -> mx.array:
    """Evaluate a polynomial ``sum(coeffs[i] * t**i)`` via Horner's method."""
    result = mx.full(t.shape, coeffs[-1], dtype=t.dtype)
    for c in reversed(coeffs[:-1]):
        result = result * t + c
    return result


def erf(x: mx.array) -> mx.array:
    r"""Error function.

    Computes

    .. math::

        \operatorname{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2}\, dt

    using the rational approximation of Abramowitz & Stegun 7.1.26 applied to
    ``|x|`` (with ``erf(-x) = -erf(x)``), whose absolute error is bounded by
    ``1.5e-7``.

    Args:
        x (array): Input array.

    Returns:
        array: ``erf(x)`` with the same shape and floating dtype as ``x``.
    """
    x = _as_float(x)
    sign = mx.sign(x)
    ax = mx.abs(x)
    t = 1.0 / (1.0 + _ERF_P * ax)
    # A&S 7.1.26: erf(ax) = 1 - (a1 t + ... + a5 t^5) exp(-ax^2).
    poly = t * _horner(t, _ERF_A)
    return sign * (1.0 - poly * mx.exp(-ax * ax))


def erfc(x: mx.array) -> mx.array:
    r"""Complementary error function, ``erfc(x) = 1 - erf(x)``.

    Rather than forming ``1 - erf(x)`` (which loses precision for large ``x``
    where ``erf(x) -> 1``), the tail term of Abramowitz & Stegun 7.1.26 is
    evaluated directly:

    .. math::

        \operatorname{erfc}(x) = (a_1 t + \dots + a_5 t^5)\, e^{-x^2},
        \quad t = \frac{1}{1 + p\,x}, \quad x \ge 0,

    and the reflection ``erfc(-x) = 2 - erfc(x)`` is used for negative inputs.
    This keeps the small positive values returned for large ``x`` accurate.

    Args:
        x (array): Input array.

    Returns:
        array: ``erfc(x)`` with the same shape and floating dtype as ``x``.
    """
    x = _as_float(x)
    ax = mx.abs(x)
    t = 1.0 / (1.0 + _ERF_P * ax)
    tail = t * _horner(t, _ERF_A) * mx.exp(-ax * ax)  # == erfc(|x|)
    return mx.where(x >= 0, tail, 2.0 - tail)


def i0(x: mx.array) -> mx.array:
    r"""Modified Bessel function of the first kind, order zero.

    Computes

    .. math::

        I_0(x) = \sum_{k=0}^{\infty} \frac{(x^2/4)^k}{(k!)^2}

    with the piecewise polynomial approximations of Abramowitz & Stegun 9.8.1
    (``|x| <= 3.75``) and 9.8.2 (``|x| >= 3.75``). ``I_0`` is even, so ``|x|``
    is used throughout. The absolute error of the approximation is bounded by
    ``1.9e-7``.

    Args:
        x (array): Input array.

    Returns:
        array: ``i0(x)`` with the same shape and floating dtype as ``x``.
    """
    x = _as_float(x)
    ax = mx.abs(x)

    # Small-argument branch (A&S 9.8.1), t = (x / 3.75)^2.
    t_small = (ax / 3.75) ** 2
    small = _horner(t_small, _I0_SMALL)

    # Large-argument branch (A&S 9.8.2), t = 3.75 / x. Clamp the denominator so
    # the (discarded) values in the small region stay finite.
    ax_large = mx.maximum(ax, 3.75)
    t_large = 3.75 / ax_large
    large = mx.exp(ax_large) / mx.sqrt(ax_large) * _horner(t_large, _I0_LARGE)

    result = mx.where(ax < 3.75, small, large)
    # I0(±inf) = +inf; the large-argument branch evaluates to inf/inf = nan there.
    return mx.where(mx.isinf(ax), mx.full(x.shape, float("inf"), dtype=x.dtype), result)


def _lanczos_lgamma(w: mx.array) -> mx.array:
    """Log-gamma via the Lanczos series; assumes ``w >= 0.5`` element-wise."""
    w1 = w - 1.0
    series = mx.full(w.shape, _LANCZOS_C[0], dtype=w.dtype)
    for i in range(1, len(_LANCZOS_C)):
        series = series + _LANCZOS_C[i] / (w1 + i)
    t = w1 + _LANCZOS_G + 0.5
    return _HALF_LOG_2PI + (w1 + 0.5) * mx.log(t) - t + mx.log(series)


def gammaln(x: mx.array) -> mx.array:
    r"""Natural logarithm of the absolute value of the gamma function.

    Returns :math:`\log |\Gamma(x)|`, matching :func:`scipy.special.gammaln`.
    The Lanczos approximation (``g = 7``, 9 coefficients) is evaluated on
    ``max(x, 1 - x)`` so its argument is always ``>= 1/2``; for ``x < 1/2`` the
    reflection formula

    .. math::

        \log|\Gamma(x)| = \log \pi - \log|\sin(\pi x)| - \log|\Gamma(1 - x)|

    is then applied. The relative error is on the order of machine precision for
    ``float64`` inputs; poles at non-positive integers return ``+inf``.

    Args:
        x (array): Input array.

    Returns:
        array: ``gammaln(x)`` with the same shape and floating dtype as ``x``.
    """
    x = _as_float(x)
    reflect = x < 0.5
    # Argument fed to the Lanczos series is always >= 0.5.
    w = mx.where(reflect, 1.0 - x, x)
    lanczos = _lanczos_lgamma(w)
    reflected = math.log(math.pi) - mx.log(mx.abs(mx.sin(math.pi * x))) - lanczos
    result = mx.where(reflect, reflected, lanczos)
    # gammaln(+inf) = +inf; the Lanczos series evaluates to inf - inf = nan there.
    pos_inf = mx.logical_and(mx.isinf(x), x > 0)
    return mx.where(pos_inf, mx.full(x.shape, float("inf"), dtype=x.dtype), result)


def digamma(x: mx.array) -> mx.array:
    r"""Digamma function, the logarithmic derivative of the gamma function.

    Computes :math:`\psi(x) = \frac{d}{dx} \log \Gamma(x)`. Non-positive
    arguments are handled with the reflection formula

    .. math::

        \psi(x) = \psi(1 - x) - \pi \cot(\pi x),

    after which the recurrence :math:`\psi(x) = \psi(x + 1) - 1/x` shifts the
    argument into the region ``x >= 6``, where the asymptotic expansion

    .. math::

        \psi(x) \sim \ln x - \frac{1}{2x}
        - \frac{1}{12x^2} + \frac{1}{120x^4}
        - \frac{1}{252x^6} + \frac{1}{240x^8}

    is accurate to better than ``1e-9``.

    Args:
        x (array): Input array.

    Returns:
        array: ``digamma(x)`` with the same shape and floating dtype as ``x``.
    """
    x = _as_float(x)
    reflect = x <= 0
    # Fold non-positive arguments to the positive half-line via reflection.
    arg = mx.where(reflect, 1.0 - x, x)

    # Upward recurrence: psi(arg) = psi(arg + 1) - 1/arg until arg >= 6.
    threshold = 6.0
    correction = mx.zeros(arg.shape, dtype=arg.dtype)
    work = arg
    # From any positive start the loop reaches `threshold` in at most 6 steps.
    while float(mx.min(work).item()) < threshold:
        below = work < threshold
        correction = correction - mx.where(below, 1.0 / work, 0.0)
        work = work + mx.where(below, 1.0, 0.0)

    inv = 1.0 / work
    inv2 = inv * inv
    # Horner form of the Bernoulli-number asymptotic series.
    series = inv2 * (1.0 / 12 - inv2 * (1.0 / 120 - inv2 * (1.0 / 252 - inv2 / 240)))
    psi_pos = correction + mx.log(work) - 0.5 * inv - series

    reflected = psi_pos - math.pi / mx.tan(math.pi * x)
    return mx.where(reflect, reflected, psi_pos)
