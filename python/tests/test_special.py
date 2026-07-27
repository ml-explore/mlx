# Copyright © 2024 Apple Inc.

"""Tests for :mod:`mlx.special`, checked against :mod:`scipy.special`."""

import math

import mlx.core as mx
import numpy as np
import pytest
from mlx import special
from scipy import special as sp

TOL = 1e-5
N = 50


def _np(x):
    """Convert an mlx array to a NumPy array."""
    return np.array(x)


def _max_abs_err(got, ref):
    return float(np.max(np.abs(np.asarray(got, dtype=np.float64) - ref)))


def _rng():
    return np.random.default_rng(0)


def _nonint_negatives(rng, n, low, high, margin=0.25):
    """Sample ``n`` negative reals in ``(low, high)`` bounded away from integers.

    Both ``gammaln`` and ``digamma`` have poles at the non-positive integers; in
    float32 the reflection terms (``log|sin(pi x)|`` and ``pi / tan(pi x)``) lose
    precision very close to those poles, so samples are kept a small margin away.
    """
    out = []
    while len(out) < n:
        v = rng.uniform(low, high)
        if abs(v - round(v)) > margin:
            out.append(v)
    return np.array(out, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Numerical accuracy vs. scipy on random inputs across the valid domain.
# --------------------------------------------------------------------------- #
class TestAccuracy:
    def test_erf(self):
        xs = _rng().uniform(-4.0, 4.0, N).astype(np.float32)
        got = _np(special.erf(mx.array(xs)))
        assert _max_abs_err(got, sp.erf(xs.astype(np.float64))) < TOL

    def test_erfc(self):
        xs = _rng().uniform(-4.0, 4.0, N).astype(np.float32)
        got = _np(special.erfc(mx.array(xs)))
        assert _max_abs_err(got, sp.erfc(xs.astype(np.float64))) < TOL

    def test_i0(self):
        xs = _rng().uniform(-6.0, 6.0, N).astype(np.float32)
        got = _np(special.i0(mx.array(xs)))
        assert _max_abs_err(got, sp.i0(xs.astype(np.float64))) < TOL

    def test_gammaln_positive(self):
        xs = _rng().uniform(0.1, 15.0, N).astype(np.float32)
        got = _np(special.gammaln(mx.array(xs)))
        assert _max_abs_err(got, sp.gammaln(xs.astype(np.float64))) < TOL

    def test_gammaln_negative(self):
        xs = _nonint_negatives(_rng(), N, -4.5, -0.25)
        got = _np(special.gammaln(mx.array(xs)))
        assert _max_abs_err(got, sp.gammaln(xs.astype(np.float64))) < TOL

    def test_digamma_positive(self):
        xs = _rng().uniform(0.1, 15.0, N).astype(np.float32)
        got = _np(special.digamma(mx.array(xs)))
        assert _max_abs_err(got, sp.digamma(xs.astype(np.float64))) < TOL

    def test_digamma_negative(self):
        xs = _nonint_negatives(_rng(), N, -4.5, -0.25)
        got = _np(special.digamma(mx.array(xs)))
        assert _max_abs_err(got, sp.digamma(xs.astype(np.float64))) < TOL


# --------------------------------------------------------------------------- #
# Shape preservation across scalar, 1D and 2D inputs.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fn", [special.erf, special.erfc, special.i0, special.gammaln, special.digamma]
)
@pytest.mark.parametrize("shape", [(), (7,), (3, 4)])
class TestShapePreservation:
    def test_shape(self, fn, shape):
        rng = _rng()
        # Positive domain keeps every function (incl. gammaln/digamma) well defined.
        xs = rng.uniform(0.5, 3.0, size=shape).astype(np.float32)
        out = fn(mx.array(xs))
        assert out.shape == xs.shape

        ref_fn = {
            special.erf: sp.erf,
            special.erfc: sp.erfc,
            special.i0: sp.i0,
            special.gammaln: sp.gammaln,
            special.digamma: sp.digamma,
        }[fn]
        assert _max_abs_err(_np(out), ref_fn(xs.astype(np.float64))) < TOL


# --------------------------------------------------------------------------- #
# Edge cases: 0, +inf, -inf where applicable.
# --------------------------------------------------------------------------- #
class TestEdgeCases:
    def test_erf_limits(self):
        out = _np(special.erf(mx.array([0.0, math.inf, -math.inf], dtype=mx.float32)))
        assert out[0] == pytest.approx(0.0, abs=TOL)
        assert out[1] == pytest.approx(1.0, abs=TOL)
        assert out[2] == pytest.approx(-1.0, abs=TOL)

    def test_erfc_limits(self):
        out = _np(special.erfc(mx.array([0.0, math.inf, -math.inf], dtype=mx.float32)))
        assert out[0] == pytest.approx(1.0, abs=TOL)
        assert out[1] == pytest.approx(0.0, abs=TOL)
        assert out[2] == pytest.approx(2.0, abs=TOL)

    def test_i0_limits(self):
        out = _np(special.i0(mx.array([0.0, math.inf], dtype=mx.float32)))
        assert out[0] == pytest.approx(1.0, abs=TOL)
        assert np.isposinf(out[1])

    def test_gammaln_known_values(self):
        # Gamma(1) = Gamma(2) = 1  ->  gammaln = 0; Gamma(5) = 24.
        out = _np(special.gammaln(mx.array([1.0, 2.0, 5.0], dtype=mx.float32)))
        assert out[0] == pytest.approx(0.0, abs=TOL)
        assert out[1] == pytest.approx(0.0, abs=TOL)
        assert out[2] == pytest.approx(math.log(24.0), abs=TOL)

    def test_gammaln_inf(self):
        out = _np(special.gammaln(mx.array([math.inf], dtype=mx.float32)))
        assert np.isposinf(out[0])

    def test_digamma_known_value(self):
        # psi(1) = -gamma (Euler-Mascheroni).
        out = _np(special.digamma(mx.array([1.0], dtype=mx.float32)))
        assert out[0] == pytest.approx(-0.5772156649, abs=TOL)

    def test_digamma_inf(self):
        out = _np(special.digamma(mx.array([math.inf], dtype=mx.float32)))
        assert np.isposinf(out[0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
