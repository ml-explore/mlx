# Copyright © 2023 Apple Inc.

import math

import mlx.core as mx
import mlx_tests
import numpy as np

# Per-dtype tolerances for comparing a gradient against a closed-form
# reference evaluated in float64.
#
# These are set from the measured agreement between MLX's float32 gradients
# and the closed forms below on well-conditioned inputs, which sits at roughly
# one float32 epsilon (~1.2e-7). The tolerances leave about two orders of
# headroom so that ordinary rounding differences across machines never trip
# them, while still being far tighter than a generic 1e-2 default.
TOLERANCES = {
    mx.float32: dict(rtol=1e-5, atol=1e-6),
    mx.float16: dict(rtol=5e-3, atol=1e-4),
}

# (name, mlx op, closed-form derivative, evaluation range)
#
# Ranges are chosen to keep every point well inside the op's domain and away
# from poles and kinks: a finite gradient compared at a non-differentiable
# point tests nothing useful. Ops whose derivative is a selection (max, abs,
# sort, ...) are deliberately not covered here for the same reason.
UNARY_OPS = [
    ("sin", mx.sin, lambda x: np.cos(x), (-1.5, 1.5)),
    ("sinh", mx.sinh, lambda x: np.cosh(x), (-1.5, 1.5)),
    ("cosh", mx.cosh, lambda x: np.sinh(x), (-1.5, 1.5)),
    ("tan", mx.tan, lambda x: 1.0 / np.cos(x) ** 2, (-1.0, 1.0)),
    ("arcsin", mx.arcsin, lambda x: 1.0 / np.sqrt(1.0 - x**2), (-0.8, 0.8)),
    ("arccos", mx.arccos, lambda x: -1.0 / np.sqrt(1.0 - x**2), (-0.8, 0.8)),
    ("arctan", mx.arctan, lambda x: 1.0 / (1.0 + x**2), (-1.5, 1.5)),
    ("arcsinh", mx.arcsinh, lambda x: 1.0 / np.sqrt(1.0 + x**2), (-1.5, 1.5)),
    ("arctanh", mx.arctanh, lambda x: 1.0 / (1.0 - x**2), (-0.8, 0.8)),
    ("expm1", mx.expm1, lambda x: np.exp(x), (-1.5, 1.5)),
    ("log1p", mx.log1p, lambda x: 1.0 / (1.0 + x), (-0.5, 1.5)),
    ("reciprocal", mx.reciprocal, lambda x: -1.0 / x**2, (0.5, 2.0)),
    ("rsqrt", mx.rsqrt, lambda x: -0.5 * x ** (-1.5), (0.5, 2.0)),
    ("erf", mx.erf, lambda x: 2.0 / np.sqrt(np.pi) * np.exp(-(x**2)), (-2.0, 2.0)),
    (
        "sigmoid",
        mx.sigmoid,
        lambda x: np.exp(-x) / (1.0 + np.exp(-x)) ** 2,
        (-2.0, 2.0),
    ),
]

# (name, mlx op, d/da, d/db, range for a, range for b)
BINARY_OPS = [
    (
        "logaddexp",
        mx.logaddexp,
        lambda a, b: 1.0 / (1.0 + np.exp(b - a)),
        lambda a, b: 1.0 / (1.0 + np.exp(a - b)),
        (-1.5, 1.5),
        (-1.5, 1.5),
    ),
    (
        "arctan2",
        mx.arctan2,
        lambda a, b: b / (a**2 + b**2),
        lambda a, b: -a / (a**2 + b**2),
        (0.5, 2.0),
        (0.5, 2.0),
    ),
]

# Enough points to catch a defect that only shows away from the origin. A
# single evaluation point can hide one: rounding the exponential in the erf
# backward pass through a lower precision, for example, is invisible at x = 0
# because exp(0) is 1 in every float format.
NUM_POINTS = 33


def _grid(lo, hi, dtype):
    x = np.linspace(lo, hi, NUM_POINTS)
    # Skip points where a reference would be evaluated at or next to a
    # singularity of one of the ops above.
    x = x[np.abs(x) > 1e-3]
    return mx.array(x).astype(dtype)


class TestGradReference(mlx_tests.MLXTestCase):
    """Compare gradients against closed-form references at many points.

    These complement the existing autograd tests, which check a gradient
    against a hand-written expected value at one or two points. Evaluating on
    a grid catches defects whose effect depends on the input.
    """

    def _check(self, name, grad, expected, dtype):
        tol = TOLERANCES[dtype]
        self.assertEqual(grad.dtype, dtype, msg=f"{name}: gradient dtype changed")
        got = np.array(grad.astype(mx.float32), dtype=np.float64)
        self.assertTrue(np.isfinite(got).all(), msg=f"{name}: gradient is not finite")
        np.testing.assert_allclose(
            got, expected, err_msg=f"{name} gradient at {dtype}", **tol
        )

    def test_unary_op_gradients(self):
        for name, op, d_op, (lo, hi) in UNARY_OPS:
            for dtype in (mx.float32, mx.float16):
                with self.subTest(op=name, dtype=dtype):
                    x = _grid(lo, hi, dtype)
                    grad = mx.grad(lambda a: mx.sum(op(a).astype(mx.float32)))(x)
                    mx.eval(grad)
                    expected = d_op(np.array(x.astype(mx.float32), dtype=np.float64))
                    self._check(name, grad, expected, dtype)

    def test_binary_op_gradients(self):
        for name, op, d_a, d_b, ra, rb in BINARY_OPS:
            for dtype in (mx.float32, mx.float16):
                with self.subTest(op=name, dtype=dtype):
                    a = _grid(*ra, dtype)
                    b = _grid(*rb, dtype)
                    ga, gb = mx.grad(
                        lambda p, q: mx.sum(op(p, q).astype(mx.float32)),
                        argnums=(0, 1),
                    )(a, b)
                    mx.eval(ga, gb)
                    an = np.array(a.astype(mx.float32), dtype=np.float64)
                    bn = np.array(b.astype(mx.float32), dtype=np.float64)
                    self._check(f"{name} d/da", ga, d_a(an, bn), dtype)
                    self._check(f"{name} d/db", gb, d_b(an, bn), dtype)

    def test_erf_gradient_away_from_zero(self):
        # The derivative of erf is 2/sqrt(pi) * exp(-x^2). At x = 0 the
        # exponential is exactly 1, so a defect in how that factor is computed
        # or stored does not show up there. Check a spread of magnitudes.
        for x in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
            with self.subTest(x=x):
                xa = mx.array([x], dtype=mx.float32)
                grad = mx.grad(lambda a: mx.sum(mx.erf(a)))(xa)
                mx.eval(grad)
                expected = 2.0 / math.sqrt(math.pi) * math.exp(-(x**2))
                self.assertAlmostEqual(grad.item(), expected, delta=1e-6)

    def test_second_order_gradients(self):
        # A wrong constant in a first derivative usually survives into the
        # second, so this is cheap extra coverage on the same references.
        cases = [
            (mx.sin, lambda x: -np.sin(x), (-1.5, 1.5)),
            (mx.sinh, lambda x: np.sinh(x), (-1.5, 1.5)),
            (mx.log1p, lambda x: -1.0 / (1.0 + x) ** 2, (-0.5, 1.5)),
        ]
        for op, d2_op, (lo, hi) in cases:
            with self.subTest(op=op.__name__):
                x = _grid(lo, hi, mx.float32)
                first = mx.grad(lambda a: mx.sum(op(a)))
                g2 = mx.grad(lambda a: mx.sum(first(a)))(x)
                mx.eval(g2)
                expected = d2_op(np.array(x, dtype=np.float64))
                np.testing.assert_allclose(
                    np.array(g2, dtype=np.float64),
                    expected,
                    **TOLERANCES[mx.float32],
                )


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
