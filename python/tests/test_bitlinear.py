# Copyright © 2026 8b-is

import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx_tests
from mlx.nn.layers.bitlinear import (
    BitLinear,
    activation_quant,
    binary_weight_quant,
    weight_quant,
)


class TestBitLinear(mlx_tests.MLXTestCase):
    def test_binary_weight_quant_is_provably_two_valued(self):
        w = mx.random.normal(shape=(256, 256))
        wq = binary_weight_quant(w)
        scale = mx.abs(w).mean()
        # Every element is exactly +-scale: no third value, no drift.
        self.assertTrue(mx.allclose(mx.abs(wq), scale).item())
        # No element can be 0 by construction (mx.where has no third branch).
        self.assertEqual((wq == 0).sum().item(), 0)

    def test_weight_quant_matches_binary_up_to_measure_zero_ties(self):
        # weight_quant (sign-based) and binary_weight_quant (where-based)
        # agree everywhere except at the immeasurably rare w == mean(w) tie,
        # which sign() could send to 0 and where() always sends to +scale.
        w = mx.random.normal(shape=(256, 256))
        wq = weight_quant(w)
        wqb = binary_weight_quant(w)
        self.assertTrue(mx.array_equal(wq, wqb))

    def test_activation_quant_preserves_shape_and_bounds_error(self):
        x = mx.random.normal(shape=(4, 32)) * 10
        xq = activation_quant(x)
        self.assertEqual(xq.shape, x.shape)
        # Per-token int8 quantization error is bounded by half the
        # per-row step size (scale = 127 / max(|x|)).
        scale = 127.0 / mx.abs(x).max(axis=-1, keepdims=True)
        self.assertTrue((mx.abs(x - xq) <= (1.0 / scale) + 1e-4).all())

    def test_forward_shape(self):
        for binary in (False, True):
            with self.subTest(binary=binary):
                layer = BitLinear(32, 16, binary=binary)
                x = mx.random.normal((4, 8, 32))
                y = layer(x)
                self.assertEqual(y.shape, (4, 8, 16))

    def test_forward_without_bias(self):
        layer = BitLinear(32, 16, bias=False)
        self.assertFalse("bias" in layer)
        x = mx.random.normal((4, 32))
        y = layer(x)
        self.assertEqual(y.shape, (4, 16))

    def test_gradient_reaches_full_precision_weight(self):
        # The whole point of the straight-through estimator: gradients must
        # flow to `weight` (what the optimizer actually updates), not just
        # to the quantized forward value.
        layer = BitLinear(16, 8)
        x = mx.random.normal((4, 16))

        def loss_fn(layer, x):
            return layer(x).sum()

        _, grads = nn.value_and_grad(layer, loss_fn)(layer, x)
        self.assertEqual(grads["weight"].shape, layer.weight.shape)
        self.assertTrue((mx.abs(grads["weight"]) > 0).any())

    def test_repr_reports_binary_flag(self):
        ternary = BitLinear(8, 4)
        binary = BitLinear(8, 4, binary=True)
        self.assertIn("binary=False", str(ternary))
        self.assertIn("binary=True", str(binary))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
