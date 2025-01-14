# Copyright © 2024 Apple Inc.

import math
import os
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestDouble(mlx_tests.MLXTestCase):
    def test_unary_ops(self):
        shape = (3, 3)
        x = mx.random.normal(shape=shape)

        if mx.default_device() == mx.gpu:
            with self.assertRaises(ValueError):
                x.astype(mx.float64)

        x_double = x.astype(mx.float64, stream=mx.cpu)

        ops = [
            mx.abs,
            mx.ceil,
            mx.erf,
            mx.erfinv,
            mx.floor,
            mx.arccos,
            mx.arccosh,
            mx.arcsin,
            mx.arcsinh,
        ]
        for op in ops:
            if mx.default_device() == mx.gpu:
                with self.assertRaises(ValueError):
                    op(x_double)
                continue
            y = op(x)
            y_double = op(x_double)
            self.assertTrue(mx.allclose(y, y_double.astype(mx.float32, mx.cpu)))


if __name__ == "__main__":
    mx.set_default_device(mx.cpu)
    unittest.main()
