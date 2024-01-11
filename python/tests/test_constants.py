# Copyright © 2023 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestConstants(mlx_tests.MLXTestCase):
    def test_constants_values(self):
        # Check if mlx constants match expected values
        self.assertAlmostEqual(mx.Inf, float("inf"))
        self.assertAlmostEqual(mx.Infinity, float("inf"))
        self.assertTrue(np.isnan(mx.NAN))
        self.assertAlmostEqual(mx.NINF, float("-inf"))
        self.assertEqual(mx.NZERO, -0.0)
        self.assertTrue(np.isnan(mx.NaN))
        self.assertAlmostEqual(mx.PINF, float("inf"))
        self.assertEqual(mx.PZERO, 0.0)
        self.assertAlmostEqual(
            mx.e, 2.71828182845904523536028747135266249775724709369995
        )
        self.assertAlmostEqual(
            mx.euler_gamma, 0.5772156649015328606065120900824024310421
        )
        self.assertAlmostEqual(mx.inf, float("inf"))
        self.assertAlmostEqual(mx.infty, float("inf"))
        self.assertTrue(np.isnan(mx.nan))
        self.assertIsNone(mx.newaxis)
        self.assertAlmostEqual(mx.pi, 3.1415926535897932384626433)

    def test_constants_availability(self):
        # Check if mlx constants are available
        self.assertTrue(hasattr(mx, "Inf"))
        self.assertTrue(hasattr(mx, "Infinity"))
        self.assertTrue(hasattr(mx, "NAN"))
        self.assertTrue(hasattr(mx, "NINF"))
        self.assertTrue(hasattr(mx, "NaN"))
        self.assertTrue(hasattr(mx, "PINF"))
        self.assertTrue(hasattr(mx, "NZERO"))
        self.assertTrue(hasattr(mx, "PZERO"))
        self.assertTrue(hasattr(mx, "e"))
        self.assertTrue(hasattr(mx, "euler_gamma"))
        self.assertTrue(hasattr(mx, "inf"))
        self.assertTrue(hasattr(mx, "infty"))
        self.assertTrue(hasattr(mx, "nan"))
        self.assertTrue(hasattr(mx, "newaxis"))
        self.assertTrue(hasattr(mx, "pi"))


if __name__ == "__main__":
    unittest.main()
