# Copyright © 2026 Apple Inc.

import math
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestInt64Scalar(mlx_tests.MLXTestCase):
    def test_arange_stop_larger_than_int32_max(self):
        n = mx.iinfo(mx.int32).max

        with mx.stream(mx.cpu):
            result = mx.arange(n - 1, n + 3)
            self.assertEqual(result.shape, (4,))
            self.assertEqual(result.dtype, mx.int32)
            self.assertEqual(result.tolist(), [n - 1, n, -2147483648, -2147483647])

    def test_arange_step_larger_than_int32_max(self):
        with mx.stream(mx.cpu):
            result = mx.arange(0, 5, 2**31)
            self.assertEqual(result.shape, (1,))
            self.assertEqual(result.tolist(), [0])

    def test_arange_values_larger_than_int32_max(self):
        with mx.stream(mx.cpu):
            result = mx.arange(2**40, 2**40 + 3)
            self.assertEqual(result.shape, (3,))
            # Values saturate to int32 max
            self.assertEqual(result.dtype, mx.int32)
            self.assertEqual(result.tolist(), [2147483647, 2147483647, 2147483647])

    def test_arange_single_arg_int64(self):
        with mx.stream(mx.cpu):
            with self.assertRaises(ValueError):
                mx.arange(2**40)

    def test_arange_int64_stop_error(self):
        with mx.stream(mx.cpu):
            INT_MAX = 2147483647
            with self.assertRaises(ValueError):
                mx.arange(0, INT_MAX + 1, 1)

    def test_arange_int64_start_stop(self):
        with mx.stream(mx.cpu):
            start = 2**32
            stop = 2**32 + 4
            result = mx.arange(start, stop)
            self.assertEqual(result.shape, (4,))
            self.assertEqual(result.dtype, mx.int32)

    def test_arange_negative_int64(self):
        with mx.stream(mx.cpu):
            start = -(2**40)
            stop = start + 5
            result = mx.arange(start, stop)
            self.assertEqual(result.shape, (5,))

    def test_arange_int64_step(self):
        with mx.stream(mx.cpu):
            step = 2**32
            result = mx.arange(0, 100, step)
            self.assertEqual(result.shape, (1,))
            self.assertEqual(result.tolist(), [0])

    def test_linspace_int64_start_stop(self):
        with mx.stream(mx.cpu):
            result = mx.linspace(0, 2**40, 5)
            self.assertEqual(result.shape, (5,))
            self.assertEqual(result.dtype, mx.float32)
            expected = [0.0, 2.74878e11, 5.49756e11, 8.24634e11, 1.09951e12]
            np.testing.assert_allclose(
                result.tolist(), expected, rtol=1e-5, err_msg="linspace values mismatch"
            )

    def test_linspace_int64_start_only(self):
        with mx.stream(mx.cpu):
            result = mx.linspace(2**40, 100, 5)
            self.assertEqual(result.shape, (5,))

    def test_linspace_int64_stop_only(self):
        with mx.stream(mx.cpu):
            result = mx.linspace(0, 2**40, 5)
            self.assertEqual(result.shape, (5,))
            self.assertEqual(result[0].item(), 0)
            self.assertEqual(result[-1].item(), 2**40)

    def test_arange_float64_values(self):
        with mx.stream(mx.cpu):
            result = mx.arange(0.0, float(2**40), float(2**40) / 4)
            self.assertEqual(result.shape, (4,))
            self.assertEqual(result.dtype, mx.float32)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
