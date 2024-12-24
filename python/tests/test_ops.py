# Copyright © 2023-2024 Apple Inc.

import math
import os
import unittest
from itertools import permutations, product

import mlx.core as mx
import mlx_tests
import numpy as np


class TestOps(mlx_tests.MLXTestCase):
    def test_full_ones_zeros(self):
        x = mx.full(2, 3.0)
        self.assertEqual(x.shape, (2,))
        self.assertEqual(x.tolist(), [3.0, 3.0])

        x = mx.full((2, 3), 2.0)
        self.assertEqual(x.dtype, mx.float32)
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(x.tolist(), [[2, 2, 2], [2, 2, 2]])

        x = mx.full([3, 2], mx.array([False, True]))
        self.assertEqual(x.dtype, mx.bool_)
        self.assertEqual(x.tolist(), [[False, True], [False, True], [False, True]])

        x = mx.full([3, 2], mx.array([2.0, 3.0]))
        self.assertEqual(x.tolist(), [[2, 3], [2, 3], [2, 3]])

        x = mx.zeros(2)
        self.assertEqual(x.shape, (2,))
        self.assertEqual(x.tolist(), [0.0, 0.0])

        x = mx.ones(2)
        self.assertEqual(x.shape, (2,))
        self.assertEqual(x.tolist(), [1.0, 1.0])

        for t in [mx.bool_, mx.int32, mx.float32]:
            x = mx.zeros([2, 2], t)
            self.assertEqual(x.dtype, t)
            self.assertTrue(mx.array_equal(x, mx.array([[0, 0], [0, 0]])))
            y = mx.zeros_like(x)
            self.assertEqual(y.dtype, t)
            self.assertTrue(mx.array_equal(y, x))

            x = mx.ones([2, 2], t)
            self.assertEqual(x.dtype, t)
            self.assertTrue(mx.array_equal(x, mx.array([[1, 1], [1, 1]])))
            y = mx.ones_like(x)
            self.assertEqual(y.dtype, t)
            self.assertTrue(mx.array_equal(y, x))

    def test_scalar_inputs(self):
        # Check combinations of python types
        a = mx.add(False, True)
        self.assertEqual(a.dtype, mx.bool_)
        self.assertEqual(a.item(), True)

        a = mx.add(1, 2)
        self.assertEqual(a.dtype, mx.int32)
        self.assertEqual(a.item(), 3)

        a = mx.add(1.0, 2.0)
        self.assertEqual(a.dtype, mx.float32)
        self.assertEqual(a.item(), 3.0)

        a = mx.add(True, 2)
        self.assertEqual(a.dtype, mx.int32)
        self.assertEqual(a.item(), 3)

        a = mx.add(True, 2.0)
        self.assertEqual(a.dtype, mx.float32)
        self.assertEqual(a.item(), 3.0)

        a = mx.add(1, 2.0)
        self.assertEqual(a.dtype, mx.float32)
        self.assertEqual(a.item(), 3.0)

        a = mx.add(2, True)
        self.assertEqual(a.dtype, mx.int32)
        self.assertEqual(a.item(), 3)

        a = mx.add(2.0, True)
        self.assertEqual(a.dtype, mx.float32)
        self.assertEqual(a.item(), 3.0)

        a = mx.add(2.0, 1)
        self.assertEqual(a.dtype, mx.float32)
        self.assertEqual(a.item(), 3.0)

        # Check combinations with mlx arrays
        a = mx.add(mx.array(True), False)
        self.assertEqual(a.dtype, mx.bool_)
        self.assertEqual(a.item(), True)

        a = mx.add(mx.array(1), False)
        self.assertEqual(a.dtype, mx.int32)
        self.assertEqual(a.item(), 1.0)

        # Edge case: take the type of the scalar
        a = mx.add(mx.array(True), 1)
        self.assertEqual(a.dtype, mx.int32)
        self.assertEqual(a.item(), 2)

        a = mx.add(mx.array(1.0), 1)
        self.assertEqual(a.dtype, mx.float32)
        self.assertEqual(a.item(), 2.0)

        a = mx.add(1, mx.array(1.0))
        self.assertEqual(a.dtype, mx.float32)
        self.assertEqual(a.item(), 2.0)

        binary_ops = [
            "add",
            "subtract",
            "multiply",
            "divide",
            "floor_divide",
            "remainder",
            "equal",
            "not_equal",
            "less",
            "greater",
            "less_equal",
            "greater_equal",
            "maximum",
            "minimum",
        ]

        for op in binary_ops:
            npop = getattr(np, op)
            mlxop = getattr(mx, op)

            # Avoid subtract from bool and divide by 0
            for x in [-1, 0, 1, -1.0, 1.0]:
                for y in [True, -1, 1, -1.0, 1.0]:
                    self.assertEqual(npop(x, y).item(), mlxop(x, y).item())

    def test_add(self):
        x = mx.array(1)
        y = mx.array(1)
        z = mx.add(x, y)
        self.assertEqual(z.item(), 2)

        x = mx.array(False, mx.bool_)
        z = x + 1
        self.assertEqual(z.dtype, mx.int32)
        self.assertEqual(z.item(), 1)
        z = 2 + x
        self.assertEqual(z.dtype, mx.int32)
        self.assertEqual(z.item(), 2)

        x = mx.array(1, mx.uint32)
        z = x + 3
        self.assertEqual(z.dtype, mx.uint32)
        self.assertEqual(z.item(), 4)

        z = 3 + x
        self.assertEqual(z.dtype, mx.uint32)
        self.assertEqual(z.item(), 4)

        z = x + 3.0
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 4.0)

        z = 3.0 + x
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 4.0)

        x = mx.array(1, mx.int64)
        z = x + 3
        self.assertEqual(z.dtype, mx.int64)
        self.assertEqual(z.item(), 4)
        z = 3 + x
        self.assertEqual(z.dtype, mx.int64)
        self.assertEqual(z.item(), 4)
        z = x + 3.0
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 4.0)
        z = 3.0 + x
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 4.0)

        x = mx.array(1, mx.float32)
        z = x + 3
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 4)
        z = 3 + x
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 4)

    def test_subtract(self):
        x = mx.array(4.0)
        y = mx.array(3.0)

        z = mx.subtract(x, y)
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 1.0)

        z = x - 3.0
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 1.0)

        z = 5.0 - x
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 1.0)

    def test_multiply(self):
        x = mx.array(2.0)
        y = mx.array(3.0)

        z = mx.multiply(x, y)
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 6.0)

        z = x * 3.0
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 6.0)

        z = 3.0 * x
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 6.0)

    def test_divide(self):
        x = mx.array(2.0)
        y = mx.array(4.0)

        z = mx.divide(x, y)
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 0.5)

        z = x / 4.0
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 0.5)

        z = 1.0 / x
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 0.5)

        x = x.astype(mx.float16)
        z = x / 4.0
        self.assertEqual(z.dtype, mx.float16)

        x = x.astype(mx.float16)
        z = 4.0 / x
        self.assertEqual(z.dtype, mx.float16)

        x = mx.array(5)
        y = mx.array(2)
        z = x / y
        self.assertEqual(z.dtype, mx.float32)
        self.assertEqual(z.item(), 2.5)

        z = x // y
        self.assertEqual(z.dtype, mx.int32)
        self.assertEqual(z.item(), 2)

    def test_remainder(self):
        for dt in [mx.int32, mx.float32]:
            x = mx.array(2, dtype=dt)
            y = mx.array(4, dtype=dt)

            z1 = mx.remainder(x, y)
            z2 = mx.remainder(y, x)
            self.assertEqual(z1.dtype, dt)
            self.assertEqual(z1.item(), 2)
            self.assertEqual(z2.item(), 0)

            z = x % 4
            self.assertEqual(z.dtype, dt)
            self.assertEqual(z.item(), 2)

            z = 1 % x
            self.assertEqual(z.dtype, dt)
            self.assertEqual(z.item(), 1)

            z = -1 % x
            self.assertEqual(z.dtype, dt)
            self.assertEqual(z.item(), 1)

            z = -1 % -x
            self.assertEqual(z.dtype, dt)
            self.assertEqual(z.item(), -1)

            x = mx.arange(10).astype(dt) - 5
            y = x % 5
            z = x % -5
            self.assertEqual(y.tolist(), [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
            self.assertEqual(z.tolist(), [0, -4, -3, -2, -1, 0, -4, -3, -2, -1])

    def test_comparisons(self):
        a = mx.array([0.0, 1.0, 5.0])
        b = mx.array([-1.0, 2.0, 5.0])

        self.assertEqual(mx.less(a, b).tolist(), [False, True, False])
        self.assertEqual(mx.less_equal(a, b).tolist(), [False, True, True])
        self.assertEqual(mx.greater(a, b).tolist(), [True, False, False])
        self.assertEqual(mx.greater_equal(a, b).tolist(), [True, False, True])

        self.assertEqual(mx.less(a, 5).tolist(), [True, True, False])
        self.assertEqual(mx.less(5, a).tolist(), [False, False, False])
        self.assertEqual(mx.less_equal(5, a).tolist(), [False, False, True])
        self.assertEqual(mx.greater(a, 1).tolist(), [False, False, True])
        self.assertEqual(mx.greater_equal(a, 1).tolist(), [False, True, True])

        a = mx.array([0.0, 1.0, 5.0, -1.0])
        b = mx.array([0.0, 2.0, 5.0, 3.0])
        self.assertEqual(mx.equal(a, b).tolist(), [True, False, True, False])
        self.assertEqual(mx.not_equal(a, b).tolist(), [False, True, False, True])

    def test_array_equal(self):
        x = mx.array([1, 2, 3, 4])
        y = mx.array([1, 2, 3, 4])
        self.assertTrue(mx.array_equal(x, y))

        y = mx.array([1, 2, 4, 5])
        self.assertFalse(mx.array_equal(x, y))

        y = mx.array([1, 2, 3])
        self.assertFalse(mx.array_equal(x, y))

        # Can still be equal with different types
        y = mx.array([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(mx.array_equal(x, y))

        x = mx.array([0.0, float("nan")])
        y = mx.array([0.0, float("nan")])
        self.assertFalse(mx.array_equal(x, y))
        self.assertTrue(mx.array_equal(x, y, equal_nan=True))

        for t in [mx.float32, mx.float16, mx.bfloat16, mx.complex64]:
            with self.subTest(type=t):
                x = mx.array([0.0, float("nan")]).astype(t)
                y = mx.array([0.0, float("nan")]).astype(t)
                self.assertFalse(mx.array_equal(x, y))
                self.assertTrue(mx.array_equal(x, y, equal_nan=True))

    def test_isnan(self):
        x = mx.array([0.0, float("nan")])
        self.assertEqual(mx.isnan(x).tolist(), [False, True])

        x = mx.array([0.0, float("nan")]).astype(mx.float16)
        self.assertEqual(mx.isnan(x).tolist(), [False, True])

        x = mx.array([0.0, float("nan")]).astype(mx.bfloat16)
        self.assertEqual(mx.isnan(x).tolist(), [False, True])

        x = mx.array([0.0, float("nan")]).astype(mx.complex64)
        self.assertEqual(mx.isnan(x).tolist(), [False, True])

        self.assertEqual(mx.isnan(0 * mx.array(float("inf"))).tolist(), True)

    def test_isinf(self):
        x = mx.array([0.0, float("inf")])
        self.assertEqual(mx.isinf(x).tolist(), [False, True])

        x = mx.array([0.0, float("inf")]).astype(mx.float16)
        self.assertEqual(mx.isinf(x).tolist(), [False, True])

        x = mx.array([0.0, float("inf")]).astype(mx.bfloat16)
        self.assertEqual(mx.isinf(x).tolist(), [False, True])

        x = mx.array([0.0, float("inf")]).astype(mx.complex64)
        self.assertEqual(mx.isinf(x).tolist(), [False, True])

        self.assertEqual(mx.isinf(0 * mx.array(float("inf"))).tolist(), False)

        x = mx.array([-2147483648, 0, 2147483647], dtype=mx.int32)
        result = mx.isinf(x)
        self.assertEqual(result.tolist(), [False, False, False])

        x = mx.array([-32768, 0, 32767], dtype=mx.int16)
        result = mx.isinf(x)
        self.assertEqual(result.tolist(), [False, False, False])

    def test_isfinite(self):
        x = mx.array([0.0, float("inf"), float("nan")])
        self.assertEqual(mx.isfinite(x).tolist(), [True, False, False])

        x = x.astype(mx.float16)
        self.assertEqual(mx.isfinite(x).tolist(), [True, False, False])

        x = x.astype(mx.bfloat16)
        self.assertEqual(mx.isfinite(x).tolist(), [True, False, False])

    def test_tri(self):
        for shape in [[4], [4, 4], [2, 10]]:
            for diag in [-1, 0, 1, -2]:
                self.assertCmpNumpy(shape, mx.tri, np.tri, k=diag)
        self.assertEqual(mx.tri(1, 1).dtype, mx.float32)
        self.assertEqual(mx.tri(1, 1, dtype=mx.bfloat16).dtype, mx.bfloat16)

    def test_tril(self):
        for diag in [-1, 0, 1, -2]:
            self.assertCmpNumpy([(10, 10)], mx.tril, np.tril, k=diag)

        with self.assertRaises(Exception):
            mx.tril(mx.zeros((1)))

    def test_triu(self):
        for diag in [-1, 0, 1, -2]:
            self.assertCmpNumpy([(10, 10)], mx.triu, np.triu, k=diag)
        with self.assertRaises(Exception):
            mx.triu(mx.zeros((1)))

    def test_minimum(self):
        x = mx.array([0.0, -5, 10.0])
        y = mx.array([1.0, -7.0, 3.0])

        expected = [0, -7, 3]
        self.assertListEqual(mx.minimum(x, y).tolist(), expected)

        a = mx.array([float("nan")])
        b = mx.array([0.0])
        self.assertTrue(math.isnan(mx.minimum(a, b).item()))
        self.assertTrue(math.isnan(mx.minimum(b, a).item()))

    def test_maximum(self):
        x = mx.array([0.0, -5, 10.0])
        y = mx.array([1.0, -7.0, 3.0])

        expected = [1, -5, 10]
        self.assertListEqual(mx.maximum(x, y).tolist(), expected)

        a = mx.array([float("nan")])
        b = mx.array([0.0])
        self.assertTrue(math.isnan(mx.maximum(a, b).item()))
        self.assertTrue(math.isnan(mx.maximum(b, a).item()))

    def test_floor(self):
        x = mx.array([-22.03, 19.98, -27, 9, 0.0, -np.inf, np.inf])
        expected = [-23, 19, -27, 9, 0, -np.inf, np.inf]
        self.assertListEqual(mx.floor(x).tolist(), expected)

        with self.assertRaises(ValueError):
            mx.floor(mx.array([22 + 3j, 19 + 98j]))

    def test_ceil(self):
        x = mx.array([-22.03, 19.98, -27, 9, 0.0, -np.inf, np.inf])
        expected = [-22, 20, -27, 9, 0, -np.inf, np.inf]
        self.assertListEqual(mx.ceil(x).tolist(), expected)

        with self.assertRaises(ValueError):
            mx.ceil(mx.array([22 + 3j, 19 + 98j]))

    def test_isposinf(self):
        x = mx.array([0.0, float("-inf")])
        self.assertEqual(mx.isposinf(x).tolist(), [False, False])

        x = mx.array([0.0, float("-inf")]).astype(mx.float16)
        self.assertEqual(mx.isposinf(x).tolist(), [False, False])

        x = mx.array([0.0, float("-inf")]).astype(mx.bfloat16)
        self.assertEqual(mx.isposinf(x).tolist(), [False, False])

        x = mx.array([0.0, float("-inf")]).astype(mx.complex64)
        self.assertEqual(mx.isposinf(x).tolist(), [False, False])

        self.assertEqual(mx.isposinf(0 * mx.array(float("inf"))).tolist(), False)

        x = mx.array([-2147483648, 0, 2147483647], dtype=mx.int32)
        result = mx.isposinf(x)
        self.assertEqual(result.tolist(), [False, False, False])

        x = mx.array([-32768, 0, 32767], dtype=mx.int16)
        result = mx.isposinf(x)
        self.assertEqual(result.tolist(), [False, False, False])

    def test_isneginf(self):
        x = mx.array([0.0, float("-inf")])
        self.assertEqual(mx.isneginf(x).tolist(), [False, True])

        x = mx.array([0.0, float("-inf")]).astype(mx.float16)
        self.assertEqual(mx.isneginf(x).tolist(), [False, True])

        x = mx.array([0.0, float("-inf")]).astype(mx.bfloat16)
        self.assertEqual(mx.isneginf(x).tolist(), [False, True])

        x = mx.array([0.0, float("-inf")]).astype(mx.complex64)
        self.assertEqual(mx.isneginf(x).tolist(), [False, True])

        self.assertEqual(mx.isneginf(0 * mx.array(float("inf"))).tolist(), False)

        x = mx.array([-2147483648, 0, 2147483647], dtype=mx.int32)
        result = mx.isneginf(x)
        self.assertEqual(result.tolist(), [False, False, False])

        x = mx.array([-32768, 0, 32767], dtype=mx.int16)
        result = mx.isneginf(x)
        self.assertEqual(result.tolist(), [False, False, False])

    def test_round(self):
        # float
        x = mx.array(
            [0.5, -0.5, 1.5, -1.5, -21.03, 19.98, -27, 9, 0.0, -np.inf, np.inf]
        )
        expected = [0, -0, 2, -2, -21, 20, -27, 9, 0, -np.inf, np.inf]
        self.assertListEqual(mx.round(x).tolist(), expected)

        # complex
        y = mx.round(mx.array([22.2 + 3.6j, 18.5 + 98.2j]))
        self.assertListEqual(y.tolist(), [22 + 4j, 18 + 98j])

        # decimals
        y0 = mx.round(mx.array([15, 122], mx.int32), decimals=0)
        y1 = mx.round(mx.array([15, 122], mx.int32), decimals=-1)
        y2 = mx.round(mx.array([15, 122], mx.int32), decimals=-2)
        self.assertEqual(y0.dtype, mx.int32)
        self.assertEqual(y1.dtype, mx.int32)
        self.assertEqual(y2.dtype, mx.int32)
        self.assertListEqual(y0.tolist(), [15, 122])
        self.assertListEqual(y1.tolist(), [20, 120])
        self.assertListEqual(y2.tolist(), [0, 100])

        y1 = mx.round(mx.array([1.537, 1.471], mx.float32), decimals=1)
        y2 = mx.round(mx.array([1.537, 1.471], mx.float32), decimals=2)
        self.assertTrue(mx.allclose(y1, mx.array([1.5, 1.5])))
        self.assertTrue(mx.allclose(y2, mx.array([1.54, 1.47])))

        # check round to nearest for different types
        dtypes = [mx.bfloat16, mx.float16, mx.float32]
        for dtype in dtypes:
            x = mx.arange(10, dtype=dtype) - 4.5
            x = mx.round(x)
            self.assertEqual(
                x.astype(mx.float32).tolist(),
                [-4.0, -4.0, -2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 4.0, 4.0],
            )

    def test_transpose_noargs(self):
        x = mx.array([[0, 1, 1], [1, 0, 0]])

        expected = [
            [0, 1],
            [1, 0],
            [1, 0],
        ]

        self.assertListEqual(mx.transpose(x).tolist(), expected)

    def test_transpose_axis(self):
        x = mx.array(
            [
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
            ]
        )
        expected = [
            [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]],
            [[12, 16, 20], [13, 17, 21], [14, 18, 22], [15, 19, 23]],
        ]

        self.assertListEqual(mx.transpose(x, axes=(0, 2, 1)).tolist(), expected)

    def test_move_swap_axes(self):
        x = mx.zeros((2, 3, 4))
        self.assertEqual(mx.moveaxis(x, 0, 2).shape, (3, 4, 2))
        self.assertEqual(x.moveaxis(0, 2).shape, (3, 4, 2))
        self.assertEqual(mx.swapaxes(x, 0, 2).shape, (4, 3, 2))
        self.assertEqual(x.swapaxes(0, 2).shape, (4, 3, 2))

    def test_sum(self):
        x = mx.array(
            [
                [1, 2],
                [3, 3],
            ]
        )
        self.assertEqual(mx.sum(x).item(), 9)
        y = mx.sum(x, keepdims=True)
        self.assertEqual(y, mx.array(9))
        self.assertEqual(y.shape, (1, 1))

        self.assertEqual(mx.sum(x, axis=0).tolist(), [4, 5])
        self.assertEqual(mx.sum(x, axis=1).tolist(), [3, 6])

        x_npy = np.arange(3 * 5 * 4 * 7).astype(np.float32)
        x_npy = np.reshape(x_npy, (3, 5, 4, 7))
        x_mlx = mx.array(x_npy)

        for axis in (None, 0, 1, 2, 3, (0, 1), (2, 3), (1, 2, 3)):
            sum_npy = np.sum(x_npy, axis=axis)
            sum_mlx = np.asarray(mx.sum(x_mlx, axis=axis))
            self.assertListEqual(list(sum_npy.shape), list(sum_mlx.shape))
            self.assertTrue(np.all(sum_npy == sum_mlx))

        x_npy = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.float32)
        x_mlx = mx.array(x_npy)

        y_npy = x_npy[0:4:2]
        y_npy = np.broadcast_to(y_npy, (2, 2))

        y_mlx = x_mlx[0:4:2]
        y_mlx = mx.broadcast_to(y_mlx, (2, 2))

        for axis in (None, 0, 1, (0, 1)):
            sum_npy = np.sum(y_npy, axis=axis)
            sum_mlx = np.asarray(mx.sum(y_mlx, axis=axis))
            self.assertListEqual(list(sum_npy.shape), list(sum_mlx.shape))
            self.assertTrue(np.all(sum_npy == sum_mlx))

        x_npy = (
            np.arange(3 * 2 * 3 * 3 * 3 * 3)
            .reshape(3, 2, 3, 3, 3, 3)
            .astype(np.float32)
        )
        x_mlx = mx.array(x_npy)

        y_mlx = x_mlx.sum(axis=(0, 1, 3, 4, 5))
        y_npy = x_npy.sum(axis=(0, 1, 3, 4, 5))

        self.assertTrue(np.array_equal(y_mlx, y_npy))

    def test_prod(self):
        x = mx.array(
            [
                [1, 2],
                [3, 3],
            ]
        )
        self.assertEqual(mx.prod(x).item(), 18)
        y = mx.prod(x, keepdims=True)
        self.assertEqual(y, mx.array(18))
        self.assertEqual(y.shape, (1, 1))

        self.assertEqual(mx.prod(x, axis=0).tolist(), [3, 6])
        self.assertEqual(mx.prod(x, axis=1).tolist(), [2, 9])

    def test_min_and_max(self):
        x = mx.array(
            [
                [1, 2],
                [3, 4],
            ]
        )
        self.assertEqual(mx.min(x).item(), 1)
        self.assertEqual(mx.max(x).item(), 4)
        y = mx.min(x, keepdims=True)
        self.assertEqual(y.shape, (1, 1))
        self.assertEqual(y, mx.array(1))

        y = mx.max(x, keepdims=True)
        self.assertEqual(y.shape, (1, 1))
        self.assertEqual(y, mx.array(4))

        self.assertEqual(mx.min(x, axis=0).tolist(), [1, 2])
        self.assertEqual(mx.min(x, axis=1).tolist(), [1, 3])
        self.assertEqual(mx.max(x, axis=0).tolist(), [3, 4])
        self.assertEqual(mx.max(x, axis=1).tolist(), [2, 4])

    def test_argmin_argmax(self):
        data = np.random.rand(10, 12, 13)
        x = mx.array(data)
        for op in ["argmin", "argmax"]:
            for axis in range(3):
                for kd in [True, False]:
                    a = getattr(mx, op)(x, axis, kd)
                    b = getattr(np, op)(data, axis, keepdims=kd)
                    self.assertEqual(a.tolist(), b.tolist())

        for op in ["argmin", "argmax"]:
            a = getattr(mx, op)(x, keepdims=True)
            b = getattr(np, op)(data, keepdims=True)
            self.assertEqual(a.tolist(), b.tolist())
            a = getattr(mx, op)(x)
            b = getattr(np, op)(data)
            self.assertEqual(a.item(), b)

    def test_broadcast(self):
        a_npy = np.reshape(np.arange(200), (10, 20))
        a_mlx = mx.array(a_npy)

        b_npy = np.broadcast_to(a_npy, (30, 10, 20))
        b_mlx = mx.broadcast_to(a_mlx, (30, 10, 20))
        self.assertListEqual(list(b_npy.shape), list(b_mlx.shape))
        self.assertTrue(np.array_equal(b_npy, b_mlx))

        b_npy = np.broadcast_to(a_npy, (1, 10, 20))
        b_mlx = mx.broadcast_to(a_mlx, (1, 10, 20))
        self.assertListEqual(list(b_npy.shape), list(b_mlx.shape))
        self.assertTrue(np.array_equal(b_npy, b_mlx))

        b_npy = np.broadcast_to(1, (10, 20))
        b_mlx = mx.broadcast_to(1, (10, 20))
        self.assertListEqual(list(b_npy.shape), list(b_mlx.shape))
        self.assertTrue(np.array_equal(b_npy, b_mlx))

    def test_logsumexp(self):
        x = mx.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        )
        xnp = np.array(x.tolist(), dtype=np.float32)
        expected = np.log(np.sum(np.exp(xnp)))
        self.assertTrue(math.isclose(mx.logsumexp(x).item(), expected.item()))

    def test_mean(self):
        x = mx.array(
            [
                [1, 2],
                [3, 4],
            ]
        )
        self.assertEqual(mx.mean(x).item(), 2.5)
        y = mx.mean(x, keepdims=True)
        self.assertEqual(y, mx.array(2.5))
        self.assertEqual(y.shape, (1, 1))

        self.assertEqual(mx.mean(x, axis=0).tolist(), [2, 3])
        self.assertEqual(mx.mean(x, axis=1).tolist(), [1.5, 3.5])

    def test_var(self):
        x = mx.array(
            [
                [1, 2],
                [3, 4],
            ]
        )
        self.assertEqual(mx.var(x).item(), 1.25)
        y = mx.var(x, keepdims=True)
        self.assertEqual(y, mx.array(1.25))
        self.assertEqual(y.shape, (1, 1))

        self.assertEqual(mx.var(x, axis=0).tolist(), [1.0, 1.0])
        self.assertEqual(mx.var(x, axis=1).tolist(), [0.25, 0.25])

        x = mx.array([1.0, 2.0])
        out = mx.var(x, ddof=2)
        self.assertEqual(out.item(), float("inf"))

        x = mx.array([1.0, 2.0])
        out = mx.var(x, ddof=3)
        self.assertEqual(out.item(), float("inf"))

    def test_std(self):
        x = mx.random.uniform(shape=(5, 5))
        x_np = np.array(x)
        self.assertAlmostEqual(mx.std(x).item(), x_np.std().item(), places=6)

    def test_abs(self):
        a = mx.array([-1.0, 1.0, -2.0, 3.0])
        result = mx.abs(a)
        expected = np.abs(a, dtype=np.float32)
        self.assertTrue(np.allclose(result, expected))

        self.assertTrue(np.allclose(a.abs(), abs(a)))

    def test_negative(self):
        a = mx.array([-1.0, 1.0, -2.0, 3.0])
        result = mx.negative(a)
        expected = np.negative(a, dtype=np.float32)
        self.assertTrue(np.allclose(result, expected))

    def test_sign(self):
        a = mx.array([-1.0, 1.0, 0.0, -2.0, 3.0])
        result = mx.sign(a)
        expected = np.sign(a, dtype=np.float32)
        self.assertTrue(np.allclose(result, expected))

        a = mx.array([-1.0, 1.0, 0.0, -2.0, 3.0])
        b = mx.array([-4.0, -3.0, 1.0, 0.0, 3.0])
        c = a + b * 1j
        result = mx.sign(c)
        # np.sign differs in NumPy 1 and 2 so
        # we manually implement the NumPy 2 version here.
        expected = c / np.abs(c)
        self.assertTrue(np.allclose(result, expected))

    def test_logical_not(self):
        a = mx.array([-1.0, 1.0, 0.0, 1.0, -2.0, 3.0])
        result = mx.logical_not(a)
        expected = np.logical_not(a)
        self.assertTrue(np.array_equal(result, expected))

    def test_logical_and(self):
        a = mx.array([True, False, True, False])
        b = mx.array([True, True, False, False])
        result = mx.logical_and(a, b)
        expected = np.logical_and(a, b)
        self.assertTrue(np.array_equal(result, expected))

        # test overloaded operator
        result = a & b
        self.assertTrue(np.array_equal(result, expected))

    def test_logical_or(self):
        a = mx.array([True, False, True, False])
        b = mx.array([True, True, False, False])
        result = mx.logical_or(a, b)
        expected = np.logical_or(a, b)
        self.assertTrue(np.array_equal(result, expected))

        # test overloaded operator
        result = a | b
        self.assertTrue(np.array_equal(result, expected))

    def test_square(self):
        a = mx.array([0.1, 0.5, 1.0, 10.0])
        result = mx.square(a)
        expected = np.square(a, dtype=np.float32)

        self.assertTrue(np.allclose(result, expected))

    def test_sqrt(self):
        a = mx.array([0.1, 0.5, 1.0, 10.0])
        result = mx.sqrt(a)
        expected = np.sqrt(a, dtype=np.float32)
        self.assertTrue(np.allclose(result, expected))

    def test_rsqrt(self):
        a = mx.array([0.1, 0.5, 1.0, 10.0])
        result = mx.rsqrt(a)
        expected = 1.0 / np.sqrt(a, dtype=np.float32)
        self.assertTrue(np.allclose(result, expected))

    def test_reciprocal(self):
        a = mx.array([0.1, 0.5, 1.0, 2.0])
        result = mx.reciprocal(a)
        expected = np.reciprocal(a, dtype=np.float32)
        self.assertTrue(np.allclose(result, expected))

    def test_logaddexp(self):
        a = mx.array([0, 1, 2, 9.0])
        b = mx.array([1, 0, 4, 2.5])

        result = mx.logaddexp(a, b)
        expected = np.logaddexp(a, b, dtype=np.float32)

        self.assertTrue(np.allclose(result, expected))

        a = mx.array([float("nan")])
        b = mx.array([0.0])
        self.assertTrue(math.isnan(mx.logaddexp(a, b).item()))

    def test_log(self):
        a = mx.array([1, 0.5, 10, 100])
        result = mx.log(a)
        expected = np.log(a, dtype=np.float32)

        self.assertTrue(np.allclose(result, expected))

    def test_log2(self):
        a = mx.array([0.5, 1, 2, 10, 16])
        result = mx.log2(a)
        expected = np.log2(a, dtype=np.float32)

        self.assertTrue(np.allclose(result, expected))

    def test_log10(self):
        a = mx.array([0.1, 1, 10, 20, 100])
        result = mx.log10(a)
        expected = np.log10(a, dtype=np.float32)

        self.assertTrue(np.allclose(result, expected))

    def test_exp(self):
        a = mx.array([0, 0.5, -0.5, 5])
        result = mx.exp(a)
        expected = np.exp(a, dtype=np.float32)

        self.assertTrue(np.allclose(result, expected))

    def test_expm1(self):
        a = mx.array([-88, -87, 0, 0.5, -0.5, 5, 87, 88, 89, 90])
        result = mx.expm1(a)
        errs = np.seterr(over="ignore")
        expected = np.expm1(a)
        np.seterr(over=errs["over"])
        self.assertTrue(np.allclose(result, expected, rtol=1e-3, atol=1e-4))

    def test_erf(self):
        inputs = [-5, 0.0, 0.5, 1.0, 2.0, 10.0]
        x = mx.array(inputs)
        expected = np.array([math.erf(i) for i in inputs])
        self.assertTrue(np.allclose(mx.erf(x), expected))

    def test_erfinv(self):
        inputs = [-5.0, -1.0, 0.5, 0.0, 0.5, 1.0, 5.0]
        x = mx.array(inputs)
        # Output of:
        # scipy.special.erfinv([-5.0, -1.0, 0.5, 0.0, 0.5, 1.0, 5.0])
        expected = np.array(
            [
                float("nan"),
                -float("inf"),
                0.47693628,
                0.0,
                0.47693628,
                float("inf"),
                float("nan"),
            ]
        ).astype(np.float32)
        self.assertTrue(np.allclose(mx.erfinv(x), expected, equal_nan=True))

    def test_sin(self):
        a = mx.array(
            [0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 4, 2 * math.pi]
        )
        result = mx.sin(a)
        expected = np.sin(a, dtype=np.float32)

        self.assertTrue(np.allclose(result, expected))

    def test_cos(self):
        a = mx.array(
            [0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 4, 2 * math.pi]
        )
        result = mx.cos(a)
        expected = np.cos(a, dtype=np.float32)

        self.assertTrue(np.allclose(result, expected))

    def test_degrees(self):
        a = mx.array(
            [0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 4, 2 * math.pi]
        )
        result = mx.degrees(a)
        expected = np.degrees(a, dtype=np.float32)

        self.assertTrue(np.allclose(result, expected))

    def test_radians(self):
        a = mx.array([0.0, 45.0, 90.0, 180.0, 270.0, 360.0])
        result = mx.radians(a)
        expected = np.radians(a, dtype=np.float32)

        self.assertTrue(np.allclose(result, expected))

    def test_log1p(self):
        a = mx.array([1, 0.5, 10, 100])
        result = mx.log1p(a)
        expected = np.log1p(a, dtype=np.float32)

        self.assertTrue(np.allclose(result, expected))

    def test_sigmoid(self):
        a = mx.array([0.0, 1.0, -1.0, 5.0, -5.0])
        result = mx.sigmoid(a)
        expected = 1 / (1 + np.exp(-a, dtype=np.float32))
        self.assertTrue(np.allclose(result, expected))

    def test_allclose(self):
        a = mx.array(1.0)
        b = mx.array(1.0)

        self.assertTrue(mx.allclose(a, b).item())

        b = mx.array(1.1)
        self.assertFalse(mx.allclose(a, b).item())
        self.assertTrue(mx.allclose(a, b, 0.1).item())
        self.assertFalse(mx.allclose(a, b, 0.01).item())
        self.assertTrue(mx.allclose(a, b, 0.01, 0.1).item())

        c = mx.array(float("inf"))
        self.assertTrue(mx.allclose(c, c).item())

    def test_isclose(self):
        a = mx.array([float("inf"), float("inf"), float("-inf")])
        b = mx.array([float("inf"), float("-inf"), float("-inf")])

        self.assertListEqual(mx.isclose(a, b).tolist(), [True, False, True])

        a = mx.array([np.nan])
        self.assertListEqual(mx.isclose(a, a).tolist(), [False])

        a = mx.array([np.nan])
        self.assertListEqual(mx.isclose(a, a, equal_nan=True).tolist(), [True])

    def test_all(self):
        a = mx.array([[True, False], [True, True]])

        self.assertFalse(mx.all(a).item())
        self.assertEqual(mx.all(a, keepdims=True).shape, (1, 1))
        self.assertFalse(mx.all(a, axis=[0, 1]).item())
        self.assertEqual(mx.all(a, axis=[0]).tolist(), [True, False])
        self.assertEqual(mx.all(a, axis=[1]).tolist(), [False, True])
        self.assertEqual(mx.all(a, axis=0).tolist(), [True, False])
        self.assertEqual(mx.all(a, axis=1).tolist(), [False, True])

    def test_any(self):
        a = mx.array([[True, False], [False, False]])

        self.assertTrue(mx.any(a).item())
        self.assertEqual(mx.any(a, keepdims=True).shape, (1, 1))
        self.assertTrue(mx.any(a, axis=[0, 1]).item())
        self.assertEqual(mx.any(a, axis=[0]).tolist(), [True, False])
        self.assertEqual(mx.any(a, axis=[1]).tolist(), [True, False])
        self.assertEqual(mx.any(a, axis=0).tolist(), [True, False])
        self.assertEqual(mx.any(a, axis=1).tolist(), [True, False])

    def test_stop_gradient(self):
        def func(x):
            return mx.sum(2 * x + mx.stop_gradient(3 * x))

        x = mx.array([0.0, 0.1, -3])
        expected = [2, 2, 2]

        self.assertListEqual(mx.grad(func)(x).tolist(), expected)

    def test_take(self):
        # Shape: 4 x 3 x 2
        l = [
            [[1, 3], [-2, -2], [-3, -2]],
            [[2, 4], [-3, 2], [-4, -2]],
            [[2, 3], [2, 4], [2, 1]],
            [[1, -5], [3, -1], [2, 3]],
        ]

        a = mx.array(l)
        a_npy = np.array(l)

        indices = [0, -1]
        flatten_take = mx.take(a, mx.array(indices)).tolist()
        flatten_take_expected = np.take(a_npy, np.array(indices)).tolist()
        self.assertListEqual(flatten_take, flatten_take_expected)

        indices = [-1, 2, 0]
        axis_take = mx.take(a, mx.array(indices), axis=0).tolist()
        axis_take_expected = np.take(a_npy, np.array(indices), axis=0).tolist()
        self.assertListEqual(axis_take, axis_take_expected)

        indices = [0, 0, -2]
        axis_take = mx.take(a, mx.array(indices), axis=1).tolist()
        axis_take_expected = np.take(a_npy, np.array(indices), axis=1).tolist()
        self.assertListEqual(axis_take, axis_take_expected)

        indices = [0, -1, -1]
        axis_take = mx.take(a, mx.array(indices), axis=-1).tolist()
        axis_take_expected = np.take(a_npy, np.array(indices), axis=-1).tolist()
        self.assertListEqual(axis_take, axis_take_expected)

        a_npy = np.arange(8 * 8 * 8, dtype=np.int32)
        a_npy = a_npy.reshape((8, 8, 8))
        idx_npy = np.arange(6, dtype=np.uint32)
        idx_npy = idx_npy.reshape((2, 3))
        a_mlx = mx.array(a_npy)
        idx_mlx = mx.array(idx_npy)

        a_npy_taken = np.take(a_npy, idx_npy)
        a_mlx_taken = mx.take(a_mlx, idx_mlx)
        self.assertEqual(a_npy_taken.shape, a_mlx_taken.shape)
        self.assertListEqual(a_npy_taken.tolist(), a_mlx_taken.tolist())

        a_npy_taken = np.take(a_npy, idx_npy, axis=0)
        a_mlx_taken = mx.take(a_mlx, idx_mlx, axis=0)
        self.assertEqual(a_npy_taken.shape, a_mlx_taken.shape)
        self.assertListEqual(a_npy_taken.tolist(), a_mlx_taken.tolist())

        a_npy_taken = np.take(a_npy, idx_npy, axis=1)
        a_mlx_taken = mx.take(a_mlx, idx_mlx, axis=1)
        self.assertEqual(a_npy_taken.shape, a_mlx_taken.shape)
        self.assertListEqual(a_npy_taken.tolist(), a_mlx_taken.tolist())

        a_npy_taken = np.take(a_npy, idx_npy, axis=2)
        a_mlx_taken = mx.take(a_mlx, idx_mlx, axis=2)
        self.assertEqual(a_npy_taken.shape, a_mlx_taken.shape)
        self.assertListEqual(a_npy_taken.tolist(), a_mlx_taken.tolist())

        # Take with integer index
        a = mx.arange(8).reshape(2, 4)
        out = mx.take(a, 1, axis=0)
        self.assertTrue(mx.array_equal(out, mx.array([4, 5, 6, 7])))
        out = mx.take(a, 1, axis=1)
        self.assertTrue(mx.array_equal(out, mx.array([1, 5])))

        # Take with multi-dim scalar preserves dims
        out = mx.take(a, mx.array(1), axis=0)
        self.assertEqual(out.shape, (4,))

        out = mx.take(a, mx.array([1]), axis=0)
        self.assertEqual(out.shape, (1, 4))

        out = mx.take(a, mx.array([[1]]), axis=0)
        self.assertEqual(out.shape, (1, 1, 4))

    def test_take_along_axis(self):
        a_np = np.arange(8).reshape(2, 2, 2)
        a_mlx = mx.array(a_np)
        idx_np = np.array([1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0])
        idx_mlx = mx.array(idx_np)

        for ax in [None, 0, 1, 2]:
            if ax == None:
                shape = [-1]
            else:
                shape = [2] * 3
                shape[ax] = 3
            out_np = np.take_along_axis(a_np, idx_np.reshape(shape), axis=ax)
            out_mlx = mx.take_along_axis(a_mlx, mx.reshape(idx_mlx, shape), axis=ax)
            self.assertTrue(np.array_equal(out_np, np.array(out_mlx)))

    def test_put_along_axis(self):
        for ax in [None, 0, 1, 2]:

            a_np = np.arange(16).reshape(2, 2, 4).astype(np.int32)
            a_mlx = mx.array(a_np)

            if ax == None:
                idx_np = np.random.permutation(a_np.size)
                values_np = np.random.randint(low=0, high=100, size=(16,))
            else:
                shape = list(a_np.shape)
                shape[ax] = 2
                idx_np = np.random.choice(a_np.shape[ax], replace=False, size=(2,))
                idx_np = np.expand_dims(idx_np, list(range(1, 2 - ax + 1)))
                idx_np = np.broadcast_to(idx_np, shape)
                values_np = np.random.randint(low=0, high=100, size=shape)

            idx_np.astype(np.int32)
            values_np.astype(a_np.dtype)

            idx_mlx = mx.array(idx_np)
            values_mlx = mx.array(values_np)

            np.put_along_axis(a_np, idx_np, values_np, axis=ax)
            out_mlx = mx.put_along_axis(a_mlx, idx_mlx, values_mlx, axis=ax)
            self.assertTrue(np.array_equal(a_np, out_mlx))

    def test_split(self):
        a = mx.array([1, 2, 3])
        splits = mx.split(a, 3)
        for e, x in enumerate(splits):
            self.assertEqual(x.item(), e + 1)

        a = mx.array([[1, 2], [3, 4], [5, 6]])
        x, y, z = mx.split(a, 3, axis=0)
        self.assertEqual(x.tolist(), [[1, 2]])
        self.assertEqual(y.tolist(), [[3, 4]])
        self.assertEqual(z.tolist(), [[5, 6]])

        with self.assertRaises(ValueError):
            mx.split(a, 3, axis=2)

        a = mx.arange(8)
        x, y, z = mx.split(a, [1, 5])
        self.assertEqual(x.tolist(), [0])
        self.assertEqual(y.tolist(), [1, 2, 3, 4])
        self.assertEqual(z.tolist(), [5, 6, 7])

    def test_arange_overload_dispatch(self):
        with self.assertRaises(ValueError):
            a = mx.arange(float("nan"), 1, 5)
        with self.assertRaises(ValueError):
            a = mx.arange(0, float("nan"), 5)
        with self.assertRaises(ValueError):
            a = mx.arange(0, 2, float("nan"))
        with self.assertRaises(ValueError):
            a = mx.arange(0, float("inf"), float("inf"))
        with self.assertRaises(ValueError):
            a = mx.arange(float("inf"), 1, float("inf"))
        with self.assertRaises(ValueError):
            a = mx.arange(float("inf"), 1, 5)
        with self.assertRaises(TypeError):
            INT_MAX = 2147483647
            a = mx.arange(0, INT_MAX + 1, 1)

        a = mx.arange(5)
        expected = [0, 1, 2, 3, 4]
        self.assertListEqual(a.tolist(), expected)

        a = mx.arange(1, 5)
        expected = [1, 2, 3, 4]
        self.assertListEqual(a.tolist(), expected)

        a = mx.arange(-3, step=-1)
        expected = [0, -1, -2]
        self.assertListEqual(a.tolist(), expected)

        a = mx.arange(stop=2, step=0.5)
        expected = [0, 0.5, 1.0, 1.5]
        self.assertListEqual(a.tolist(), expected)

        with self.assertRaises(TypeError):
            mx.arange(start=1, step=2)

        a = mx.arange(stop=3)
        expected = [0, 1, 2]
        self.assertListEqual(a.tolist(), expected)

    def test_arange_inferred_dtype(self):
        a = mx.arange(5)
        self.assertEqual(a.dtype, mx.int32)

        a = mx.arange(5.0)
        self.assertEqual(a.dtype, mx.float32)

        a = mx.arange(1, 3.0)
        self.assertEqual(a.dtype, mx.float32)

        a = mx.arange(1, 3, dtype=mx.float32)
        self.assertEqual(a.dtype, mx.float32)

        a = mx.arange(1, 5, 1)
        self.assertEqual(a.dtype, mx.int32)

        a = mx.arange(1.0, 5, 1)
        self.assertEqual(a.dtype, mx.float32)

        a = mx.arange(1, 5.0, 1)
        self.assertEqual(a.dtype, mx.float32)

        a = mx.arange(1, 5, 1.0)
        self.assertEqual(a.dtype, mx.float32)

        a = mx.arange(1.0, 3.0, 0.2, dtype=mx.int32)
        self.assertEqual(a.dtype, mx.int32)

    def test_arange_corner_cases_cast(self):
        a = mx.arange(0, 3, 0.2, dtype=mx.int32)
        expected = [0] * 15
        self.assertListEqual(a.tolist(), expected)
        self.assertEqual(a.dtype, mx.int32)

        a = mx.arange(-1, -4, -0.9, dtype=mx.int32)
        expected = [-1] * 4
        self.assertListEqual(a.tolist(), expected)
        self.assertEqual(a.dtype, mx.int32)

        a = mx.arange(-1, -20, -1.2, dtype=mx.int32)
        expected = [
            -1,
            -2,
            -3,
            -4,
            -5,
            -6,
            -7,
            -8,
            -9,
            -10,
            -11,
            -12,
            -13,
            -14,
            -15,
            -16,
        ]
        self.assertListEqual(a.tolist(), expected)
        self.assertEqual(a.dtype, mx.int32)

        a = mx.arange(0, 10, 100)
        expected = [0]
        self.assertListEqual(a.tolist(), expected)
        self.assertEqual(a.dtype, mx.int32)

        a = mx.arange(10, 0, 1)
        expected = []
        self.assertListEqual(a.tolist(), expected)

        a = mx.arange(10, 0, float("inf"))
        expected = []
        self.assertListEqual(a.tolist(), expected)

        a = mx.arange(0, 10, float("inf"))
        expected = [0]
        self.assertListEqual(a.tolist(), expected)

        a = mx.arange(0, -10, float("-inf"))
        expected = [0]
        self.assertListEqual(a.tolist(), expected)

    def test_unary_ops(self):
        def test_ops(npop, mlxop, x, y, atol):
            r_np = npop(x)
            r_mlx = mlxop(y)
            mx.eval(r_mlx)

            self.assertTrue(np.allclose(r_np, r_mlx, atol=atol))

        x = np.random.rand(18, 28, 38)
        for op in ["abs", "exp", "log", "square", "sqrt"]:
            with self.subTest(op=op):
                float_dtypes = [("float16", 1e-3), ("float32", 1e-6)]

                for dtype, atol in float_dtypes:
                    with self.subTest(dtype=dtype):
                        x_ = x.astype(getattr(np, dtype))
                        y_ = mx.array(x_)
                        test_ops(getattr(np, op), getattr(mx, op), x_, y_, atol)

    def test_unary_ops_from_non_array(self):
        unary_ops = [
            "abs",
            "exp",
            "log",
            "square",
            "sqrt",
            "sin",
            "cos",
            "tan",
            "sinh",
            "cosh",
            "tanh",
            "sign",
            "negative",
            "expm1",
            "arcsin",
            "arccos",
            "arctan",
            "arcsinh",
            "arctanh",
            "degrees",
            "radians",
            "log2",
            "log10",
            "log1p",
            "floor",
            "ceil",
            "conjugate",
        ]

        x = 0.5
        x_np = np.random.rand(10).astype(np.float32)
        for op in unary_ops:
            with self.subTest(op=op):
                # Test from scalar
                expected = getattr(np, op)(x)
                out = getattr(mx, op)(x)

                # Check close
                self.assertTrue(np.allclose(expected, out, equal_nan=True))

                # Test from NumPy
                expected = getattr(np, op)(x_np)
                out = getattr(mx, op)(x_np)

                # Check close
                self.assertTrue(np.allclose(expected, np.array(out), equal_nan=True))

    def test_trig_ops(self):
        def test_ops(npop, mlxop, x, y, atol):
            r_np = npop(x)
            r_mlx = mlxop(y)
            mx.eval(r_mlx)

            self.assertTrue(np.allclose(r_np, r_mlx, atol=atol))

        x = np.random.rand(9, 12, 18)
        xi = np.random.rand(9, 12, 18)
        base_ops = ["sin", "cos", "tan"]
        hyperbolic_ops = ["sinh", "cosh", "tanh"]
        all_fwd_ops = base_ops + hyperbolic_ops

        for op in all_fwd_ops:
            with self.subTest(op=op):
                float_dtypes = [("float16", 1e-3), ("float32", 1e-6)]

                for dtype, atol in float_dtypes:
                    with self.subTest(dtype=dtype):
                        x_ = x.astype(getattr(np, dtype))
                        y_ = mx.array(x_)
                        test_ops(getattr(np, op), getattr(mx, op), x_, y_, atol)

            with self.subTest(op=op):
                float_dtypes = [("complex64", 1e-5)]

                for dtype, atol in float_dtypes:
                    with self.subTest(dtype=dtype):
                        x_ = x + 1.0j * xi
                        x_ = x_.astype(getattr(np, dtype))
                        y_ = mx.array(x_)
                        test_ops(getattr(np, op), getattr(mx, op), x_, y_, atol)

            with self.subTest(op="arc" + op):
                float_dtypes = [("float16", 1e-3), ("float32", 1e-6)]
                op_inv = "arc" + op

                for dtype, atol in float_dtypes:
                    with self.subTest(dtype=dtype):
                        np_op_fwd = getattr(np, op)
                        x_ = np_op_fwd(x).astype(getattr(np, dtype))
                        y_ = mx.array(x_)
                        test_ops(getattr(np, op_inv), getattr(mx, op_inv), x_, y_, atol)

        # Test grads
        np_vjp_funcs = {
            "sin": lambda primal, cotan: cotan * np.cos(primal),
            "cos": lambda primal, cotan: -cotan * np.sin(primal),
            "tan": lambda primal, cotan: cotan / (np.cos(primal) ** 2),
            "sinh": lambda primal, cotan: cotan * np.cosh(primal),
            "cosh": lambda primal, cotan: cotan * np.sinh(primal),
            "tanh": lambda primal, cotan: cotan / (np.cosh(primal) ** 2),
            "arcsin": lambda primal, cotan: cotan / np.sqrt(1.0 - primal**2),
            "arccos": lambda primal, cotan: -cotan / np.sqrt(1.0 - primal**2),
            "arctan": lambda primal, cotan: cotan / (1.0 + primal**2),
            "arctan2": lambda primal, cotan: cotan / (1.0 + primal**2),
            "arcsinh": lambda primal, cotan: cotan / np.sqrt(primal**2 + 1),
            "arccosh": lambda primal, cotan: cotan / np.sqrt(primal**2 - 1),
            "arctanh": lambda primal, cotan: cotan / (1.0 - primal**2),
        }
        with self.subTest(name="grads"):
            for op in all_fwd_ops:
                with self.subTest(op=op):
                    primal_np = xi.astype(np.float32)
                    primal_mx = mx.array(primal_np)
                    x_ = x.astype(np.float32)
                    y_ = mx.array(x_)
                    op_ = op
                    atol_ = 1e-5

                    np_vjp = lambda x: np_vjp_funcs[op_](primal_np, x)
                    mx_vjp = lambda x: mx.vjp(getattr(mx, op_), [primal_mx], [x])[1][0]
                    test_ops(np_vjp, mx_vjp, x_, y_, atol_)

                with self.subTest(op="arc" + op):
                    np_op_fwd = getattr(np, op)
                    primal_np = np_op_fwd(xi).astype(np.float32)

                    # To avoid divide by zero error
                    if op == "cosh":
                        primal_np[np.isclose(primal_np, 1.0)] += 1e-3
                    elif op == "cos":
                        primal_np[np.isclose(primal_np, 1.0)] -= 1e-3

                    primal_mx = mx.array(primal_np)
                    x_ = x.astype(np.float32)
                    y_ = mx.array(x_)
                    op_ = "arc" + op
                    atol_ = 1e-5

                    np_vjp = lambda x: np_vjp_funcs[op_](primal_np, x)
                    mx_vjp = lambda x: mx.vjp(getattr(mx, op_), [primal_mx], [x])[1][0]
                    test_ops(np_vjp, mx_vjp, x_, y_, atol_)

    def test_binary_ops(self):
        def test_ops(npop, mlxop, x1, x2, y1, y2, atol):
            r_np = npop(x1, x2)
            r_mlx = mlxop(y1, y2)
            mx.eval(r_mlx)
            self.assertTrue(np.allclose(r_np, r_mlx, atol=atol))

            r_np = npop(x1[:1], x2)
            r_mlx = mlxop(y1[:1], y2)
            mx.eval(r_mlx)
            self.assertTrue(np.allclose(r_np, r_mlx, atol=atol))

            r_np = npop(x1[:, :1], x2)
            r_mlx = mlxop(y1[:, :1], y2)
            mx.eval(r_mlx)
            self.assertTrue(np.allclose(r_np, r_mlx, atol=atol))

            r_np = npop(x1[:, :, :1], x2)
            r_mlx = mlxop(y1[:, :, :1], y2)
            mx.eval(r_mlx)
            self.assertTrue(np.allclose(r_np, r_mlx, atol=atol))

        x1 = np.maximum(np.random.rand(18, 28, 38), 0.1)
        x2 = np.maximum(np.random.rand(18, 28, 38), 0.1)
        y1 = mx.array(x1)
        y2 = mx.array(x2)
        mx.eval(y1, y2)
        for op in [
            "add",
            "subtract",
            "multiply",
            "divide",
            "floor_divide",
            "maximum",
            "minimum",
            "power",
        ]:
            with self.subTest(op=op):
                int_dtypes = [
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                ]
                float_dtypes = ["float16", "float32"]

                dtypes = {
                    "divide": float_dtypes,
                    "power": float_dtypes,
                    "floor_divide": ["float32"] + int_dtypes,
                }
                dtypes = dtypes.get(op, int_dtypes + float_dtypes)

                for dtype in dtypes:
                    atol = 1e-3 if dtype == "float16" else 1e-6
                    with self.subTest(dtype=dtype):
                        m = 10 if dtype in int_dtypes else 1
                        x1_ = (x1 * m).astype(getattr(np, dtype))
                        x2_ = (x2 * m).astype(getattr(np, dtype))
                        y1_ = mx.array(x1_)
                        y2_ = mx.array(x2_)
                        test_ops(
                            getattr(np, op), getattr(mx, op), x1_, x2_, y1_, y2_, atol
                        )

    def test_irregular_binary_ops(self):
        # Check transposed binary ops
        dims = [2, 3, 4, 5]
        size = 3
        trial_mul = 2
        np.random.seed(0)
        for d in dims:
            anp = np.random.randint(-20, 20, (size**d,)).reshape([size] * d)
            bnp = np.random.randint(-20, 20, (size**d,)).reshape([size] * d)
            for _ in range(trial_mul * d):
                amlx = mx.array(anp)
                bmlx = mx.array(bnp)
                a_t = np.random.permutation(d).tolist()
                b_t = np.random.permutation(d).tolist()
                outnp = np.add(anp.transpose(a_t), bnp.transpose(b_t))
                outmlx = mx.add(mx.transpose(amlx, a_t), mx.transpose(bmlx, b_t))
                self.assertTrue(np.array_equal(outnp, outmlx))

        # Check broadcast binary ops
        for d in dims:
            anp = np.random.randint(-20, 20, (size**d,)).reshape([size] * d)
            for n_bsx in range(d):
                bnp = np.random.randint(-20, 20, (size**n_bsx,)).reshape([size] * n_bsx)
                for _ in range(trial_mul * d):
                    amlx = mx.array(anp)
                    bmlx = mx.array(bnp)
                    b_shape = [1] * (d - n_bsx) + [size] * n_bsx
                    np.random.shuffle(b_shape)
                    outnp = np.add(anp, bnp.reshape(b_shape))
                    outmlx = mx.add(amlx, mx.reshape(bmlx, b_shape))
                    self.assertTrue(np.array_equal(outnp, outmlx))

        # Check strided binary ops
        for d in dims:
            a = np.random.randint(-20, 20, (10,) * d)
            b = np.random.randint(-20, 20, (10,) * d)
            a_ = mx.array(a)
            b_ = mx.array(b)
            for t in permutations(range(d)):
                for s in range(d):
                    idx = tuple(
                        [slice(None)] * s
                        + [slice(None, None, 2)]
                        + [slice(None)] * (d - s - 1)
                    )
                    c = a.transpose(t)[idx] + b[idx]
                    c_ = mx.transpose(a_, t)[idx] + b_[idx]
                    self.assertTrue(np.array_equal(c, c_))

    def test_softmax(self):
        cases = [(np.float32, 1e-6), (np.float16, 1e-3)]

        for dtype, atol in cases:
            a_npy = np.random.randn(16, 8, 32).astype(dtype)
            a_mlx = mx.array(a_npy)

            def np_softmax(x, axis):
                ex = np.exp(x - np.max(x, axis=axis, keepdims=True))
                return ex / np.sum(ex, axis=axis, keepdims=True)

            for axes in (None, 0, 1, 2, (0, 1), (1, 2), (0, 2), (0, 1, 2)):
                b_npy = np_softmax(a_npy, axes)
                b_mlx = mx.softmax(a_mlx, axes)
                self.assertTrue(np.allclose(b_npy, b_mlx, atol=atol))

        for s in [100, 2049, 4097, 8193]:
            a = np.full(s, -np.inf)
            a[-1] = 0.0
            a = mx.softmax(mx.array(a))
            self.assertFalse(np.any(np.isnan(a)))
            self.assertTrue((a[:-1] < 1e-9).all())
            self.assertEqual(a[-1], 1)

        # Sliced inputs
        y = mx.random.uniform(shape=(8, 4))
        out = mx.softmax(y[:, 0:2], axis=-1)
        self.assertAlmostEqual(out.sum().item(), 8.0, 5)

        # Precise
        for t in [mx.float16, mx.bfloat16]:
            a = (10 * mx.random.normal(shape=(1024,))).astype(t)
            out_expect = mx.softmax(a.astype(mx.float32)).astype(t)
            out = mx.softmax(a, axis=-1, precise=True)
            self.assertTrue(mx.allclose(out_expect, out))

        # All Infs give NaNs
        for n in [127, 128, 129]:
            x = mx.full((n,), vals=-float("inf"))
            self.assertTrue(mx.all(mx.isnan(mx.softmax(x))))

    def test_concatenate(self):
        a_npy = np.random.randn(32, 32, 32)
        b_npy = np.random.randn(32, 32, 32)
        a_mlx = mx.array(a_npy)
        b_mlx = mx.array(b_npy)

        for axis in (None, 0, 1, 2):
            for p in permutations([0, 1, 2]):
                c_npy = np.concatenate([a_npy, np.transpose(b_npy, p)], axis=axis)
                c_mlx = mx.concatenate([a_mlx, mx.transpose(b_mlx, p)], axis=axis)
                self.assertEqual(list(c_npy.shape), list(c_mlx.shape))
                self.assertTrue(np.allclose(c_npy, c_mlx, atol=1e-6))

        with self.assertRaises(ValueError):
            a = mx.array([[1, 2], [1, 2], [1, 2]])
            b = mx.array([1, 2])
            mx.concatenate([a, b], axis=0)

        # Cocnatenate with 0-sized array
        a = mx.zeros((2, 0, 2))
        b = mx.zeros((2, 2, 2))
        out = mx.concatenate([a, b], axis=1)
        self.assertTrue(mx.array_equal(out, b))

    def test_meshgrid(self):
        x = mx.array([1, 2, 3], dtype=mx.int32)
        y = np.array([1, 2, 3], dtype=np.int32)

        # Test single input
        a_mlx = mx.meshgrid(x)
        a_np = np.meshgrid(y)
        self.assertEqualArray(a_mlx[0], mx.array(a_np[0]))

        # Test sparse
        a_mlx, b_mlx, c_mlx = mx.meshgrid(x, x, x, sparse=True)
        a_np, b_np, c_np = np.meshgrid(y, y, y, sparse=True)
        self.assertEqualArray(a_mlx, mx.array(a_np))
        self.assertEqualArray(b_mlx, mx.array(b_np))
        self.assertEqualArray(c_mlx, mx.array(c_np))

        # Test different lengths
        x = mx.array([1, 2], dtype=mx.int32)
        y = mx.array([1, 2, 3], dtype=mx.int32)
        z = np.array([1, 2], dtype=np.int32)
        w = np.array([1, 2, 3], dtype=np.int32)
        a_mlx, b_mlx = mx.meshgrid(x, y)
        a_np, b_np = np.meshgrid(z, w)
        self.assertEqualArray(a_mlx, mx.array(a_np))
        self.assertEqualArray(b_mlx, mx.array(b_np))

        # Test empty input
        x = mx.array([], dtype=mx.int32)
        y = np.array([], dtype=np.int32)
        a_mlx = mx.meshgrid(x)
        a_np = np.meshgrid(y)
        self.assertEqualArray(a_mlx[0], mx.array(a_np[0]))

        # Test float32 input
        x = mx.array([1.1, 2.2, 3.3], dtype=mx.float32)
        y = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        a_mlx = mx.meshgrid(x, x, x)
        a_np = np.meshgrid(y, y, y)
        self.assertEqualArray(a_mlx[0], mx.array(a_np[0]))
        self.assertEqualArray(a_mlx[1], mx.array(a_np[1]))
        self.assertEqualArray(a_mlx[2], mx.array(a_np[2]))

        # Test ij indexing
        x = mx.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=mx.float32)
        y = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)
        a_mlx = mx.meshgrid(x, x, indexing="ij")
        a_np = np.meshgrid(y, y, indexing="ij")
        self.assertEqualArray(a_mlx[0], mx.array(a_np[0]))
        self.assertEqualArray(a_mlx[1], mx.array(a_np[1]))

        # Test different lengths, sparse, and ij indexing
        a = mx.array([1, 2], dtype=mx.int64)
        b = mx.array([1, 2, 3], dtype=mx.int64)
        c = mx.array([1, 2, 3, 4], dtype=mx.int64)
        x = np.array([1, 2], dtype=np.int64)
        y = np.array([1, 2, 3], dtype=np.int64)
        z = np.array([1, 2, 3, 4], dtype=np.int64)
        a_mlx, b_mlx, c_mlx = mx.meshgrid(a, b, c, sparse=True, indexing="ij")
        a_np, b_np, c_np = np.meshgrid(x, y, z, sparse=True, indexing="ij")
        self.assertEqualArray(a_mlx, mx.array(a_np))
        self.assertEqualArray(b_mlx, mx.array(b_np))
        self.assertEqualArray(c_mlx, mx.array(c_np))

    def test_pad(self):
        pad_width_and_values = [
            ([(1, 1), (1, 1), (1, 1)], 0),
            ([(1, 1), (1, 1), (1, 1)], 5),
            ([(3, 0), (0, 2), (5, 7)], 0),
            ([(3, 0), (0, 2), (5, 7)], -7),
            ([(0, 0), (0, 0), (0, 0)], 0),
        ]

        for pw, v in pad_width_and_values:
            with self.subTest(pad_width=pw, value=v):
                a_npy = np.random.randn(16, 16, 16).astype(np.float32)
                a_mlx = mx.array(a_npy)

                b_npy = np.pad(a_npy, pw, constant_values=v)
                b_mlx = mx.pad(a_mlx, pw, constant_values=v)

                self.assertEqual(list(b_npy.shape), list(b_mlx.shape))
                self.assertTrue(np.allclose(b_npy, b_mlx, atol=1e-6))

                b_npy = np.pad(a_npy, pw, mode="edge")
                b_mlx = mx.pad(a_mlx, pw, mode="edge")

                self.assertEqual(list(b_npy.shape), list(b_mlx.shape))
                self.assertTrue(np.allclose(b_npy, b_mlx, atol=1e-6))

        a = mx.zeros((1, 1, 1))
        self.assertEqual(mx.pad(a, 1).shape, (3, 3, 3))
        self.assertEqual(mx.pad(a, (1,)).shape, (3, 3, 3))
        self.assertEqual(mx.pad(a, [1]).shape, (3, 3, 3))
        self.assertEqual(mx.pad(a, (1, 2)).shape, (4, 4, 4))
        self.assertEqual(mx.pad(a, [(1, 2)]).shape, (4, 4, 4))
        self.assertEqual(mx.pad(a, ((1, 2),)).shape, (4, 4, 4))
        self.assertEqual(mx.pad(a, ((1, 2), (2, 1), (2, 2))).shape, (4, 4, 5))

        # Test grads
        a_fwd = mx.array(np.random.rand(16, 16).astype(np.float32))
        a_bwd = mx.ones((22, 22))
        f = lambda x: mx.pad(x, ((4, 2), (2, 4)))

        _, df = mx.vjp(f, [a_fwd], [a_bwd])
        self.assertTrue(mx.allclose(a_bwd[4:-2, 2:-4], df[0]).item())

    def test_where(self):
        self.assertCmpNumpy([True, mx.array([[1, 2], [3, 4]]), 1], mx.where, np.where)
        self.assertCmpNumpy([True, 1, mx.array([[1, 2], [3, 4]])], mx.where, np.where)
        self.assertCmpNumpy(
            [
                mx.array([[True, False], [False, True]]),
                mx.array([[1, 2], [3, 4]]),
                mx.array([5, 6]),
            ],
            mx.where,
            np.where,
        )

    def test_nan_to_num(self):
        a = mx.array([6, float("inf"), 2, 0])
        out_mx = mx.nan_to_num(a)
        out_np = np.nan_to_num(a)
        self.assertTrue(np.allclose(out_mx, out_np))

        for t in [mx.float32, mx.float16]:
            a = mx.array([float("inf"), 6.9, float("nan"), float("-inf")])
            out_mx = mx.nan_to_num(a)
            out_np = np.nan_to_num(a)
            self.assertTrue(np.allclose(out_mx, out_np))

            a = mx.array([float("inf"), 6.9, float("nan"), float("-inf")]).astype(t)
            out_np = np.nan_to_num(a, nan=0.0, posinf=1000, neginf=-1000)
            out_mx = mx.nan_to_num(a, nan=0.0, posinf=1000, neginf=-1000)
            self.assertTrue(np.allclose(out_mx, out_np))

    def test_as_strided(self):
        x_npy = np.random.randn(128).astype(np.float32)
        x_mlx = mx.array(x_npy)

        shapes = [(10, 10), (5, 5), (2, 20), (10,)]
        strides = [(3, 3), (7, 1), (1, 5), (4,)]
        for shape, stride in zip(shapes, strides):
            for offset in [0, 1, 3]:
                y_npy = np.lib.stride_tricks.as_strided(
                    x_npy[offset:], shape, np.multiply(stride, 4)
                )
                y_mlx = mx.as_strided(x_mlx, shape, stride, offset)
                self.assertTrue(np.array_equal(y_npy, y_mlx))

        x = mx.random.uniform(shape=(32,))
        y = mx.as_strided(x, (x.size,), (-1,), x.size - 1)
        self.assertTrue(mx.array_equal(y, x[::-1]))

    def test_scans(self):
        a_npy = np.random.randn(32, 32, 32).astype(np.float32)
        a_mlx = mx.array(a_npy)

        for op in ["cumsum", "cumprod"]:
            npop = getattr(np, op)
            mxop = getattr(mx, op)
            for axis in (None, 0, 1, 2):
                c_npy = npop(a_npy, axis=axis)
                c_mlx = mxop(a_mlx, axis=axis)
                self.assertTrue(np.allclose(c_npy, c_mlx, rtol=1e-3, atol=1e-3))

        a_mlx = mx.random.randint(shape=(32, 32, 32), low=-100, high=100)
        for dt in [mx.int32, mx.int64]:
            mxx = a_mlx.astype(dt)
            npx = np.array(mxx)
            for op in ["cumsum", "cumprod"]:
                npop = getattr(np, op)
                mxop = getattr(mx, op)
                for axis in (None, 0, 1, 2):
                    c_npy = npop(npx, axis=axis, dtype=npx.dtype)
                    c_mlx = mxop(mxx, axis=axis)
                    self.assertTrue(np.array_equal(c_npy, c_mlx))

        a_mlx = mx.random.randint(shape=(32, 32, 32), low=-100, high=100)
        for op in ["cumsum", "cumprod", "cummax", "cummin"]:
            mxop = getattr(mx, op)
            c1 = mxop(a_mlx, axis=2)
            c2 = mxop(a_mlx, axis=2, inclusive=False, reverse=False)
            self.assertTrue(mx.array_equal(c1[:, :, :-1], c2[:, :, 1:]))
            c1 = mxop(a_mlx, axis=1)
            c2 = mxop(a_mlx, axis=1, inclusive=False, reverse=False)
            self.assertTrue(mx.array_equal(c1[:, :-1, :], c2[:, 1:, :]))
            c1 = mxop(a_mlx, axis=0)
            c2 = mxop(a_mlx, axis=0, inclusive=False, reverse=False)
            self.assertTrue(mx.array_equal(c1[:-1, :, :], c2[1:, :, :]))

            rev_idx = mx.arange(31, -1, -1)
            c1 = mxop(a_mlx[:, :, rev_idx], axis=2)[:, :, rev_idx]
            c2 = mxop(a_mlx, axis=2, inclusive=True, reverse=True)
            self.assertTrue(mx.array_equal(c1, c2))
            c1 = mxop(a_mlx[:, rev_idx, :], axis=1)[:, rev_idx, :]
            c2 = mxop(a_mlx, axis=1, inclusive=True, reverse=True)
            self.assertTrue(mx.array_equal(c1, c2))
            c1 = mxop(a_mlx[rev_idx, :, :], axis=0)[rev_idx, :, :]
            c2 = mxop(a_mlx, axis=0, inclusive=True, reverse=True)
            self.assertTrue(mx.array_equal(c1, c2))

            rev_idx = mx.arange(31, -1, -1)
            c1 = mxop(a_mlx[:, :, rev_idx], axis=2)[:, :, rev_idx][:, :, 1:]
            c2 = mxop(a_mlx, axis=2, inclusive=False, reverse=True)[:, :, :-1]
            self.assertTrue(mx.array_equal(c1, c2))
            c1 = mxop(a_mlx[:, rev_idx, :], axis=1)[:, rev_idx, :][:, 1:, :]
            c2 = mxop(a_mlx, axis=1, inclusive=False, reverse=True)[:, :-1, :]
            self.assertTrue(mx.array_equal(c1, c2))
            c1 = mxop(a_mlx[rev_idx, :, :], axis=0)[rev_idx, :, :][1:, :, :]
            c2 = mxop(a_mlx, axis=0, inclusive=False, reverse=True)[:-1, :, :]
            self.assertTrue(mx.array_equal(c1, c2))

        a = mx.random.uniform(shape=(8, 32))
        mat = mx.tri(32)
        for t in [mx.float16, mx.bfloat16]:
            a_t = a.astype(t)
            mat_t = mat.astype(t)
            out = mx.cumsum(a_t, axis=-1)
            expected = (mat_t * a_t[:, None, :]).sum(axis=-1)
            self.assertTrue(mx.allclose(out, expected, rtol=0.02, atol=1e-3))
        sizes = [1023, 1024, 1025, 2047, 2048, 2049]
        for s in sizes:
            a = mx.ones((s,), mx.int32)
            out = mx.cumsum(a)
            expected = mx.arange(1, s + 1, dtype=mx.int32)
            self.assertTrue(mx.array_equal(expected, out))

            # non-contiguous scan
            a = mx.ones((s, 2), mx.int32)
            out = mx.cumsum(a, axis=0)
            expected = mx.repeat(expected[:, None], 2, axis=1)
            self.assertTrue(mx.array_equal(expected, out))

    def test_squeeze_expand(self):
        a = mx.zeros((2, 1, 2, 1))
        self.assertEqual(mx.squeeze(a).shape, (2, 2))
        self.assertEqual(mx.squeeze(a, 1).shape, (2, 2, 1))
        self.assertEqual(mx.squeeze(a, [1, 3]).shape, (2, 2))
        self.assertEqual(a.squeeze().shape, (2, 2))
        self.assertEqual(a.squeeze(1).shape, (2, 2, 1))
        self.assertEqual(a.squeeze([1, 3]).shape, (2, 2))

        a = mx.zeros((2, 2))
        self.assertEqual(mx.squeeze(a).shape, (2, 2))

        self.assertEqual(mx.expand_dims(a, 0).shape, (1, 2, 2))
        self.assertEqual(mx.expand_dims(a, (0, 1)).shape, (1, 1, 2, 2))
        self.assertEqual(mx.expand_dims(a, [0, -1]).shape, (1, 2, 2, 1))

    def test_sort(self):
        shape = (6, 4, 10)
        tests = product(
            ("int32", "float32"),  # type
            (None, 0, 1, 2),  # axis
            (True, False),  # strided
        )
        for dtype, axis, strided in tests:
            with self.subTest(dtype=dtype, axis=axis, strided=strided):
                np.random.seed(0)
                np_dtype = getattr(np, dtype)
                a_np = np.random.uniform(0, 100, size=shape).astype(np_dtype)
                a_mx = mx.array(a_np)
                if strided:
                    a_mx = a_mx[::2, :, ::2]
                    a_np = a_np[::2, :, ::2]

                b_np = np.sort(a_np, axis=axis)
                b_mx = mx.sort(a_mx, axis=axis)

                self.assertTrue(np.array_equal(b_np, b_mx))
                self.assertEqual(b_mx.dtype, a_mx.dtype)

                c_np = np.argsort(a_np, axis=axis)
                c_mx = mx.argsort(a_mx, axis=axis)
                d_np = np.take_along_axis(a_np, c_np, axis=axis)
                d_mx = mx.take_along_axis(a_mx, c_mx, axis=axis)

                self.assertTrue(np.array_equal(d_np, d_mx))
                self.assertEqual(c_mx.dtype, mx.uint32)

        # Set random seed
        np.random.seed(0)

        # Test multi-block sort
        for strided in (False, True):
            with self.subTest(strided=strided):
                a_np = np.random.normal(size=(32769,)).astype(np.float32)
                a_mx = mx.array(a_np)

                if strided:
                    a_mx = a_mx[::3]
                    a_np = a_np[::3]

                b_np = np.sort(a_np)
                b_mx = mx.sort(a_mx)

                self.assertTrue(np.array_equal(b_np, b_mx))
                self.assertEqual(b_mx.dtype, a_mx.dtype)

                # Test multi-dum multi-block sort
                a_np = np.random.normal(size=(2, 4, 32769)).astype(np.float32)
                a_mx = mx.array(a_np)

                if strided:
                    a_mx = a_mx[..., ::3]
                    a_np = a_np[..., ::3]

                b_np = np.sort(a_np, axis=-1)
                b_mx = mx.sort(a_mx, axis=-1)

                self.assertTrue(np.array_equal(b_np, b_mx))
                self.assertEqual(b_mx.dtype, a_mx.dtype)

                a_np = np.random.normal(size=(2, 32769, 4)).astype(np.float32)
                a_mx = mx.array(a_np)

                if strided:
                    a_mx = a_mx[:, ::3]
                    a_np = a_np[:, ::3]

                b_np = np.sort(a_np, axis=1)
                b_mx = mx.sort(a_mx, axis=1)

                self.assertTrue(np.array_equal(b_np, b_mx))
                self.assertEqual(b_mx.dtype, a_mx.dtype)

        # test 0 strides
        a_np = np.array([1, 0, 2, 1, 3, 0, 4, 0])
        a_mx = mx.array(a_np)
        b_np = np.broadcast_to(a_np, (16, 8))
        b_mx = mx.broadcast_to(a_mx, (16, 8))
        mx.eval(b_mx)
        for axis in (0, 1):
            c_np = np.sort(b_np, axis=axis)
            c_mx = mx.sort(b_mx, axis=axis)
            self.assertTrue(np.array_equal(c_np, c_mx))
            self.assertEqual(b_mx.dtype, c_mx.dtype)

        # Test very large array
        if mx.default_device() == mx.gpu:
            a_np = np.random.normal(20, 20, size=(2**22)).astype(np.float32)
            a_mx = mx.array(a_np)

            b_np = np.sort(a_np)
            b_mx = mx.sort(a_mx)
            self.assertTrue(np.array_equal(b_np, b_mx))

        # 1D strided sort
        a = mx.array([[4, 3], [2, 1], [5, 4], [3, 2]])
        out = mx.argsort(a[:, 1])
        expected = mx.array([1, 3, 0, 2], dtype=mx.uint32)
        self.assertTrue(mx.array_equal(out, expected))

    def test_partition(self):
        shape = (3, 4, 5)
        for dtype in ("int32", "float32"):
            for axis in (None, 0, 1, 2):
                for kth in (-2, 0, 2):
                    with self.subTest(dtype=dtype, axis=axis, kth=kth):
                        np.random.seed(0)
                        np_dtype = getattr(np, dtype)
                        a_np = np.random.uniform(0, 100, size=shape).astype(np_dtype)
                        a_mx = mx.array(a_np)

                        b_np = np.partition(a_np, kth, axis=axis)
                        b_mx = mx.partition(a_mx, kth, axis=axis)

                        c_np = np.take(b_np, (kth,), axis=axis)
                        c_mx = np.take(np.array(b_mx), (kth,), axis=axis)

                        self.assertTrue(np.array_equal(c_np, c_mx))
                        self.assertEqual(b_mx.dtype, a_mx.dtype)

                        if kth >= 0:
                            top_k_mx = mx.topk(a_mx, kth, axis=axis)
                            top_k_np = np.take(
                                np.partition(a_np, -kth, axis=axis), (-kth,), axis=axis
                            )
                            self.assertTrue(np.all(top_k_np <= top_k_mx))
                            self.assertEqual(top_k_mx.dtype, a_mx.dtype)
                            N = a_mx.shape[axis] if axis is not None else a_mx.size
                            M = top_k_mx.shape[axis or 0]
                            self.assertEqual(M, (kth + N) % N)

    def test_argpartition(self):
        x = mx.broadcast_to(mx.array([1, 2, 3]), (2, 3))
        out = mx.argpartition(x, kth=1, axis=0)
        expected = mx.array([[0, 0, 0], [1, 1, 1]])
        self.assertTrue(mx.array_equal(out, expected))

        x = mx.array([[1, 2], [3, 4]]).T
        out = mx.argpartition(x, kth=1, axis=0)
        expected = mx.array([[0, 0], [1, 1]])
        self.assertTrue(mx.array_equal(out, expected))

    @unittest.skipIf(
        os.getenv("LOW_MEMORY", None) is not None,
        "This test requires a lot of memory",
    )
    def test_large_binary(self):
        a = mx.ones([1000, 2147484], mx.int8)
        b = mx.ones([2147484], mx.int8)
        self.assertEqual((a + b)[0, 0].item(), 2)

    def test_eye(self):
        self.assertCmpNumpy([3], mx.eye, np.eye)
        # Test for non-square matrix
        self.assertCmpNumpy([3, 4], mx.eye, np.eye)
        # Test with positive k parameter
        self.assertCmpNumpy([3, 4], mx.eye, np.eye, k=1)
        # Test with negative k parameter
        self.assertCmpNumpy([5, 6], mx.eye, np.eye, k=-2)

    def test_stack(self):
        a = mx.ones((2,))
        np_a = np.ones((2,))
        b = mx.ones((2,))
        np_b = np.ones((2,))

        # One dimensional stack axis=0
        c = mx.stack([a, b])
        np_c = np.stack([np_a, np_b])
        self.assertTrue(np.array_equal(c, np_c))

        # One dimensional stack axis=1
        c = mx.stack([a, b], axis=1)
        np_c = np.stack([np_a, np_b], axis=1)
        self.assertTrue(np.array_equal(c, np_c))

        a = mx.ones((1, 2))
        np_a = np.ones((1, 2))
        b = mx.ones((1, 2))
        np_b = np.ones((1, 2))

        # Two dimensional stack axis=0
        c = mx.stack([a, b])
        np_c = np.stack([np_a, np_b])
        self.assertTrue(np.array_equal(c, np_c))

        # Two dimensional stack axis=1
        c = mx.stack([a, b], axis=1)
        np_c = np.stack([np_a, np_b], axis=1)
        self.assertTrue(np.array_equal(c, np_c))

    def test_flatten(self):
        x = mx.zeros([2, 3, 4])
        self.assertEqual(mx.flatten(x).shape, (2 * 3 * 4,))
        self.assertEqual(mx.flatten(x, start_axis=1).shape, (2, 3 * 4))
        self.assertEqual(mx.flatten(x, end_axis=1).shape, (2 * 3, 4))
        self.assertEqual(x.flatten().shape, (2 * 3 * 4,))
        self.assertEqual(x.flatten(start_axis=1).shape, (2, 3 * 4))
        self.assertEqual(x.flatten(end_axis=1).shape, (2 * 3, 4))

    def test_clip(self):
        a = np.array([1, 4, 3, 8, 5], np.int32)
        expected = np.clip(a, 2, 6)
        clipped = mx.clip(mx.array(a), 2, 6)
        self.assertTrue(np.array_equal(clipped, expected))

        a = np.array([-1, 1, 0, 5], np.int32)
        expected = np.clip(a, 0, None)
        clipped = mx.clip(mx.array(a), 0, None)
        self.assertTrue(np.array_equal(clipped, expected))

        a = np.array([2, 3, 4, 5], np.int32)
        expected = np.clip(a, None, 4)
        clipped = mx.clip(mx.array(a), None, 4)
        self.assertTrue(np.array_equal(clipped, expected))

        mins = np.array([3, 1, 5, 5])
        a = np.array([2, 3, 4, 5], np.int32)
        expected = np.clip(a, mins, 4)
        clipped = mx.clip(mx.array(a), mx.array(mins), 4)
        self.assertTrue(np.array_equal(clipped, expected))

        maxs = np.array([5, -1, 2, 9])
        a = np.array([2, 3, 4, 5], np.int32)
        expected = np.clip(a, mins, maxs)
        clipped = mx.clip(mx.array(a), mx.array(mins), mx.array(maxs))
        self.assertTrue(np.array_equal(clipped, expected))

        # Check clip output types
        a = mx.array([1, 2, 3], mx.int16)
        out_t = mx.clip(a, a_min=0, a_max=5).dtype
        self.assertEqual(out_t, mx.int16)

        out_t = mx.clip(a, a_min=0.0, a_max=5).dtype
        self.assertEqual(out_t, mx.float32)

        a = mx.array([1, 2, 3], mx.float16)
        out_t = mx.clip(a, a_min=0.0, a_max=5).dtype
        self.assertEqual(out_t, mx.float16)

        a = mx.array([1, 2, 3], mx.float16)
        out_t = mx.clip(a, a_min=0.0, a_max=mx.array(1.0)).dtype
        self.assertEqual(out_t, mx.float32)

    def test_linspace(self):
        # Test default num = 50
        a = mx.linspace(0, 1)
        expected = mx.array(np.linspace(0, 1))
        self.assertEqualArray(a, expected)

        # Test int64 dtype
        b = mx.linspace(0, 10, 5, mx.int64)
        expected = mx.array(np.linspace(0, 10, 5, dtype=int))
        self.assertEqualArray(b, expected)

        # Test negative sequence with float start and stop
        c = mx.linspace(-2.7, -0.7, 7)
        expected = mx.array(np.linspace(-2.7, -0.7, 7))
        self.assertEqualArray(c, expected)

        # Test irrational step size of 1/9
        d = mx.linspace(0, 1, 10)
        expected = mx.array(np.linspace(0, 1, 10))
        self.assertEqualArray(d, expected)

        # Test num equal to 1
        d = mx.linspace(1, 10, 1)
        expected = mx.array(np.linspace(1, 10, 1))
        self.assertEqualArray(d, expected)

    def test_repeat(self):
        # Setup data for the tests
        data = mx.array([[[13, 3], [16, 6]], [[14, 4], [15, 5]], [[11, 1], [12, 2]]])
        # Test repeat 0 times
        self.assertCmpNumpy([data, 0], mx.repeat, np.repeat)
        # Test repeat along axis 0
        self.assertCmpNumpy([data, 2], mx.repeat, np.repeat, axis=0)
        # Test repeat along axis 1
        self.assertCmpNumpy([data, 2], mx.repeat, np.repeat, axis=1)
        # Test repeat along the last axis (default)
        self.assertCmpNumpy([data, 2], mx.repeat, np.repeat)
        # Test repeat with a 1D array along axis 0
        self.assertCmpNumpy([mx.array([1, 3, 2]), 3], mx.repeat, np.repeat, axis=0)
        # Test repeat with a 2D array along axis 0
        self.assertCmpNumpy(
            [mx.array([[1, 2, 3], [4, 5, 4], [0, 1, 2]]), 2],
            mx.repeat,
            np.repeat,
            axis=0,
        )

    def test_tensordot(self):
        # No fp16 matmuls on common cpu backend
        if not self.is_apple_silicon:
            dtypes = [mx.float32]
        else:
            dtypes = [mx.float16, mx.float32]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                self.assertCmpNumpy(
                    [(3, 4, 5), (4, 3, 2)],
                    mx.tensordot,
                    np.tensordot,
                    dtype=dtype,
                    axes=([1, 0], [0, 1]),
                )
                self.assertCmpNumpy(
                    [(3, 4, 5), (4, 5, 6)],
                    mx.tensordot,
                    np.tensordot,
                    dtype=dtype,
                    axes=2,
                )
                self.assertCmpNumpy(
                    [(3, 5, 4, 6), (6, 4, 5, 3)],
                    mx.tensordot,
                    np.tensordot,
                    dtype=dtype,
                    axes=([2, 1, 3], [1, 2, 0]),
                )

    def test_inner(self):
        self.assertCmpNumpy([(3,), (3,)], mx.inner, np.inner)
        self.assertCmpNumpy([(1, 1, 2), (3, 2)], mx.inner, np.inner)
        self.assertCmpNumpy([(2, 3, 4), (4,)], mx.inner, np.inner)

    def test_outer(self):
        self.assertCmpNumpy([(3,), (3,)], mx.outer, np.outer)
        self.assertCmpNumpy(
            [
                mx.ones(
                    5,
                ),
                mx.linspace(-2, 2, 5),
            ],
            mx.outer,
            np.outer,
        )
        self.assertCmpNumpy(
            [
                1j * mx.linspace(2, -2, 5),
                mx.ones(
                    5,
                ),
            ],
            mx.outer,
            np.outer,
        )

    def test_divmod(self):
        # A few sizes for the inputs with and without broadcasting
        sizes = [
            ((1,), (1,)),
            ((1,), (10,)),
            ((10,), (1,)),
            ((3,), (3,)),
            ((2, 2, 2), (1, 2, 1)),
            ((2, 1, 2), (1, 2, 1)),
            ((2, 2, 2, 2), (2, 2, 2, 2)),
        ]
        types = [np.uint16, np.uint32, np.int32, np.float16, np.float32]
        for s1, s2 in sizes:
            for t in types:
                a_np = np.random.uniform(1, 100, size=s1).astype(t)
                b_np = np.random.uniform(1, 100, size=s2).astype(t)
                np_out = np.divmod(a_np, b_np)
                mx_out = mx.divmod(mx.array(a_np), mx.array(b_np))
                self.assertTrue(
                    np.allclose(np_out[0], mx_out[0]), msg=f"Shapes {s1} {s2}, Type {t}"
                )

    def test_tile(self):
        self.assertCmpNumpy([(2,), [2]], mx.tile, np.tile)
        self.assertCmpNumpy([(2, 3, 4), [2]], mx.tile, np.tile)
        self.assertCmpNumpy([(2, 3, 4), [2, 1]], mx.tile, np.tile)
        self.assertCmpNumpy(
            [
                (2, 3, 4),
                [
                    2,
                    2,
                ],
            ],
            mx.tile,
            np.tile,
        )
        self.assertCmpNumpy([(3,), [2, 2, 2]], mx.tile, np.tile)

    def test_empty_matmuls(self):
        a = mx.array([])
        b = mx.array([])
        self.assertEqual(mx.inner(a, b).item(), 0.0)

        a = mx.zeros((10, 0))
        b = mx.zeros((0, 10))
        out = a @ b
        self.assertTrue(mx.array_equal(out, mx.zeros((10, 10))))

    def test_diagonal(self):
        x = mx.array(
            [
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]],
            ]
        )
        expected = [[0, 13], [4, 17], [8, 21]]

        self.assertListEqual(mx.diagonal(x, 0, -1, 0).tolist(), expected)

        expected = [[1, 14], [5, 18], [9, 22]]
        self.assertListEqual(mx.diagonal(x, -1, 2, 0).tolist(), expected)

    def test_diag(self):
        # Test 1D input
        x = mx.array([1, 2, 3, 4])
        expected = mx.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]])
        result = mx.diag(x)
        self.assertTrue(mx.array_equal(result, expected))

        # Test 1D with offset
        x = mx.array([2, 6])
        result = mx.diag(x, k=5)
        expected = mx.array(np.diag(x, k=5))
        self.assertTrue(mx.array_equal(result, expected))

        # Test 2D input
        x = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected = mx.array([1, 5, 9])
        result = mx.diag(x)
        self.assertTrue(mx.array_equal(result, expected))

        # Test with offset
        expected = mx.array([2, 6])
        result = mx.diag(x, 1)
        self.assertTrue(mx.array_equal(result, expected))

        # Test non-square
        x = mx.array([[1, 2, 3], [4, 5, 6]])
        result = mx.diag(x)
        expected = mx.array(np.diag(x))
        self.assertTrue(mx.array_equal(result, expected))

        result = mx.diag(x, k=10)
        expected = mx.array(np.diag(x, k=10))
        self.assertTrue(mx.array_equal(result, expected))

        result = mx.diag(x, k=-10)
        expected = mx.array(np.diag(x, k=-10))
        self.assertTrue(mx.array_equal(result, expected))

        result = mx.diag(x, k=-1)
        expected = mx.array(np.diag(x, k=-1))
        self.assertTrue(mx.array_equal(result, expected))

    def test_trace(self):
        a_mx = mx.arange(9, dtype=mx.int64).reshape((3, 3))
        a_np = np.arange(9, dtype=np.int64).reshape((3, 3))

        # Test 2D array
        result = mx.trace(a_mx)
        expected = np.trace(a_np)
        self.assertEqualArray(result, mx.array(expected))

        # Test dtype
        result = mx.trace(a_mx, dtype=mx.float16)
        expected = np.trace(a_np, dtype=np.float16)
        self.assertEqualArray(result, mx.array(expected))

        # Test offset
        result = mx.trace(a_mx, offset=1)
        expected = np.trace(a_np, offset=1)
        self.assertEqualArray(result, mx.array(expected))

        # Test axis1 and axis2
        b_mx = mx.arange(27, dtype=mx.int64).reshape(3, 3, 3)
        b_np = np.arange(27, dtype=np.int64).reshape(3, 3, 3)

        result = mx.trace(b_mx, axis1=1, axis2=2)
        expected = np.trace(b_np, axis1=1, axis2=2)
        self.assertEqualArray(result, mx.array(expected))

        # Test offset, axis1, axis2, and dtype
        result = mx.trace(b_mx, offset=1, axis1=1, axis2=2, dtype=mx.float32)
        expected = np.trace(b_np, offset=1, axis1=1, axis2=2, dtype=np.float32)
        self.assertEqualArray(result, mx.array(expected))

    def test_atleast_1d(self):
        def compare_nested_lists(x, y):
            if isinstance(x, list) and isinstance(y, list):
                if len(x) != len(y):
                    return False
                for i in range(len(x)):
                    if not compare_nested_lists(x[i], y[i]):
                        return False
                return True
            else:
                return x == y

        # Test 1D input
        arrays = [
            [1],
            [1, 2, 3],
            [1, 2, 3, 4],
            [[1], [2], [3]],
            [[1, 2], [3, 4]],
            [[1, 2, 3], [4, 5, 6]],
            [[[[1]], [[2]], [[3]]]],
        ]

        mx_arrays = [mx.atleast_1d(mx.array(x)) for x in arrays]
        atleast_arrays = mx.atleast_1d(*mx_arrays)

        for i, array in enumerate(arrays):
            mx_res = mx.atleast_1d(mx.array(array))
            np_res = np.atleast_1d(np.array(array))
            self.assertTrue(compare_nested_lists(mx_res.tolist(), np_res.tolist()))
            self.assertEqual(mx_res.shape, np_res.shape)
            self.assertEqual(mx_res.ndim, np_res.ndim)
            self.assertTrue(mx.all(mx.equal(mx_res, atleast_arrays[i])))

    def test_atleast_2d(self):
        def compare_nested_lists(x, y):
            if isinstance(x, list) and isinstance(y, list):
                if len(x) != len(y):
                    return False
                for i in range(len(x)):
                    if not compare_nested_lists(x[i], y[i]):
                        return False
                return True
            else:
                return x == y

        # Test 1D input
        arrays = [
            [1],
            [1, 2, 3],
            [1, 2, 3, 4],
            [[1], [2], [3]],
            [[1, 2], [3, 4]],
            [[1, 2, 3], [4, 5, 6]],
            [[[[1]], [[2]], [[3]]]],
        ]

        mx_arrays = [mx.atleast_2d(mx.array(x)) for x in arrays]
        atleast_arrays = mx.atleast_2d(*mx_arrays)

        for i, array in enumerate(arrays):
            mx_res = mx.atleast_2d(mx.array(array))
            np_res = np.atleast_2d(np.array(array))
            self.assertTrue(compare_nested_lists(mx_res.tolist(), np_res.tolist()))
            self.assertEqual(mx_res.shape, np_res.shape)
            self.assertEqual(mx_res.ndim, np_res.ndim)
            self.assertTrue(mx.all(mx.equal(mx_res, atleast_arrays[i])))

    def test_atleast_3d(self):
        def compare_nested_lists(x, y):
            if isinstance(x, list) and isinstance(y, list):
                if len(x) != len(y):
                    return False
                for i in range(len(x)):
                    if not compare_nested_lists(x[i], y[i]):
                        return False
                return True
            else:
                return x == y

        # Test 1D input
        arrays = [
            [1],
            [1, 2, 3],
            [1, 2, 3, 4],
            [[1], [2], [3]],
            [[1, 2], [3, 4]],
            [[1, 2, 3], [4, 5, 6]],
            [[[[1]], [[2]], [[3]]]],
        ]

        mx_arrays = [mx.atleast_3d(mx.array(x)) for x in arrays]
        atleast_arrays = mx.atleast_3d(*mx_arrays)

        for i, array in enumerate(arrays):
            mx_res = mx.atleast_3d(mx.array(array))
            np_res = np.atleast_3d(np.array(array))
            self.assertTrue(compare_nested_lists(mx_res.tolist(), np_res.tolist()))
            self.assertEqual(mx_res.shape, np_res.shape)
            self.assertEqual(mx_res.ndim, np_res.ndim)
            self.assertTrue(mx.all(mx.equal(mx_res, atleast_arrays[i])))

    def test_issubdtype(self):
        self.assertTrue(mx.issubdtype(mx.bfloat16, mx.inexact))

        cats = [
            "complexfloating",
            "floating",
            "inexact",
            "signedinteger",
            "unsignedinteger",
            "integer",
            "number",
            "generic",
            "bool_",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "complex64",
        ]

        for a in cats:
            for b in cats:
                self.assertEqual(
                    mx.issubdtype(getattr(mx, a), getattr(mx, b)),
                    np.issubdtype(getattr(np, a), getattr(np, b)),
                    f"mx and np don't aggree on {a}, {b}",
                )

    def test_bitwise_ops(self):
        types = [
            mx.uint8,
            mx.uint16,
            mx.uint32,
            mx.uint64,
            mx.int8,
            mx.int16,
            mx.int32,
            mx.int64,
        ]
        a = mx.random.randint(0, 4096, (1000,))
        b = mx.random.randint(0, 4096, (1000,))
        for op in ["bitwise_and", "bitwise_or", "bitwise_xor"]:
            for t in types:
                a_mlx = a.astype(t)
                b_mlx = b.astype(t)
                a_np = np.array(a_mlx)
                b_np = np.array(b_mlx)
                out_mlx = getattr(mx, op)(a_mlx, b_mlx)
                out_np = getattr(np, op)(a_np, b_np)
                self.assertTrue(np.array_equal(np.array(out_mlx), out_np))
        for op in ["left_shift", "right_shift"]:
            for t in types:
                a_mlx = a.astype(t)
                b_mlx = mx.random.randint(0, t.size, (1000,)).astype(t)
                a_np = np.array(a_mlx)
                b_np = np.array(b_mlx)
                out_mlx = getattr(mx, op)(a_mlx, b_mlx)
                out_np = getattr(np, op)(a_np, b_np)
                self.assertTrue(np.array_equal(np.array(out_mlx), out_np))

        # Check broadcasting
        a = mx.ones((3, 1, 5), dtype=mx.bool_)
        b = mx.zeros((1, 2, 5), dtype=mx.bool_)
        c = a | b
        self.assertEqual(c.shape, (3, 2, 5))
        self.assertTrue(mx.array_equal(c, mx.ones((3, 2, 5), dtype=mx.bool_)))

    def test_bitwise_grad(self):
        a = np.random.randint(0, 10, size=(4, 3))
        b = np.random.randint(0, 10, size=(4, 3))
        cotangent = np.random.randint(0, 10, size=(4, 3))
        a = mx.array(a)
        b = mx.array(b)
        cotangent = mx.array(cotangent)

        def bitwise(a, b):
            return a.astype(mx.int32) & b.astype(mx.int32)

        _, vjps = mx.vjp(bitwise, [a, b], [cotangent])
        for vjp in vjps:
            self.assertFalse(np.any(np.array(vjp)))

    def test_conjugate(self):
        shape = (3, 5, 7)
        a = np.random.normal(size=shape) + 1j * np.random.normal(size=shape)
        a = a.astype(np.complex64)
        ops = ["conjugate", "conj"]
        for op in ops:
            out_mlx = getattr(mx, op)(mx.array(a))
            out_np = getattr(np, op)(a)
            self.assertTrue(np.array_equal(np.array(out_mlx), out_np))
        out_mlx = mx.array(a).conj()
        out_np = a.conj()
        self.assertTrue(np.array_equal(np.array(out_mlx), out_np))

    def test_view(self):
        # Check scalar
        out = mx.array(1, mx.int8).view(mx.uint8).item()
        self.assertEqual(out, 1)

        a = mx.random.randint(shape=(4, 2, 4), low=-100, high=100)
        a_np = np.array(a)

        for t in ["bool_", "int16", "float32", "int64"]:
            out = a.view(getattr(mx, t))
            expected = a_np.view(getattr(np, t))
            self.assertTrue(np.array_equal(out, expected, equal_nan=True))

        # Irregular strides
        a = mx.random.randint(shape=(2, 4), low=-100, high=100)
        a = mx.broadcast_to(a, shape=(4, 2, 4))

        for t in ["bool_", "int16", "float32", "int64"]:
            out = a.view(getattr(mx, t))
            a_out = out.view(mx.int32)
            self.assertTrue(mx.array_equal(a_out, a, equal_nan=True))

        a = mx.random.randint(shape=(4, 4), low=-100, high=100).T
        for t in ["bool_", "int16", "float32", "int64"]:
            out = a.view(getattr(mx, t))
            a_out = out.view(mx.int32)
            self.assertTrue(mx.array_equal(a_out, a, equal_nan=True))

    def _hadamard(self, N):
        # Matches scipy.linalg.hadamard
        H = np.array([[1]], dtype=np.int64)
        for i in range(0, np.log2(N).astype(np.int64)):
            H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))
        return H

    def test_hadamard(self):
        h28_str = """
        +------++----++-+--+-+--++--
        -+-----+++-----+-+--+-+--++-
        --+-----+++---+-+-+----+--++
        ---+-----+++---+-+-+-+--+--+
        ----+-----+++---+-+-+++--+--
        -----+-----++++--+-+--++--+-
        ------++----++-+--+-+--++--+
        --++++-+-------++--+++-+--+-
        ---++++-+-----+-++--+-+-+--+
        +---+++--+----++-++--+-+-+--
        ++---++---+----++-++--+-+-+-
        +++---+----+----++-++--+-+-+
        ++++--------+-+--++-++--+-+-
        -++++--------+++--++--+--+-+
        -+-++-++--++--+--------++++-
        +-+-++--+--++--+--------++++
        -+-+-++--+--++--+----+---+++
        +-+-+-++--+--+---+---++---++
        ++-+-+-++--+------+--+++---+
        -++-+-+-++--+------+-++++---
        +-++-+---++--+------+-++++--
        -++--++-+-++-+++----++------
        +-++--++-+-++-+++-----+-----
        ++-++---+-+-++-+++-----+----
        -++-++-+-+-+-+--+++-----+---
        --++-++++-+-+----+++-----+--
        +--++-+-++-+-+----+++-----+-
        ++--++-+-++-+-+----++------+
        """

        def parse_h_string(h_str):
            return np.array(
                [[1 if s == "+" else -1 for s in row] for row in h_str.split()]
            )

        h28 = parse_h_string(h28_str)

        np.random.seed(7)
        tests = product([np.float32, np.float16, np.int32], [1, 28], range(1, 15))
        for dtype, m, k in tests:
            # skip large m=28 cases because they're very slow in NumPy
            if (m > 1 and k > 8) or (dtype != np.float16 and k == 14):
                continue
            with self.subTest(dtype=dtype, m=m, k=k):
                n = m * 2**k
                b = 4
                scale = 0.34
                x = np.random.normal(size=(b, n)).astype(dtype)
                # contiguity check
                x = mx.array(x)[::2]
                y = mx.hadamard_transform(x, scale=scale)
                mx.eval(y)
                h = (
                    self._hadamard(2**k)
                    if m == 1
                    else np.kron(h28, self._hadamard(2**k))
                )
                y_np = np.einsum("ij,bj->bi", h, x) * scale
                atol = 2e-4 if dtype == np.float32 else 5e-2 * k
                np.testing.assert_allclose(y, y_np, atol=atol)

                # bfloat16 emulation on M1 means 2**14 doesn't fit in threadgroup memory
                if dtype == np.float16 and k < 14:
                    y_bf16 = mx.hadamard_transform(x.astype(mx.bfloat16), scale=scale)
                    np.testing.assert_allclose(
                        y_bf16.astype(mx.float16), y, atol=atol * 2
                    )

    def test_hadamard_grad_vmap(self):
        np.random.seed(4)

        for k in range(2, 8):
            n = 2**k
            x = np.random.normal(size=(n,))
            h = self._hadamard(n)
            c = np.random.normal(size=(n,))
            x = mx.array(x).astype(mx.float32)
            h = mx.array(h).astype(mx.float32)
            c = mx.array(c).astype(mx.float32)

            def hadamard_transform(x):
                return h @ x / mx.sqrt(x.shape[-1])

            out = mx.vjp(hadamard_transform, [x], [c])
            out_t = mx.vjp(mx.hadamard_transform, [x], [c])
            np.testing.assert_allclose(out, out_t, atol=1e-4)

            for axis in (0, 1, 2):
                vht = mx.vmap(mx.vmap(hadamard_transform, 0, 0), axis, axis)
                vht_t = mx.vmap(mx.vmap(mx.hadamard_transform, 0, 0), axis, axis)

                xb = mx.array(np.random.normal(size=(n, n, n)))
                out = vht(xb)
                out_t = vht_t(xb)
                np.testing.assert_allclose(out, out_t, atol=1e-4)

    def test_roll(self):
        x = mx.arange(10).reshape(2, 5)

        for s in [-2, -1, 0, 1, 2]:
            y1 = np.roll(x, s)
            y2 = mx.roll(x, s)
            self.assertTrue(mx.array_equal(y1, y2).item())

            y1 = np.roll(x, (s, s, s))
            y2 = mx.roll(x, (s, s, s))
            self.assertTrue(mx.array_equal(y1, y2).item())

        shifts = [
            1,
            2,
            -1,
            -2,
            (1, 1),
            (-1, 2),
            (33, 33),
        ]
        axes = [
            0,
            1,
            (1, 0),
            (0, 1),
            (0, 0),
            (1, 1),
        ]
        for s, a in product(shifts, axes):
            y1 = np.roll(x, s, a)
            y2 = mx.roll(x, s, a)
            self.assertTrue(mx.array_equal(y1, y2).item())

    def test_real_imag(self):
        x = mx.random.uniform(shape=(4, 4))
        out = mx.real(x)
        self.assertTrue(mx.array_equal(x, out))

        out = mx.imag(x)
        self.assertTrue(mx.array_equal(mx.zeros_like(x), out))

        y = mx.random.uniform(shape=(4, 4))
        z = x + 1j * y
        self.assertEqual(mx.real(z).dtype, mx.float32)
        self.assertTrue(mx.array_equal(mx.real(z), x))
        self.assertEqual(mx.imag(z).dtype, mx.float32)
        self.assertTrue(mx.array_equal(mx.imag(z), y))

    def test_broadcast_arrays(self):
        a = mx.array(1)
        b = mx.array(1.0)
        a, b = mx.broadcast_arrays(a, b)
        self.assertEqual(a.shape, ())
        self.assertEqual(a.dtype, mx.int32)
        self.assertEqual(b.shape, ())
        self.assertEqual(b.dtype, mx.float32)

        a, b = mx.broadcast_arrays(mx.zeros((3, 1, 2)), mx.zeros((4, 1)))
        self.assertEqual(a.shape, (3, 4, 2))
        self.assertEqual(b.shape, (3, 4, 2))


if __name__ == "__main__":
    unittest.main()
