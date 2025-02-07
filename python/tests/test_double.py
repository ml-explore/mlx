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
            mx.arccos,
            mx.arccosh,
            mx.arcsin,
            mx.arcsinh,
            mx.arctan,
            mx.arctanh,
            mx.ceil,
            mx.erf,
            mx.erfinv,
            mx.exp,
            mx.expm1,
            mx.floor,
            mx.log,
            mx.logical_not,
            mx.negative,
            mx.round,
            mx.sin,
            mx.sinh,
            mx.sqrt,
            mx.rsqrt,
            mx.tan,
            mx.tanh,
        ]
        for op in ops:
            if mx.default_device() == mx.gpu:
                with self.assertRaises(ValueError):
                    op(x_double)
                continue
            y = op(x)
            y_double = op(x_double)
            self.assertTrue(
                mx.allclose(y, y_double.astype(mx.float32, mx.cpu), equal_nan=True)
            )

    def test_binary_ops(self):
        shape = (3, 3)
        a = mx.random.normal(shape=shape)
        b = mx.random.normal(shape=shape)

        a_double = a.astype(mx.float64, stream=mx.cpu)
        b_double = b.astype(mx.float64, stream=mx.cpu)

        ops = [
            mx.add,
            mx.arctan2,
            mx.divide,
            mx.multiply,
            mx.subtract,
            mx.logical_and,
            mx.logical_or,
            mx.remainder,
            mx.maximum,
            mx.minimum,
            mx.power,
            mx.equal,
            mx.greater,
            mx.greater_equal,
            mx.less,
            mx.less_equal,
            mx.not_equal,
            mx.logaddexp,
        ]
        for op in ops:
            if mx.default_device() == mx.gpu:
                with self.assertRaises(ValueError):
                    op(a_double, b_double)
                continue
            y = op(a, b)
            y_double = op(a_double, b_double)
            self.assertTrue(
                mx.allclose(y, y_double.astype(mx.float32, mx.cpu), equal_nan=True)
            )

    def test_where(self):
        shape = (3, 3)
        cond = mx.random.uniform(shape=shape) > 0.5
        a = mx.random.normal(shape=shape)
        b = mx.random.normal(shape=shape)

        a_double = a.astype(mx.float64, stream=mx.cpu)
        b_double = b.astype(mx.float64, stream=mx.cpu)

        if mx.default_device() == mx.gpu:
            with self.assertRaises(ValueError):
                mx.where(cond, a_double, b_double)
            return
        y = mx.where(cond, a, b)
        y_double = mx.where(cond, a_double, b_double)
        self.assertTrue(mx.allclose(y, y_double.astype(mx.float32, mx.cpu)))

    def test_reductions(self):
        shape = (32, 32)
        a = mx.random.normal(shape=shape)
        a_double = a.astype(mx.float64, stream=mx.cpu)

        axes = [0, 1, (0, 1)]
        ops = [mx.sum, mx.prod, mx.min, mx.max, mx.any, mx.all]

        for op in ops:
            for ax in axes:
                if mx.default_device() == mx.gpu:
                    with self.assertRaises(ValueError):
                        op(a_double, axis=ax)
                    continue
                y = op(a)
                y_double = op(a_double)
                self.assertTrue(mx.allclose(y, y_double.astype(mx.float32, mx.cpu)))

    def test_get_and_set_item(self):
        shape = (3, 3)
        a = mx.random.normal(shape=shape)
        b = mx.random.normal(shape=(2,))
        a_double = a.astype(mx.float64, stream=mx.cpu)
        b_double = b.astype(mx.float64, stream=mx.cpu)
        idx_i = mx.array([0, 2])
        idx_j = mx.array([0, 2])

        if mx.default_device() == mx.gpu:
            with self.assertRaises(ValueError):
                a_double[idx_i, idx_j]
        else:
            y = a[idx_i, idx_j]
            y_double = a_double[idx_i, idx_j]
            self.assertTrue(mx.allclose(y, y_double.astype(mx.float32, mx.cpu)))

        if mx.default_device() == mx.gpu:
            with self.assertRaises(ValueError):
                a_double[idx_i, idx_j] = b_double
        else:
            a[idx_i, idx_j] = b
            a_double[idx_i, idx_j] = b_double
            self.assertTrue(mx.allclose(a, a_double.astype(mx.float32, mx.cpu)))

    def test_gemm(self):
        shape = (8, 8)
        a = mx.random.normal(shape=shape)
        b = mx.random.normal(shape=shape)

        a_double = a.astype(mx.float64, stream=mx.cpu)
        b_double = b.astype(mx.float64, stream=mx.cpu)

        if mx.default_device() == mx.gpu:
            with self.assertRaises(ValueError):
                a_double @ b_double
            return
        y = a @ b
        y_double = a_double @ b_double
        self.assertTrue(
            mx.allclose(y, y_double.astype(mx.float32, mx.cpu), equal_nan=True)
        )


if __name__ == "__main__":
    unittest.main()
