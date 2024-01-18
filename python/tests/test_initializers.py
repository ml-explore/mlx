# Copyright © 2023 Apple Inc.
import unittest

import mlx.core as mx
import mlx.nn.initializers as init
import mlx_tests
import numpy as np


class TestInitializers(mlx_tests.MLXTestCase):
    def test_constant(self):
        value = 5.0
        dtype = mx.float32
        initializer = init.constant(value, dtype)

        for shape in [
            [
                3,
            ],
            [3, 3],
            [3, 3, 3],
        ]:
            result = initializer(mx.array(np.empty(shape)))
            with self.subTest(shape=shape):
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.dtype, dtype)

    def test_normal(self):
        mean = 0.0
        std = 1.0
        dtype = mx.float32
        initializer = init.normal(mean, std, dtype)

        for shape in [
            [
                3,
            ],
            [3, 3],
            [3, 3, 3],
        ]:
            result = initializer(mx.array(np.empty(shape)))
            with self.subTest(shape=shape):
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.dtype, dtype)

    def test_uniform(self):
        low = -1.0
        high = 1.0
        dtype = mx.float32
        initializer = init.uniform(low, high, dtype)

        for shape in [
            [
                3,
            ],
            [3, 3],
            [3, 3, 3],
        ]:
            result = initializer(mx.array(np.empty(shape)))
            with self.subTest(shape=shape):
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.dtype, dtype)
                self.assertTrue(mx.all(result >= low) and mx.all(result <= high))

    def test_glorot_normal(self):
        dtype = mx.float32
        initializer = init.glorot_normal(dtype)

        for shape in [[3, 3], [3, 3, 3]]:
            result = initializer(mx.array(np.empty(shape)))
            with self.subTest(shape=shape):
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.dtype, dtype)

    def test_glorot_uniform(self):
        dtype = mx.float32
        initializer = init.glorot_uniform(dtype)

        for shape in [[3, 3], [3, 3, 3]]:
            result = initializer(mx.array(np.empty(shape)))
            with self.subTest(shape=shape):
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.dtype, dtype)

    def test_he_normal(self):
        dtype = mx.float32
        initializer = init.he_normal(dtype)

        for shape in [[3, 3], [3, 3, 3]]:
            result = initializer(mx.array(np.empty(shape)))
            with self.subTest(shape=shape):
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.dtype, dtype)

    def test_he_uniform(self):
        dtype = mx.float32
        initializer = init.he_uniform(dtype)

        for shape in [[3, 3], [3, 3, 3]]:
            result = initializer(mx.array(np.empty(shape)))
            with self.subTest(shape=shape):
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.dtype, dtype)


if __name__ == "__main__":
    unittest.main()
