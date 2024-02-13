# Copyright © 2023 Apple Inc.

import unittest
from itertools import product

import mlx.core as mx
import mlx_tests


class TestQuantized(mlx_tests.MLXTestCase):
    def test_quantize_dequantize(self):
        w = mx.random.normal(shape=(128, 512))
        for gs in [32, 64, 128]:
            for b in [2, 4, 8]:
                w_q, scales, biases = mx.quantize(w, gs, b)
                w_hat = mx.dequantize(w_q, scales, biases, gs, b)
                errors = (w - w_hat).abs().reshape(*scales.shape, -1)
                eps = 1e-6
                self.assertTrue((errors <= (scales[..., None] / 2 + eps)).all())

    def test_qmm(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [128, 64, 32],  # group_size
            [2, 4, 8],  # bits
            [8, 32, 33, 64],  # M
            [512, 1024],  # N
            [512, 1024],  # K
            [True, False],  # transposed
        )
        for group_size, bits, M, N, K, transposed in tests:
            with self.subTest(
                shape=(M, N, K),
                group_size=group_size,
                bits=bits,
                transposed=transposed,
            ):
                x = mx.random.normal(shape=(M, K), key=k1)
                w = mx.random.normal(shape=(N, K) if transposed else (K, N), key=k2)
                w_q, scales, biases = mx.quantize(w, group_size, bits)
                w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                y_q = mx.quantized_matmul(
                    x, w_q, scales, biases, transposed, group_size, bits
                )
                y_hat = (x @ w_hat.T) if transposed else (x @ w_hat)
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qmm_shapes(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        group_size = 64
        bits = 4
        w = mx.random.normal(shape=(32, 256), key=k2)
        w_q, scales, biases = mx.quantize(w, group_size, bits)
        w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
        for s in [(3, 256), (2, 1, 7, 256)]:
            x = mx.random.normal(shape=s, key=k1)
            y_q = mx.quantized_matmul(x, w_q, scales, biases, True, group_size, bits)
            y_hat = x @ w_hat.T
            self.assertEqual(y_q.shape, y_hat.shape)
            self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        w = mx.random.normal(shape=(256, 256), key=k2)
        w_q, scales, biases = mx.quantize(w, group_size, bits)
        w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
        for s in [(3, 256), (2, 1, 7, 256)]:
            x = mx.random.normal(shape=s, key=k1)
            y_q = mx.quantized_matmul(x, w_q, scales, biases, False, group_size, bits)
            y_hat = x @ w_hat
            self.assertEqual(y_q.shape, y_hat.shape)
            self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qmv(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [128, 64, 32],  # group_size
            [2, 4, 8],  # bits
            [512, 1024],  # M
            [512, 1024],  # N
        )
        for group_size, bits, M, N in tests:
            with self.subTest(shape=(M, N), group_size=group_size, bits=bits):
                x = mx.random.normal(shape=(1, N), key=k1)
                w = mx.random.normal(shape=(M, N), key=k2)
                w_q, scales, biases = mx.quantize(w, group_size, bits)
                w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                y_q = mx.quantized_matmul(
                    x, w_q, scales, biases, True, group_size, bits
                )
                y_hat = x @ w_hat.T
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qvm(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [128, 64, 32],  # group_size
            [2, 4, 8],  # bits
            [512, 1024],  # M
            [512, 1024],  # N
        )
        for group_size, bits, M, N in tests:
            with self.subTest(shape=(M, N), group_size=group_size, bits=bits):
                x = mx.random.normal(shape=(1, N), key=k1)
                w = mx.random.normal(shape=(N, M), key=k2)
                w_q, scales, biases = mx.quantize(w, group_size, bits)
                w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                y_q = mx.quantized_matmul(
                    x, w_q, scales, biases, False, group_size, bits
                )
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_throw(self):
        x = mx.random.normal(shape=(10, 512))
        w = mx.random.normal(shape=(32, 512))
        w_q, scales, biases = mx.quantize(w)

        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, w_q.T, scales, biases)
        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, w_q.T, scales.T, biases)
        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, w_q, scales, biases, False)
        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, w_q, scales.T, biases.T)
        y = mx.quantized_matmul(x, w_q, scales, biases, True)
        mx.eval(y)

    def test_small_matrix(self):
        w = mx.random.normal(shape=(8, 256))
        w_q, scales, biases = mx.quantize(w)
        w_hat = mx.dequantize(w_q, scales, biases)

        # Test qmv
        x = mx.random.normal(shape=(1, 256))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qmm_t
        x = mx.random.normal(shape=(10, 256))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qmv
        x = mx.random.normal(shape=(1, 8))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=False)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qmm
        x = mx.random.normal(shape=(10, 8))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=False)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_non_multiples(self):
        w = mx.random.normal(shape=(33, 256))
        w_q, scales, biases = mx.quantize(w)
        w_hat = mx.dequantize(w_q, scales, biases)

        # Test qmv
        x = mx.random.normal(shape=(1, 256))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qmm_t
        x = mx.random.normal(shape=(10, 256))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qvm
        x = mx.random.normal(shape=(1, 33))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=False)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qmm
        x = mx.random.normal(shape=(10, 33))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=False)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Smaller than 8
        w = mx.random.normal(shape=(3, 256))
        w_q, scales, biases = mx.quantize(w)
        w_hat = mx.dequantize(w_q, scales, biases)

        # Test qmv
        x = mx.random.normal(shape=(1, 256))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qmm_t
        x = mx.random.normal(shape=(10, 256))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qvm
        x = mx.random.normal(shape=(1, 3))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=False)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test qmm
        x = mx.random.normal(shape=(10, 3))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=False)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
