# Copyright © 2023 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests


class TestQuantized(mlx_tests.MLXTestCase):
    def test_quantize_dequantize(self):
        w = mx.random.normal(shape=(128, 128))
        for b in [2, 4, 8]:
            w_q, scales, biases = mx.quantize(w, 64, b)
            w_hat = mx.dequantize(w_q, scales, biases, 64, b)
            errors = (w - w_hat).abs().reshape(*scales.shape, -1)
            self.assertTrue((errors <= scales[..., None] / 2).all())

    def test_qmm(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        for group_size in [128, 64]:
            for bits in [2, 4, 8]:
                for M in [8, 32, 33, 64]:
                    for N in [512, 1024]:
                        for K in [512, 1024]:
                            with self.subTest(
                                shape=(M, N, K), group_size=group_size, bits=bits
                            ):
                                x = mx.random.normal(shape=(M, K), key=k1)
                                w = mx.random.normal(shape=(N, K), key=k2)
                                w_q, scales, biases = mx.quantize(w, group_size, bits)
                                w_hat = mx.dequantize(
                                    w_q, scales, biases, group_size, bits
                                )
                                y_q = mx.quantized_matmul(
                                    x, w_q.T, scales, biases, group_size, bits
                                )
                                y_hat = x @ w_hat.T
                                self.assertEqual(y_q.shape, y_hat.shape)
                                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qmm_shapes(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        group_size = 64
        bits = 4
        w = mx.random.normal(shape=(32, 128), key=k2)
        w_q, scales, biases = mx.quantize(w, group_size, bits)
        w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
        for s in [(3, 128), (2, 1, 7, 128)]:
            x = mx.random.normal(shape=(3, 128), key=k1)
            y_q = mx.quantized_matmul(x, w_q.T, scales, biases, group_size, bits)
            y_hat = x @ w_hat.T
            self.assertEqual(y_q.shape, y_hat.shape)
            self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qmv(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        for group_size in [128, 64]:
            for bits in [2, 4, 8]:
                for M in [512, 1024]:
                    for N in [512, 1024]:
                        with self.subTest(
                            shape=(M, N), group_size=group_size, bits=bits
                        ):
                            x = mx.random.normal(shape=(1, N), key=k1)
                            w = mx.random.normal(shape=(M, N), key=k2)
                            w_q, scales, biases = mx.quantize(w, group_size, bits)
                            w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                            y_q = mx.quantized_matmul(
                                x, w_q.T, scales, biases, group_size, bits
                            )
                            y_hat = x @ w_hat.T
                            self.assertEqual(y_q.shape, y_hat.shape)
                            self.assertLess((y_q - y_hat).abs().max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
