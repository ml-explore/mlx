# Copyright © 2023 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


def select_bits(w, width, start):
    shift_left = 32 - (start + width)
    shift_right = shift_left + start
    return (w * (2**shift_left)) // (2**shift_right)


def dequantize(w, scales, biases, width):
    w_full = mx.concatenate(
        [select_bits(w, width, i)[..., None] for i in range(0, 32, width)], axis=-1
    )
    w_full = w_full.reshape(len(w), scales.shape[-1], -1)
    w_full = scales[..., None] * w_full + biases[..., None]
    w_full = w_full.reshape(len(w), -1)

    return w_full


def quantize(w, width, groups):
    w = w.reshape(len(w), -1, groups)
    w_max = w.max(-1, keepdims=True)
    w_min = w.min(-1, keepdims=True)
    delta = (w_max - w_min) / (2**width - 1)

    w_int = mx.array(np.round((w - w_min) / delta), dtype=mx.uint32)
    scales = delta.squeeze(-1)
    biases = w_min.squeeze(-1)

    shifts = mx.array([2**i for i in range(0, 32, width)], dtype=mx.uint32)
    w_int = w_int.reshape(len(w), -1, 32 // width)
    w_int = w_int * shifts[None, None]
    packed_w = w_int.sum(-1)

    return packed_w, scales, biases


class TestQuantized(mlx_tests.MLXTestCase):
    def setUp(self):
        super().setUp()

        if mx.default_device() != mx.gpu:
            self.skipTest("Quantization only implemented on the GPU for now")

    def test_quantize_dequantize(self):
        w = mx.random.normal(shape=(128, 128))
        w_q, scales, biases = quantize(w, 4, 64)
        w_hat = dequantize(w_q, scales, biases, 4)
        w_hat2 = dequantize(*quantize(w_hat, 4, 64), 4)
        self.assertTrue(mx.allclose(w_hat, w_hat2))

    def test_qmm(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        for groups in [128, 64]:
            for width in [2, 4, 8]:
                for M in [512, 1024]:
                    for N in [512, 1024]:
                        with self.subTest(shape=(M, N), groups=groups, width=width):
                            x = mx.random.normal(shape=(32, N), key=k1)
                            w = mx.random.normal(shape=(M, N), key=k2)
                            w_q, scales, biases = quantize(w, width, groups)
                            w_hat = dequantize(w_q, scales, biases, width)
                            y_q = mx.quantized_matmul(
                                x, w_q.T, scales, biases, width=width, groups=groups
                            )
                            y_hat = x @ w_hat.T
                            self.assertEqual(y_q.shape, y_hat.shape)
                            self.assertLess((y_q - y_hat).abs().max(), 0.1)

    def test_qmv(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        for groups in [128, 64]:
            for width in [2, 4, 8]:
                for M in [512, 1024]:
                    for N in [512, 1024]:
                        # with self.subTest(shape=(M, N), groups=groups, width=width):
                        x = mx.random.normal(shape=(1, N), key=k1)
                        w = mx.random.normal(shape=(M, N), key=k2)
                        w_q, scales, biases = quantize(w, width, groups)
                        w_hat = dequantize(w_q, scales, biases, width)
                        y_q = mx.quantized_matmul(
                            x, w_q.T, scales, biases, width=width, groups=groups
                        )
                        y_hat = x @ w_hat.T
                        self.assertEqual(y_q.shape, y_hat.shape)
                        self.assertLess((y_q - y_hat).abs().max(), 0.1)


if __name__ == "__main__":
    unittest.main()
