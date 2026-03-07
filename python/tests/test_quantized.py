# Copyright © 2023 Apple Inc.

import unittest
from itertools import product

import mlx.core as mx
import mlx_tests


class TestQuantized(mlx_tests.MLXTestCase):
    def test_quantize_dequantize(self):
        w = mx.random.normal(shape=(128, 512))
        for gs in [32, 64, 128]:
            for b in [1, 2, 3, 5, 6, 4, 8]:
                with self.subTest(gs=gs, b=b):
                    w_q, scales, biases = mx.quantize(w, group_size=gs, bits=b)
                    w_hat = mx.dequantize(w_q, scales, biases, gs, b)
                    errors = (w - w_hat).abs().reshape(*scales.shape, -1)
                    eps = 1e-6
                    self.assertTrue((errors <= (scales[..., None] + eps).abs()).all())

        # test quantize/dequantize 0s
        a = mx.zeros((256, 512))
        for gs in [32, 64, 128]:
            for b in [1, 2, 3, 4, 5, 6, 8]:
                w_q, scales, biases = mx.quantize(a, gs, b)
                a_hat = mx.dequantize(w_q, scales, biases, gs, b)
                self.assertTrue(mx.all(a_hat == 0))

    def test_mxfp4_quantize_dequantize(self):
        lut = mx.array(
            [
                +0.0,
                +0.5,
                +1.0,
                +1.5,
                +2.0,
                +3.0,
                +4.0,
                +6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
        )
        w = lut[mx.random.randint(0, 16, shape=(128, 512))]
        w = w.reshape(-1, 32)
        w[:, 0] = 6
        w = (w + 3e-6).astype(mx.bfloat16)

        # Invalid bits / group size
        with self.assertRaises(ValueError):
            mx.quantize(w, bits=3, mode="mxfp4")

        with self.assertRaises(ValueError):
            mx.quantize(w, group_size=64, mode="mxfp4")

        w_q, scales = mx.quantize(w, mode="mxfp4")
        with self.assertRaises(ValueError):
            mx.dequantize(w_q, scales, bits=3, mode="mxfp4")

        with self.assertRaises(ValueError):
            mx.dequantize(w_q, scales, group_size=64, mode="mxfp4")

        # Invalid output type
        with self.assertRaises(ValueError):
            mx.dequantize(
                w_q, scales, group_size=32, bits=4, mode="mxfp4", dtype=mx.int32
            )

        w_hat = mx.dequantize(w_q, scales, mode="mxfp4")
        self.assertTrue(mx.allclose(w, w_hat, rtol=1e-5, atol=1e-5))

        # test quantize/dequantize 0s
        a = mx.zeros((256, 512))
        w_q, scales = mx.quantize(a, mode="mxfp4")
        w_hat = mx.dequantize(w_q, scales, mode="mxfp4")
        self.assertTrue(mx.all(w_hat == 0))

    def test_mxfp8_quantize_dequantize(self):
        w = 2 * mx.random.uniform(shape=(512, 32)) - 1
        w = w.astype(mx.bfloat16)

        # Invalid bits / group size
        with self.assertRaises(ValueError):
            mx.quantize(w, bits=3, mode="mxfp8")

        with self.assertRaises(ValueError):
            mx.quantize(w, group_size=32, bits=7, mode="mxfp8")
        w_q, scales = mx.quantize(w, group_size=32, mode="mxfp8")

        with self.assertRaises(ValueError):
            mx.dequantize(w_q, scales, group_size=16, mode="mxfp8")

        with self.assertRaises(ValueError):
            mx.dequantize(w_q, scales, bits=4, mode="mxfp8")

        w_hat = mx.dequantize(w_q, scales, mode="mxfp8")

        self.assertTrue(mx.allclose(w, w_hat, rtol=1e-1, atol=1e-1))

        # test quantize/dequantize 0s
        a = mx.zeros((256, 512))
        w_q, scales = mx.quantize(a, mode="mxfp8")
        w_hat = mx.dequantize(w_q, scales, mode="mxfp8")
        self.assertTrue(mx.all(w_hat == 0))

    def test_nvfp4_quantize_dequantize(self):
        lut = mx.array(
            [
                +0.0,
                +0.5,
                +1.0,
                +1.5,
                +2.0,
                +3.0,
                +4.0,
                +6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
        )
        w = lut[mx.random.randint(0, 16, shape=(128, 512))]
        w = w.reshape(-1, 16)
        w[:, 0] = 6
        w = (w + 3e-6).astype(mx.bfloat16)

        # Invalid bits / group size
        with self.assertRaises(ValueError):
            mx.quantize(w, bits=3, mode="nvfp4")

        with self.assertRaises(ValueError):
            mx.quantize(w, group_size=64, mode="nvfp4")

        w_q, scales = mx.quantize(w, mode="nvfp4")

        with self.assertRaises(ValueError):
            mx.dequantize(w_q, scales, bits=3, mode="nvfp4")

        with self.assertRaises(ValueError):
            mx.dequantize(w_q, scales, group_size=32, mode="nvfp4")

        w_hat = mx.dequantize(w_q, scales, mode="nvfp4")
        self.assertTrue(mx.allclose(w, w_hat, rtol=1e-5, atol=1e-5))

        # test quantize/dequantize 0s
        a = mx.zeros((256, 512))
        w_q, scales = mx.quantize(a, mode="nvfp4")
        w_hat = mx.dequantize(w_q, scales, mode="nvfp4")
        self.assertTrue(mx.all(w_hat == 0))

        # Test nvfp4 quantize/dequantize with tensor-scale global_scale
        # currently supported only on cpu and cuda
        if not mx.metal.is_available():
            global_scale = w.abs().max().astype(mx.float32)
        else:
            global_scale = None

        w_q, scales = mx.quantize(w, mode="nvfp4", global_scale=global_scale)
        w_hat = mx.dequantize(
            w_q, scales, group_size=16, bits=4, mode="nvfp4", global_scale=global_scale
        )
        self.assertTrue(mx.allclose(w, w_hat, rtol=1e-5, atol=1e-5))

    def test_1bit_quantize_dequantize(self):
        """Test 1-bit affine quantization."""

        # Symmetric binary weights {-0.5, +0.5} should round-trip perfectly
        # (affine formula gives scale=1.0, bias=-0.5)
        for gs in [32, 64, 128]:
            with self.subTest(gs=gs, case="pack_symmetric_weights"):
                signs = (mx.random.uniform(shape=(128, 512)) > 0.5).astype(mx.float32)
                w = signs * 1.0 - (1 - signs) * 1.0  # {-1.0, +1.0}
                w = w * 0.5  # {-0.5, +0.5}

                w_q, scales, biases = mx.quantize(w, group_size=gs, bits=1)
                w_hat = mx.dequantize(w_q, scales, biases, gs, 1)

                self.assertLess((w - w_hat).abs().max(), 1e-5)

        # Asymmetric binary weights {0.1, 0.9} should round-trip perfectly
        # (affine formula gives scale=0.8, bias=0.1)
        for gs in [32, 64, 128]:
            with self.subTest(gs=gs, case="pack_asymmetric_weights"):
                bits = (mx.random.uniform(shape=(128, 512)) > 0.5).astype(mx.float32)
                w = bits * 0.9 + (1 - bits) * 0.1  # {0.1, 0.9}

                w_q, scales, biases = mx.quantize(w, group_size=gs, bits=1)
                w_hat = mx.dequantize(w_q, scales, biases, gs, 1)

                self.assertLess((w - w_hat).abs().max(), 1e-5)

        # Verify dequantized values are exactly {bias, bias + scale}
        w = mx.random.normal(shape=(64, 256))
        for gs in [32, 64, 128]:
            with self.subTest(gs=gs, case="dequant_values"):
                w_q, scales, biases = mx.quantize(w, group_size=gs, bits=1)
                w_hat = mx.dequantize(w_q, scales, biases, gs, 1)

                for i in range(scales.shape[0]):
                    for j in range(scales.shape[1]):
                        s = scales[i, j].item()
                        b = biases[i, j].item()
                        row_start = j * gs
                        row_end = row_start + gs
                        vals = w_hat[i, row_start:row_end]
                        mx.eval(vals)
                        for v in vals.tolist():
                            self.assertTrue(
                                abs(v - b) < 1e-5 or abs(v - (b + s)) < 1e-5,
                                f"Value {v} not in {{bias={b}, bias+scale={b+s}}}",
                            )

        # 1-bit quantize/dequantize zeros — scale floors to eps, bias=0
        a = mx.zeros((256, 512))
        for gs in [32, 64, 128]:
            w_q, scales, biases = mx.quantize(a, gs, 1)
            a_hat = mx.dequantize(w_q, scales, biases, gs, 1)
            self.assertLess(a_hat.abs().max(), 1e-5)

        # Quantized matmul with symmetric binary weights
        key = mx.random.key(42)
        k1, k2 = mx.random.split(key)
        for gs in [32, 64, 128]:
            with self.subTest(gs=gs, case="quantized_matmul_symmetric"):
                x = mx.random.normal(shape=(4, 256), key=k1)
                signs = (mx.random.uniform(shape=(128, 256), key=k2) > 0.5).astype(
                    mx.float32
                )
                w = signs * 0.3 - (1 - signs) * 0.3  # {-0.3, +0.3}

                w_q, scales, biases = mx.quantize(w, gs, 1)
                w_hat = mx.dequantize(w_q, scales, biases, gs, 1)
                y_q = mx.quantized_matmul(x, w_q, scales, biases, True, gs, 1)
                y_hat = x @ w_hat.T
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-5)

        # Quantized matmul with asymmetric binary weights
        for gs in [32, 64, 128]:
            with self.subTest(gs=gs, case="quantized_matmul_asymmetric"):
                x = mx.random.normal(shape=(4, 256), key=k1)
                bits = (mx.random.uniform(shape=(128, 256), key=k2) > 0.5).astype(
                    mx.float32
                )
                w = bits * 0.7 + (1 - bits) * 0.1  # {0.1, 0.7}

                w_q, scales, biases = mx.quantize(w, gs, 1)
                w_hat = mx.dequantize(w_q, scales, biases, gs, 1)
                y_q = mx.quantized_matmul(x, w_q, scales, biases, True, gs, 1)
                y_hat = x @ w_hat.T
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-5)

    def test_qqmv(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [256, 512, 67],  # M
            [64, 256],  # N
        )
        modes = ["nvfp4", "mxfp8"]
        for M, N in tests:
            for mode in modes:
                with self.subTest(shape=(M, N), mode=mode):
                    x_shape = (1, N)
                    w_shape = (M, N)

                    x = mx.random.normal(shape=x_shape, key=k1)
                    x_hat = mx.dequantize(
                        *mx.quantize(x, mode=mode), mode=mode, dtype=mx.float32
                    )

                    w = mx.random.normal(shape=w_shape, key=k2)
                    w_q, scales = mx.quantize(w, mode=mode)
                    w_hat = mx.dequantize(w_q, scales, mode=mode, dtype=mx.float32)
                    y_q = mx.qqmm(
                        x,
                        w_q,
                        scales,
                        mode=mode,
                    )
                    y_hat = x_hat @ mx.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qmm(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        dtype = mx.float16 if (mx.default_device() == mx.gpu) else mx.float32
        tests = product(
            [128, 64, 32],  # group_size
            [1, 2, 4, 8],  # bits
            [8, 32, 33, 64],  # M
            [128, 256],  # N
            [128, 256],  # K
            [True, False],  # transposed
        )
        for group_size, bits, M, N, K, transposed in tests:
            with self.subTest(
                shape=(M, N, K),
                group_size=group_size,
                bits=bits,
                transposed=transposed,
            ):
                x = mx.random.normal(shape=(M, K), key=k1) / K**0.5
                w = (
                    mx.random.normal(shape=(N, K) if transposed else (K, N), key=k2)
                    / K**0.5
                )
                x = x.astype(dtype)
                w = w.astype(dtype)
                w_q, scales, biases = mx.quantize(w, group_size, bits)
                w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                y_q = mx.quantized_matmul(
                    x, w_q, scales, biases, transposed, group_size, bits
                )
                y_hat = (x @ w_hat.T) if transposed else (x @ w_hat)
                self.assertEqual(y_q.shape, y_hat.shape)

                tol = 1e-3 if dtype == mx.float32 else 1.5e-3
                self.assertLess((y_q - y_hat).abs().max(), tol)

    def test_qmm_vjp(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)

        bits = 8
        group_size = 64
        M = 64
        N = 1024
        K = 512

        x = mx.random.normal(shape=(2, M, K), key=k1)
        c = mx.ones(shape=(2, M, N))

        transposes = [True, False]
        for transposed in transposes:
            w = mx.random.normal(shape=(N, K) if transposed else (K, N), key=k2)
            w_q, scales, biases = mx.quantize(w, group_size, bits)

            def fn(x):
                return mx.quantized_matmul(
                    x, w_q, scales, biases, transposed, group_size, bits
                )

            _, vjp_out = mx.vjp(fn, primals=(x,), cotangents=(c,))

            expected_out = mx.quantized_matmul(
                c, w_q, scales, biases, not transposed, group_size, bits
            )
            self.assertTrue(mx.allclose(vjp_out[0], expected_out))

    def test_qmm_jvp(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)

        bits = 8
        group_size = 64
        M = 64
        N = 128
        K = 128

        x = mx.random.normal(shape=(2, M, K), key=k1)
        x_tan = mx.ones(shape=(2, M, N))

        transposes = [True, False]
        for transposed in transposes:
            w = mx.random.normal(shape=(N, K) if transposed else (K, N), key=k2)
            w_q, scales, biases = mx.quantize(w, group_size, bits)

            def fn(x):
                return mx.quantized_matmul(
                    x, w_q, scales, biases, transposed, group_size, bits
                )

            _, jvp_out = mx.jvp(fn, primals=(x,), tangents=(x_tan,))

            expected_out = mx.quantized_matmul(
                x_tan, w_q, scales, biases, transposed, group_size, bits
            )
            self.assertTrue(mx.allclose(jvp_out[0], expected_out))

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
            [2, 3, 4, 5, 6, 8],  # bits
            [256, 512, 67],  # M
            [64, 256],  # N
            [0, 1, 3, 8],  # B
        )
        for group_size, bits, M, N, B in tests:
            if group_size > N:
                continue
            with self.subTest(shape=(B, M, N), group_size=group_size, bits=bits):
                x_shape = (3, 1, N) if B == 0 else (B, 1, N)
                w_shape = (M, N) if B == 0 else (B, M, N)
                x = mx.random.normal(shape=x_shape, key=k1)
                w = mx.random.normal(shape=w_shape, key=k2)
                w_q, scales, biases = mx.quantize(w, group_size, bits)
                w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                y_q = mx.quantized_matmul(
                    x, w_q, scales, biases, True, group_size, bits
                )
                y_hat = x @ mx.swapaxes(w_hat, -1, -2)
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_fp_qmv(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [256, 512, 67],  # M
            [64, 256],  # N
            [0, 1, 3, 8],  # B
        )
        modes = ["mxfp4", "nvfp4", "mxfp8"]
        for M, N, B in tests:
            for mode in modes:
                with self.subTest(shape=(B, M, N), mode=mode):
                    x_shape = (3, 1, N) if B == 0 else (B, 1, N)
                    w_shape = (M, N) if B == 0 else (B, M, N)
                    x = mx.random.normal(shape=x_shape, key=k1)
                    w = mx.random.normal(shape=w_shape, key=k2)
                    w_q, scales = mx.quantize(w, mode=mode)
                    w_hat = mx.dequantize(w_q, scales, mode=mode)
                    y_q = mx.quantized_matmul(
                        x,
                        w_q,
                        scales,
                        transpose=True,
                        mode=mode,
                    )
                    y_hat = x @ mx.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), 1e-3)

        # Test multiple of 16 but not 32
        M = 128
        N = 48
        mode = "nvfp4"
        with self.subTest(shape=(B, M, N), mode=mode):
            x_shape = (1, N)
            w_shape = (M, N)
            x = mx.random.normal(shape=x_shape, key=k1)
            w = mx.random.normal(shape=w_shape, key=k2)
            w_q, scales = mx.quantize(w, mode=mode)
            w_hat = mx.dequantize(w_q, scales, mode=mode)
            y_q = mx.quantized_matmul(
                x,
                w_q,
                scales,
                transpose=True,
                mode=mode,
            )
            y_hat = x @ mx.swapaxes(w_hat, -1, -2)
            self.assertEqual(y_q.shape, y_hat.shape)
            self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qvm(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [128, 64, 32],  # group_size
            [2, 3, 4, 5, 6, 8],  # bits
            [32, 128, 256],  # M
            [128, 256, 67],  # N
            [0, 1, 3, 8],  # B
        )
        for group_size, bits, M, N, B in tests:
            with self.subTest(shape=(B, M, N), group_size=group_size, bits=bits):
                if M < group_size:
                    continue
                x_shape = (1, N) if B == 0 else (B, 1, N)
                w_shape = (N, M) if B == 0 else (B, N, M)
                x = mx.random.normal(shape=x_shape, key=k1)
                w = mx.random.normal(shape=w_shape, key=k2)
                w_q, scales, biases = mx.quantize(w, group_size, bits)
                w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                y_q = mx.quantized_matmul(
                    x, w_q, scales, biases, False, group_size, bits
                )
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qvm_splitk(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [128, 64, 32],  # group_size
            [2, 4, 8],  # bits
            [128],  # M
            [16384],  # N
            [1, 3],  # B
        )
        for group_size, bits, M, N, B in tests:
            with self.subTest(shape=(B, M, N), group_size=group_size, bits=bits):
                x_shape = (1, N) if B == 0 else (B, 1, N)
                w_shape = (N, M) if B == 0 else (B, N, M)
                x = 1e-1 * mx.random.normal(shape=x_shape, key=k1)
                w = 1e-1 * mx.random.normal(shape=w_shape, key=k2)
                w_q, scales, biases = mx.quantize(w, group_size, bits)
                w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                y_q = mx.quantized_matmul(
                    x, w_q, scales, biases, False, group_size, bits
                )
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 2e-3)

        # Test with 1D vector
        group_size = 32
        bits = 8
        N = 2048
        x = 1e-1 * mx.random.normal(shape=(N,), key=k1)
        w = 1e-1 * mx.random.normal(shape=(N, N), key=k2)
        w_q, scales, biases = mx.quantize(w, group_size, bits)
        w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
        y_q = mx.quantized_matmul(x, w_q, scales, biases, False, group_size, bits)
        y_hat = x @ w_hat
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 2e-3)

    def test_fp_qvm(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [32, 128, 256],  # M
            [128, 256, 67],  # N
            [0, 1, 3, 8],  # B
        )
        # Add a splitk
        tests = list(tests)
        tests.append((128, 16384, 0))
        modes = ["mxfp4", "nvfp4", "mxfp8"]

        for M, N, B in tests:
            for mode in modes:
                with self.subTest(shape=(B, M, N), mode=mode):
                    x_shape = (1, N) if B == 0 else (B, 1, N)
                    w_shape = (N, M) if B == 0 else (B, N, M)
                    x = mx.random.normal(shape=x_shape, key=k1)
                    w = mx.random.normal(shape=w_shape, key=k2)
                    w_q, scales = mx.quantize(w, mode=mode)
                    w_hat = mx.dequantize(w_q, scales, mode=mode)
                    y_q = mx.quantized_matmul(
                        x,
                        w_q,
                        scales,
                        transpose=False,
                        mode=mode,
                    )
                    y_hat = x @ w_hat
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), 2e-3)

    def test_mode_error_cases(self):
        w = mx.random.normal(shape=(256, 256))
        x = mx.random.normal(shape=(1, 256))

        # Invalid mode
        with self.assertRaises(ValueError):
            mx.quantize(w, mode="xyz")

        wq, scales, biases = mx.quantize(w, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            mx.dequantize(wq, scales, biases, bits=4, group_size=32, mode="xyz")

        with self.assertRaises(ValueError):
            mx.quantized_matmul(
                x, wq, scales, biases, bits=4, group_size=32, mode="xyz"
            )

        rhs_indices = mx.array(0)
        with self.assertRaises(ValueError):
            mx.gather_qmm(
                x,
                wq,
                scales,
                biases,
                rhs_indices=rhs_indices,
                bits=4,
                group_size=32,
                mode="xyz",
            )

        # Only quantize floating point types
        with self.assertRaises(ValueError):
            mx.quantize(mx.zeros((128, 128), mx.int32))

        with self.assertRaises(ValueError):
            mx.quantize(mx.zeros((128, 128), mx.int32), mode="mxfp4")

        # Must have bias for affine
        with self.assertRaises(ValueError):
            mx.dequantize(wq, scales, None, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, wq, scales, None, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            mx.gather_qmm(
                x, wq, scales, None, rhs_indices=rhs_indices, bits=4, group_size=32
            )

        # Must be floating point
        x = mx.zeros(shape=(256,), dtype=mx.int32)
        scales = mx.zeros(scales.shape, dtype=mx.int32)
        biases = mx.zeros(scales.shape, dtype=mx.int32)
        with self.assertRaises(ValueError):
            mx.dequantize(wq, scales, biases, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            mx.quantized_matmul(x, wq, scales, biases, bits=4, group_size=32)

        with self.assertRaises(ValueError):
            mx.gather_qmm(
                x, wq, scales, biases, rhs_indices=rhs_indices, bits=4, group_size=32
            )

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
        for w_shape in [(8, 256), (1, 8, 256), (3, 8, 256)]:
            with self.subTest(w_shape=w_shape):
                w = mx.random.normal(shape=(w_shape))
                w_q, scales, biases = mx.quantize(w)
                w_hat = mx.dequantize(w_q, scales, biases)

                # Test qmv
                for shape in [(3, 1, 256), (3, 4, 256)]:
                    x = mx.random.normal(shape=shape)
                    y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
                    y_hat = x @ mx.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), 1e-3)

                # Test qmm_t
                x = mx.random.normal(shape=(3, 10, 256))
                y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
                y_hat = x @ mx.swapaxes(w_hat, -1, -2)
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

                # Test qvm
                x = mx.random.normal(shape=(3, 1, 8))
                y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=False)
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 1e-3)

                # Test qmm
                x = mx.random.normal(shape=(3, 10, 8))
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

        # Test with larger than 128 unaligned sizes
        w = mx.random.normal(shape=(99, 256))
        w_q, scales, biases = mx.quantize(w)
        w_hat = mx.dequantize(w_q, scales, biases)
        x = mx.random.normal(shape=(129, 256))
        y_q = mx.quantized_matmul(x, w_q, scales, biases, transpose=True)
        y_hat = x @ w_hat.T
        self.assertEqual(y_q.shape, y_hat.shape)
        self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_qmv_small_non_multiples(self):
        # Test very small K and N dimensions (e.g., [MxK] x [NxK].T = [MxN])
        # Each tuple is (M, K, N) representing input rows, weight cols, weight rows
        test_cases = [
            (1, 32, 3),
            (2, 32, 10),
            (1, 32, 5),
            (4, 32, 7),
        ]

        # Test different quantization settings (bits, group_size, mode)
        quantization_settings = [
            (4, 32, "affine"),
            (6, 32, "affine"),
            (4, 16, "nvfp4"),
        ]

        for M, K, N in test_cases:
            for bits, group_size, mode in quantization_settings:
                # Test without batch dimension
                with self.subTest(
                    M=M,
                    K=K,
                    N=N,
                    batch=None,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                ):
                    w = mx.random.normal(shape=(N, K))
                    w_q, *sb = mx.quantize(
                        w,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                    )
                    w_hat = mx.dequantize(
                        w_q,
                        *sb,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                    )

                    # Test qmv/qmm_t (transpose=True): [MxK] @ [NxK].T = [MxN]
                    x = mx.random.normal(shape=(M, K))
                    y_q = mx.quantized_matmul(
                        x,
                        w_q,
                        *sb,
                        transpose=True,
                        group_size=group_size,
                        bits=bits,
                        mode=mode,
                    )
                    y_hat = x @ mx.swapaxes(w_hat, -1, -2)
                    self.assertEqual(y_q.shape, y_hat.shape)
                    self.assertLess((y_q - y_hat).abs().max(), 1e-3)

    def test_gather_qmm(self):
        def quantize(w, transpose=True, group_size=None, bits=None, mode="affine"):
            if mode == "affine":
                qw, s, b = mx.quantize(w, group_size=group_size, bits=bits, mode=mode)
            else:
                qw, s = mx.quantize(w, group_size=group_size, bits=bits, mode=mode)
                b = None
            w_hat = mx.dequantize(qw, s, b, group_size=group_size, bits=bits, mode=mode)
            if transpose:
                w_hat = w_hat.swapaxes(-1, -2)
            return w_hat, qw, s, b

        def test_shape(
            M,
            N,
            K,
            dtype=mx.float32,
            batch_A=(),
            batch_B=(),
            lhs_indices=None,
            rhs_indices=None,
            transpose=True,
            group_size=None,
            bits=None,
            mode="affine",
        ):
            with self.subTest(
                M=M,
                N=N,
                K=K,
                dtype=dtype,
                batch_A=batch_A,
                batch_B=batch_B,
                lhs_indices=lhs_indices,
                rhs_indices=rhs_indices,
                transpose=transpose,
                group_size=group_size,
                bits=bits,
                mode=mode,
            ):
                x = mx.random.normal(shape=batch_A + (M, K)).astype(dtype)
                w = mx.random.normal(
                    shape=batch_B + ((N, K) if transpose else (K, N))
                ).astype(dtype)
                w_hat, qw, s, b = quantize(w, transpose, group_size, bits, mode=mode)

                if lhs_indices is not None:
                    lhs_indices = mx.array(lhs_indices)
                if rhs_indices is not None:
                    rhs_indices = mx.array(rhs_indices)

                c1 = mx.gather_mm(x, w_hat, lhs_indices, rhs_indices)
                c2 = mx.gather_qmm(
                    x,
                    qw,
                    s,
                    b,
                    lhs_indices,
                    rhs_indices,
                    transpose=transpose,
                    group_size=group_size,
                    bits=bits,
                    mode=mode,
                )
                self.assertTrue(mx.allclose(c1, c2, atol=1e-4))

        inputs = (
            {
                "batch_A": (1,),
                "lhs_indices": (0,),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (1,),
                "lhs_indices": None,
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (2,),
                "lhs_indices": None,
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (3,),
                "lhs_indices": (0, 2),
                "batch_B": (1,),
                "rhs_indices": (0,),
            },
            {
                "batch_A": (5,),
                "lhs_indices": (0, 2),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
            },
            {
                "batch_A": (4, 2),
                "lhs_indices": (
                    (7, 6),
                    (5, 4),
                    (1, 2),
                ),
                "batch_B": (4, 1),
                "rhs_indices": ((2,), (0,), (1,)),
            },
            {
                "batch_A": (1,),
                "lhs_indices": (0,),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
                "mode": "nvfp4",
            },
            {
                "batch_A": (1,),
                "lhs_indices": (0,),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
                "mode": "mxfp4",
            },
            {
                "batch_A": (1,),
                "lhs_indices": (0,),
                "batch_B": (3,),
                "rhs_indices": (2, 1),
                "mode": "mxfp8",
            },
        )

        for kwargs in inputs:
            test_shape(1, 32, 128, **kwargs)
            test_shape(32, 32, 256, **kwargs)
            test_shape(1, 32, 256, **kwargs)
            test_shape(32, 256, 32, transpose=False, **kwargs)
            test_shape(1, 256, 32, transpose=False, **kwargs)
            test_shape(32, 32, 512, **kwargs)
            test_shape(1, 32, 512, **kwargs)
            test_shape(32, 512, 32, transpose=False, **kwargs)
            test_shape(1, 512, 32, transpose=False, **kwargs)

    def test_qmm_fp_type(self):
        indices = mx.array([[2], [0], [1]], dtype=mx.uint32)

        modes = ["mxfp8", "mxfp4"]
        for mode in modes:
            for t in [mx.bfloat16, mx.float16, mx.float32]:
                x = mx.random.normal((32, 256)).astype(t)

                w = mx.random.normal((32, 256))
                wq, s = mx.quantize(w, mode=mode)
                out = mx.quantized_matmul(x, wq, s, mode=mode)
                self.assertEqual(out.dtype, t)

                w = mx.random.normal((4, 32, 256))
                wq, s = mx.quantize(w, mode=mode)

                out = mx.gather_qmm(x, wq, s, rhs_indices=indices, mode=mode)
                self.assertEqual(out.dtype, t)

    def test_gather_matmul_grad(self):
        def quantize(w, transpose=True, group_size=64, bits=4):
            qw, s, b = mx.quantize(w, group_size=group_size, bits=bits)
            w_hat = mx.dequantize(qw, s, b, group_size=group_size, bits=bits)
            if transpose:
                w_hat = w_hat.swapaxes(-1, -2)
            return w_hat, qw, s, b

        lhs_indices = mx.array([[7, 6], [4, 1], [0, 2]], dtype=mx.uint32)
        rhs_indices = mx.array([[2], [0], [1]], dtype=mx.uint32)

        x = mx.random.normal((4, 2, 32, 256))
        w = mx.random.normal((4, 1, 32, 256))
        w_hat, qw, s, b = quantize(w)

        def f_ref(x, w, i1, i2):
            return mx.gather_mm(x, w, i1, i2).sum()

        def f_test(x, qw, s, b, i1, i2):
            return mx.gather_qmm(x, qw, s, b, i1, i2, transpose=True).sum()

        r1 = f_ref(x, w_hat, lhs_indices, rhs_indices)
        r2 = f_test(x, qw, s, b, lhs_indices, rhs_indices)
        self.assertTrue(mx.allclose(r1, r2, atol=1e-4))

        g1 = mx.grad(f_ref)(x, w_hat, lhs_indices, rhs_indices)
        g2 = mx.grad(f_test)(x, qw, s, b, lhs_indices, rhs_indices)
        self.assertTrue(mx.allclose(g1, g2, atol=1e-4))

    def test_gather_qmm_sorted(self):
        def quantize(w, transpose=True, group_size=None, mode="affine"):
            if mode == "affine":
                qw, s, b = mx.quantize(w, group_size=group_size, mode=mode)
            else:
                qw, s = mx.quantize(w, mode=mode)
                b = None

            w_hat = mx.dequantize(qw, s, b, group_size=group_size, mode=mode)
            if transpose:
                w_hat = w_hat.swapaxes(-1, -2)
            return w_hat, qw, s, b

        def gather_sort(x, indices):
            N, M = indices.shape
            indices = indices.flatten()
            order = mx.argsort(indices)
            inv_order = mx.argsort(order)
            return x.flatten(0, -3)[order // M], indices[order], inv_order

        def scatter_unsort(x, inv_order, shape=None):
            x = x[inv_order]
            if shape is not None:
                x = mx.unflatten(x, 0, shape)
            return x

        parameters = [
            # L, K, D, E, I, transpose
            (32, 512, 512, 4, 2, True, "affine"),
            (32, 512, 544, 4, 2, True, "mxfp4"),
            (32, 512, 544, 4, 2, True, "nvfp4"),
            (32, 512, 544, 4, 2, True, "mxfp8"),
            (133, 512, 512, 4, 2, True, "affine"),
            (133, 512, 555, 4, 2, True, "affine"),
            (133, 512, 512, 4, 2, True, "affine"),
            (64, 512, 512, 4, 2, False, "affine"),
            (64, 512, 544, 4, 2, False, "mxfp4"),
            (64, 512, 544, 4, 2, False, "nvfp4"),
            (64, 512, 544, 4, 2, False, "mxfp8"),
            (133, 512, 512, 4, 2, False, "affine"),
            (133, 512, 544, 4, 2, False, "affine"),
            (133, 512, 555, 4, 2, False, "affine"),
            (64, 512, 512, 4, 2, False, "affine"),
        ]

        key = mx.random.key(0)
        k1, k2, k3 = mx.random.split(key, 3)
        dtype = mx.float16 if (mx.default_device() == mx.gpu) else mx.float32

        for L, K, D, E, I, transpose, mode in parameters:
            with self.subTest(L=L, K=K, D=D, E=E, I=I, transpose=transpose, mode=mode):
                if mode != "affine":
                    group_size = None
                    dtype = (
                        mx.bfloat16 if (mx.default_device() == mx.gpu) else mx.float32
                    )
                else:
                    group_size = 64
                    dtype = (
                        mx.float16 if (mx.default_device() == mx.gpu) else mx.float32
                    )

                K, D = (K, D) if transpose else (D, K)
                ishape = (L, I)
                xshape = (L, 1, 1, K)
                wshape = (E, D, K) if transpose else (E, K, D)

                indices = (mx.random.uniform(shape=ishape, key=k1) * E).astype(
                    mx.uint32
                )
                x = mx.random.normal(xshape, key=k2) / K**0.5
                w = mx.random.normal(wshape, key=k3) / K**0.5

                x = x.astype(dtype)
                w = w.astype(dtype)

                w, *wq = quantize(
                    w, group_size=group_size, mode=mode, transpose=transpose
                )

                y1 = mx.gather_mm(x, w, rhs_indices=indices)
                y2 = mx.gather_qmm(
                    x,
                    *wq,
                    group_size=group_size,
                    mode=mode,
                    transpose=transpose,
                    rhs_indices=indices,
                )
                xs, idx, inv_order = gather_sort(x, indices)
                y3 = mx.gather_mm(xs, w, rhs_indices=idx, sorted_indices=True)

                y4 = mx.gather_qmm(
                    xs,
                    *wq,
                    group_size=group_size,
                    mode=mode,
                    rhs_indices=idx,
                    transpose=transpose,
                    sorted_indices=True,
                )
                y3 = scatter_unsort(y3, inv_order, indices.shape)
                y4 = scatter_unsort(y4, inv_order, indices.shape)

                tol = 1.5e-5 if (dtype == mx.float32) else 2.5e-4

                self.assertLess((y1 - y2).abs().max(), tol)
                self.assertLess((y1 - y3).abs().max(), tol)
                self.assertLess((y1 - y4).abs().max(), tol)

                self.assertTrue(mx.allclose(y1, y2, atol=tol))
                self.assertTrue(mx.allclose(y1, y3, atol=tol))
                self.assertTrue(mx.allclose(y1, y4, atol=tol))

    def test_gather_qmm_grad(self):
        def gather_qmm_ref(x, w, s, b, lhs, rhs, trans, sort):
            if lhs is not None:
                x = x[lhs]
            if rhs is not None:
                w = w[rhs]
                s = s[rhs]
                b = b[rhs]
            return mx.quantized_matmul(x, w, s, b, transpose=trans)

        def gather_qmm(x, w, s, b, lhs, rhs, trans, sort):
            return mx.gather_qmm(
                x,
                w,
                s,
                b,
                transpose=trans,
                lhs_indices=lhs,
                rhs_indices=rhs,
                sorted_indices=sort,
            )

        key = mx.random.key(0)
        k1, k2, k3, k4 = mx.random.split(key, 4)
        dtype = mx.float32

        x = mx.random.normal((16, 1, 256), key=k1).astype(dtype)
        w, s, b = mx.quantize(mx.random.normal((4, 256, 256), key=k2).astype(dtype))
        indices = mx.sort(mx.random.randint(0, 4, shape=(16,), key=k3))
        cotan = mx.random.normal((16, 1, 256), key=k4).astype(dtype)

        (o1,), (dx1, ds1, db1) = mx.vjp(
            lambda x, s, b: gather_qmm_ref(x, w, s, b, None, indices, True, True),
            [x, s, b],
            [cotan],
        )
        (o2,), (dx2, ds2, db2) = mx.vjp(
            lambda x, s, b: gather_qmm(x, w, s, b, None, indices, True, True),
            [x, s, b],
            [cotan],
        )

        self.assertLess((o1 - o2).abs().max(), 1e-4)
        self.assertTrue(mx.allclose(o1, o2, atol=1e-4))
        self.assertTrue(mx.allclose(dx1, dx2, atol=1e-4))
        self.assertTrue(mx.allclose(ds1, ds2, atol=1e-3))
        self.assertTrue(mx.allclose(db1, db2, atol=1e-3))

    def test_vjp_scales_biases(self):
        mx.random.seed(0)
        x = mx.random.normal(shape=(2, 2, 512))
        w = mx.random.normal(shape=(512, 512))
        wq, s, b = mx.quantize(w, bits=4, group_size=64)

        def mm(sb, x, wq):
            return mx.quantized_matmul(x, wq, *sb, bits=4, group_size=64).sum()

        params = (s, b)
        dparams = mx.grad(mm)((s, b), x, wq)

        eps = 8e-3
        # numerical grad check with a few indices
        indices = [(0, 0), (11, 4), (22, 7)]
        for idx in indices:
            for p in [0, 1]:
                params[p][idx] += eps
                out_up = mm(params, x, wq)
                params[p][idx] -= 2 * eps
                out_down = mm(params, x, wq)
                params[p][idx] += eps
                num_ds = (out_up - out_down) / (2 * eps)
                self.assertAlmostEqual(dparams[p][idx], num_ds, delta=2e-2)

    def test_fp_vjp_scales_throws(self):
        mx.random.seed(0)
        x = mx.random.normal(shape=(2, 512))
        w = mx.random.normal(shape=(512, 512))
        for mode in ["mxfp4", "mxfp8", "nvfp4"]:
            wq, s = mx.quantize(w, mode=mode)

            def mm(s, x, wq):
                return mx.quantized_matmul(x, wq, s, mode=mode).sum()

            # Should raise
            with self.assertRaises(ValueError):
                ds = mx.grad(mm)(s, x, wq)

            rhs_indices = mx.array(0)
            with self.assertRaises(ValueError):

                def gmm(s, x, wq):
                    return mx.gather_qmm(
                        x,
                        wq,
                        s,
                        rhs_indices=rhs_indices,
                        mode=mode,
                    ).sum()

                ds = mx.grad(gmm)(s, x, wq)

    def test_quantize_strided(self):
        N = 64
        mode = "nvfp4"
        w = mx.random.normal(shape=(N, N))
        w_q, scales = mx.quantize(w, mode="nvfp4")

        scales = mx.broadcast_to(mx.array(56, mx.uint8), scales.shape)
        w_hat = mx.dequantize(w_q, scales, mode=mode)
        expected = mx.dequantize(w_q, mx.contiguous(scales), mode=mode)
        self.assertTrue(mx.allclose(w_hat, expected))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
