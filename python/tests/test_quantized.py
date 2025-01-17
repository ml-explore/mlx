# Copyright © 2023 Apple Inc.

import unittest
from itertools import product

import mlx.core as mx
import mlx_tests


class TestQuantized(mlx_tests.MLXTestCase):
    def test_quantize_dequantize(self):
        w = mx.random.normal(shape=(128, 512))
        for gs in [32, 64, 128]:
            for b in [2, 3, 6, 4, 8]:
                with self.subTest(gs=gs, b=b):
                    w_q, scales, biases = mx.quantize(w, group_size=gs, bits=b)
                    w_hat = mx.dequantize(w_q, scales, biases, gs, b)
                    errors = (w - w_hat).abs().reshape(*scales.shape, -1)
                    eps = 1e-6
                    self.assertTrue((errors <= (scales[..., None] + eps).abs()).all())

        # test quantize/dequantize 0s
        a = mx.zeros((256, 512))
        for gs in [32, 64, 128]:
            for b in [2, 3, 4, 6, 8]:
                w_q, scales, biases = mx.quantize(a, gs, b)
                a_hat = mx.dequantize(w_q, scales, biases, gs, b)
                self.assertTrue(mx.all(a_hat == 0))

    def test_qmm(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [128, 64, 32],  # group_size
            [2, 4, 8],  # bits
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
            [2, 3, 4, 6, 8],  # bits
            [256, 512, 67],  # M
            [64, 128],  # N
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

    def test_qvm(self):
        key = mx.random.key(0)
        k1, k2 = mx.random.split(key)
        tests = product(
            [128, 64, 32],  # group_size
            [2, 3, 4, 6, 8],  # bits
            [128, 256],  # M
            [128, 256, 67],  # N
            [0, 1, 3, 8],  # B
        )
        for group_size, bits, M, N, B in tests:
            with self.subTest(shape=(B, M, N), group_size=group_size, bits=bits):
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
                x = mx.random.normal(shape=x_shape, key=k1)
                w = mx.random.normal(shape=w_shape, key=k2)
                w_q, scales, biases = mx.quantize(w, group_size, bits)
                w_hat = mx.dequantize(w_q, scales, biases, group_size, bits)
                y_q = mx.quantized_matmul(
                    x, w_q, scales, biases, False, group_size, bits
                )
                y_hat = x @ w_hat
                self.assertEqual(y_q.shape, y_hat.shape)
                self.assertLess((y_q - y_hat).abs().max(), 2e-3)

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

    def test_gather_qmm(self):
        def quantize(w, transpose=True, group_size=64, bits=4):
            qw, s, b = mx.quantize(w, group_size=group_size, bits=bits)
            w_hat = mx.dequantize(qw, s, b, group_size=group_size, bits=bits)
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
            group_size=64,
            bits=4,
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
            ):
                x = mx.random.normal(shape=batch_A + (M, K)).astype(dtype)
                w = mx.random.normal(
                    shape=batch_B + ((N, K) if transpose else (K, N))
                ).astype(dtype)
                w_hat, qw, s, b = quantize(w, transpose, group_size, bits)

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
        )

        for kwargs in inputs:
            test_shape(32, 32, 256, **kwargs)
            test_shape(1, 32, 256, **kwargs)
            test_shape(32, 256, 32, transpose=False, **kwargs)
            test_shape(1, 256, 32, transpose=False, **kwargs)
            test_shape(32, 32, 512, **kwargs)
            test_shape(1, 32, 512, **kwargs)
            test_shape(32, 512, 32, transpose=False, **kwargs)
            test_shape(1, 512, 32, transpose=False, **kwargs)

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


if __name__ == "__main__":
    unittest.main()
