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

    def test_type_promotion(self):
        import mlx.core as mx

        a = mx.array([4, 8], mx.float64)
        b = mx.array([4, 8], mx.int32)

        with mx.stream(mx.cpu):
            c = a + b
            self.assertEqual(c.dtype, mx.float64)

    def test_lapack(self):
        with mx.stream(mx.cpu):
            # QRF
            A = mx.array([[2.0, 3.0], [1.0, 2.0]], dtype=mx.float64)
            Q, R = mx.linalg.qr(A)
            out = Q @ R
            self.assertTrue(mx.allclose(out, A))
            out = Q.T @ Q
            self.assertTrue(mx.allclose(out, mx.eye(2)))
            self.assertTrue(mx.allclose(mx.tril(R, -1), mx.zeros_like(R)))
            self.assertEqual(Q.dtype, mx.float64)
            self.assertEqual(R.dtype, mx.float64)

            # SVD
            A = mx.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=mx.float64
            )
            U, S, Vt = mx.linalg.svd(A)
            self.assertTrue(mx.allclose(U[:, : len(S)] @ mx.diag(S) @ Vt, A))

            # Inverse
            A = mx.array([[1, 2, 3], [6, -5, 4], [-9, 8, 7]], dtype=mx.float64)
            A_inv = mx.linalg.inv(A)
            self.assertTrue(mx.allclose(A @ A_inv, mx.eye(A.shape[0])))

            # Tri inv
            A = mx.array([[1, 0, 0], [6, -5, 0], [-9, 8, 7]], dtype=mx.float64)
            B = mx.array([[7, 0, 0], [3, -2, 0], [1, 8, 3]], dtype=mx.float64)
            AB = mx.stack([A, B])
            invs = mx.linalg.tri_inv(AB, upper=False)
            for M, M_inv in zip(AB, invs):
                self.assertTrue(mx.allclose(M @ M_inv, mx.eye(M.shape[0])))

            # Cholesky
            sqrtA = mx.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=mx.float64
            )
            A = sqrtA.T @ sqrtA / 81
            L = mx.linalg.cholesky(A)
            U = mx.linalg.cholesky(A, upper=True)
            self.assertTrue(mx.allclose(L @ L.T, A))
            self.assertTrue(mx.allclose(U.T @ U, A))

            # Psueod inverse
            A = mx.array([[1, 2, 3], [6, -5, 4], [-9, 8, 7]], dtype=mx.float64)
            A_plus = mx.linalg.pinv(A)
            self.assertTrue(mx.allclose(A @ A_plus @ A, A))

            # Eigh
            def check_eigs_and_vecs(A_np, kwargs={}):
                A = mx.array(A_np, dtype=mx.float64)
                eig_vals, eig_vecs = mx.linalg.eigh(A, **kwargs)
                eig_vals_np, _ = np.linalg.eigh(A_np, **kwargs)
                self.assertTrue(np.allclose(eig_vals, eig_vals_np))
                self.assertTrue(
                    mx.allclose(A @ eig_vecs, eig_vals[..., None, :] * eig_vecs)
                )

                eig_vals_only = mx.linalg.eigvalsh(A, **kwargs)
                self.assertTrue(mx.allclose(eig_vals, eig_vals_only))

            # Test a simple 2x2 symmetric matrix
            A_np = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
            check_eigs_and_vecs(A_np)

            # Test a larger random symmetric matrix
            n = 5
            np.random.seed(1)
            A_np = np.random.randn(n, n).astype(np.float64)
            A_np = (A_np + A_np.T) / 2
            check_eigs_and_vecs(A_np)

            # Test with upper triangle
            check_eigs_and_vecs(A_np, {"UPLO": "U"})

            # LU factorization
            # Test 3x3 matrix
            a = mx.array(
                [[3.0, 1.0, 2.0], [1.0, 8.0, 6.0], [9.0, 2.0, 5.0]], dtype=mx.float64
            )
            P, L, U = mx.linalg.lu(a)
            self.assertTrue(mx.allclose(L[P, :] @ U, a))

            # Solve triangular
            # Test lower triangular matrix
            a = mx.array(
                [[4.0, 0.0, 0.0], [2.0, 3.0, 0.0], [1.0, -2.0, 5.0]], dtype=mx.float64
            )
            b = mx.array([8.0, 14.0, 3.0], dtype=mx.float64)

            result = mx.linalg.solve_triangular(a, b, upper=False)
            expected = np.linalg.solve(np.array(a), np.array(b))
            self.assertTrue(np.allclose(result, expected))

            # Test upper triangular matrix
            a = mx.array(
                [[3.0, 2.0, 1.0], [0.0, 5.0, 4.0], [0.0, 0.0, 6.0]], dtype=mx.float64
            )
            b = mx.array([13.0, 33.0, 18.0], dtype=mx.float64)

            result = mx.linalg.solve_triangular(a, b, upper=True)
            expected = np.linalg.solve(np.array(a), np.array(b))
            self.assertTrue(np.allclose(result, expected))

    def test_conversion(self):
        a = mx.array([1.0, 2.0], mx.float64)
        b = np.array(a)
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
