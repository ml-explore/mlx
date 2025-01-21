# Copyright © 2023 Apple Inc.

import itertools
import math
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestLinalg(mlx_tests.MLXTestCase):
    def test_norm(self):
        vector_ords = [None, 0.5, 0, 1, 2, 3, -1, float("inf"), -float("inf")]
        matrix_ords = [None, "fro", -1, 1, float("inf"), -float("inf")]

        for shape in [(3,), (2, 3), (2, 3, 3)]:
            x_mx = mx.arange(1, math.prod(shape) + 1).reshape(shape)
            x_np = np.arange(1, math.prod(shape) + 1).reshape(shape)
            # Test when at least one axis is provided
            for num_axes in range(1, len(shape)):
                if num_axes == 1:
                    ords = vector_ords
                else:
                    ords = matrix_ords
                for axis in itertools.combinations(range(len(shape)), num_axes):
                    for keepdims in [True, False]:
                        for o in ords:
                            out_np = np.linalg.norm(
                                x_np, ord=o, axis=axis, keepdims=keepdims
                            )
                            out_mx = mx.linalg.norm(
                                x_mx, ord=o, axis=axis, keepdims=keepdims
                            )
                            with self.subTest(
                                shape=shape, ord=o, axis=axis, keepdims=keepdims
                            ):
                                self.assertTrue(
                                    np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6)
                                )

        # Test only ord provided
        for shape in [(3,), (2, 3)]:
            x_mx = mx.arange(1, math.prod(shape) + 1).reshape(shape)
            x_np = np.arange(1, math.prod(shape) + 1).reshape(shape)
            for o in [None, 1, -1, float("inf"), -float("inf")]:
                for keepdims in [True, False]:
                    out_np = np.linalg.norm(x_np, ord=o, keepdims=keepdims)
                    out_mx = mx.linalg.norm(x_mx, ord=o, keepdims=keepdims)
                    with self.subTest(shape=shape, ord=o, keepdims=keepdims):
                        self.assertTrue(
                            np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6)
                        )

        # Test no ord and no axis provided
        for shape in [(3,), (2, 3), (2, 3, 3)]:
            x_mx = mx.arange(1, math.prod(shape) + 1).reshape(shape)
            x_np = np.arange(1, math.prod(shape) + 1).reshape(shape)
            for keepdims in [True, False]:
                out_np = np.linalg.norm(x_np, keepdims=keepdims)
                out_mx = mx.linalg.norm(x_mx, keepdims=keepdims)
                with self.subTest(shape=shape, keepdims=keepdims):
                    self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6))

    def test_complex_norm(self):
        for shape in [(3,), (2, 3), (2, 3, 3)]:
            x_np = np.random.uniform(size=shape).astype(
                np.float32
            ) + 1j * np.random.uniform(size=shape).astype(np.float32)
            x_mx = mx.array(x_np)
            out_np = np.linalg.norm(x_np)
            out_mx = mx.linalg.norm(x_mx)
            with self.subTest(shape=shape):
                self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6))
            for num_axes in range(1, len(shape)):
                for axis in itertools.combinations(range(len(shape)), num_axes):
                    out_np = np.linalg.norm(x_np, axis=axis)
                    out_mx = mx.linalg.norm(x_mx, axis=axis)
                    with self.subTest(shape=shape, axis=axis):
                        self.assertTrue(
                            np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6)
                        )

        x_np = np.random.uniform(size=(4, 4)).astype(
            np.float32
        ) + 1j * np.random.uniform(size=(4, 4)).astype(np.float32)
        x_mx = mx.array(x_np)
        out_np = np.linalg.norm(x_np, ord="fro")
        out_mx = mx.linalg.norm(x_mx, ord="fro")
        self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6))

    def test_qr_factorization(self):
        with self.assertRaises(ValueError):
            mx.linalg.qr(mx.array(0.0))

        with self.assertRaises(ValueError):
            mx.linalg.qr(mx.array([0.0, 1.0]))

        with self.assertRaises(ValueError):
            mx.linalg.qr(mx.array([[0, 1], [1, 0]]))

        A = mx.array([[2.0, 3.0], [1.0, 2.0]])
        Q, R = mx.linalg.qr(A, stream=mx.cpu)
        out = Q @ R
        self.assertTrue(mx.allclose(out, A))
        out = Q.T @ Q
        self.assertTrue(mx.allclose(out, mx.eye(2), rtol=1e-5, atol=1e-7))
        self.assertTrue(mx.allclose(mx.tril(R, -1), mx.zeros_like(R)))
        self.assertEqual(Q.dtype, mx.float32)
        self.assertEqual(R.dtype, mx.float32)

        # Multiple matrices
        B = mx.array([[-1.0, 2.0], [-4.0, 1.0]])
        A = mx.stack([A, B])
        Q, R = mx.linalg.qr(A, stream=mx.cpu)
        for a, q, r in zip(A, Q, R):
            out = q @ r
            self.assertTrue(mx.allclose(out, a))
            out = q.T @ q
            self.assertTrue(mx.allclose(out, mx.eye(2), rtol=1e-5, atol=1e-7))
            self.assertTrue(mx.allclose(mx.tril(r, -1), mx.zeros_like(r)))

        # Non square matrices
        for shape in [(4, 8), (8, 4)]:
            A = mx.random.uniform(shape=shape)
            Q, R = mx.linalg.qr(A, stream=mx.cpu)
            out = Q @ R
            self.assertTrue(mx.allclose(out, A, rtol=1e-4, atol=1e-6))
            out = Q.T @ Q
            self.assertTrue(
                mx.allclose(out, mx.eye(min(A.shape)), rtol=1e-4, atol=1e-6)
            )

    def test_svd_decomposition(self):
        A = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=mx.float32)
        U, S, Vt = mx.linalg.svd(A, stream=mx.cpu)
        self.assertTrue(
            mx.allclose(U[:, : len(S)] @ mx.diag(S) @ Vt, A, rtol=1e-5, atol=1e-7)
        )

        # Multiple matrices
        B = A + 10.0
        AB = mx.stack([A, B])
        Us, Ss, Vts = mx.linalg.svd(AB, stream=mx.cpu)
        for M, U, S, Vt in zip([A, B], Us, Ss, Vts):
            self.assertTrue(
                mx.allclose(U[:, : len(S)] @ mx.diag(S) @ Vt, M, rtol=1e-5, atol=1e-7)
            )

    def test_inverse(self):
        A = mx.array([[1, 2, 3], [6, -5, 4], [-9, 8, 7]], dtype=mx.float32)
        A_inv = mx.linalg.inv(A, stream=mx.cpu)
        self.assertTrue(mx.allclose(A @ A_inv, mx.eye(A.shape[0]), rtol=0, atol=1e-6))

        # Multiple matrices
        B = A - 100
        AB = mx.stack([A, B])
        invs = mx.linalg.inv(AB, stream=mx.cpu)
        for M, M_inv in zip(AB, invs):
            self.assertTrue(
                mx.allclose(M @ M_inv, mx.eye(M.shape[0]), rtol=0, atol=1e-5)
            )

    def test_tri_inverse(self):
        for upper in (False, True):
            A = mx.array([[1, 0, 0], [6, -5, 0], [-9, 8, 7]], dtype=mx.float32)
            B = mx.array([[7, 0, 0], [3, -2, 0], [1, 8, 3]], dtype=mx.float32)
            if upper:
                A = A.T
                B = B.T
            AB = mx.stack([A, B])
            invs = mx.linalg.tri_inv(AB, upper=upper, stream=mx.cpu)
            for M, M_inv in zip(AB, invs):
                self.assertTrue(
                    mx.allclose(M @ M_inv, mx.eye(M.shape[0]), rtol=0, atol=1e-5)
                )

    def test_cholesky(self):
        sqrtA = mx.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=mx.float32
        )
        A = sqrtA.T @ sqrtA / 81
        L = mx.linalg.cholesky(A, stream=mx.cpu)
        U = mx.linalg.cholesky(A, upper=True, stream=mx.cpu)
        self.assertTrue(mx.allclose(L @ L.T, A, rtol=1e-5, atol=1e-7))
        self.assertTrue(mx.allclose(U.T @ U, A, rtol=1e-5, atol=1e-7))

        # Multiple matrices
        B = A + 1 / 9
        AB = mx.stack([A, B])
        Ls = mx.linalg.cholesky(AB, stream=mx.cpu)
        for M, L in zip(AB, Ls):
            self.assertTrue(mx.allclose(L @ L.T, M, rtol=1e-5, atol=1e-7))

    def test_pseudo_inverse(self):
        A = mx.array([[1, 2, 3], [6, -5, 4], [-9, 8, 7]], dtype=mx.float32)
        A_plus = mx.linalg.pinv(A, stream=mx.cpu)
        self.assertTrue(mx.allclose(A @ A_plus @ A, A, rtol=0, atol=1e-5))

        # Multiple matrices
        B = A - 100
        AB = mx.stack([A, B])
        pinvs = mx.linalg.pinv(AB, stream=mx.cpu)
        for M, M_plus in zip(AB, pinvs):
            self.assertTrue(mx.allclose(M @ M_plus @ M, M, rtol=0, atol=1e-3))

    def test_cholesky_inv(self):
        mx.random.seed(7)

        sqrtA = mx.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=mx.float32
        )
        A = sqrtA.T @ sqrtA / 81

        N = 3
        A = mx.random.uniform(shape=(N, N))
        A = A @ A.T

        for upper in (False, True):
            L = mx.linalg.cholesky(A, upper=upper, stream=mx.cpu)
            A_inv = mx.linalg.cholesky_inv(L, upper=upper, stream=mx.cpu)
            self.assertTrue(mx.allclose(A @ A_inv, mx.eye(N), atol=1e-4))

        # Multiple matrices
        B = A + 1 / 9
        AB = mx.stack([A, B])
        Ls = mx.linalg.cholesky(AB, stream=mx.cpu)
        for upper in (False, True):
            Ls = mx.linalg.cholesky(AB, upper=upper, stream=mx.cpu)
            AB_inv = mx.linalg.cholesky_inv(Ls, upper=upper, stream=mx.cpu)
            for M, M_inv in zip(AB, AB_inv):
                self.assertTrue(mx.allclose(M @ M_inv, mx.eye(N), atol=1e-4))

    def test_cross_product(self):
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        result = mx.linalg.cross(a, b)
        expected = np.cross(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test with negative values
        a = mx.array([-1.0, -2.0, -3.0])
        b = mx.array([4.0, -5.0, 6.0])
        result = mx.linalg.cross(a, b)
        expected = np.cross(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test with integer values
        a = mx.array([1, 2, 3])
        b = mx.array([4, 5, 6])
        result = mx.linalg.cross(a, b)
        expected = np.cross(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test with 2D arrays and axis parameter
        a = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = mx.array([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]])
        result = mx.linalg.cross(a, b, axis=1)
        expected = np.cross(a, b, axis=1)
        self.assertTrue(np.allclose(result, expected))

        # Test with broadcast
        a = mx.random.uniform(shape=(2, 1, 3))
        b = mx.random.uniform(shape=(1, 2, 3))
        result = mx.linalg.cross(a, b)
        expected = np.cross(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Type promotion
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4, 5, 6])
        result = mx.linalg.cross(a, b)
        expected = np.cross(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test with incorrect vector size (should raise an exception)
        a = mx.array([1.0])
        b = mx.array([4.0])
        with self.assertRaises(ValueError):
            mx.linalg.cross(a, b)

    def test_eigh(self):
        tols = {"atol": 1e-5, "rtol": 1e-5}

        def check_eigs_and_vecs(A_np, kwargs={}):
            A = mx.array(A_np)
            eig_vals, eig_vecs = mx.linalg.eigh(A, stream=mx.cpu, **kwargs)
            eig_vals_np, _ = np.linalg.eigh(A_np, **kwargs)
            self.assertTrue(np.allclose(eig_vals, eig_vals_np, **tols))
            self.assertTrue(
                mx.allclose(A @ eig_vecs, eig_vals[..., None, :] * eig_vecs, **tols)
            )

            eig_vals_only = mx.linalg.eigvalsh(A, stream=mx.cpu, **kwargs)
            self.assertTrue(mx.allclose(eig_vals, eig_vals_only, **tols))

        # Test a simple 2x2 symmetric matrix
        A_np = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        check_eigs_and_vecs(A_np)

        # Test a larger random symmetric matrix
        n = 5
        np.random.seed(1)
        A_np = np.random.randn(n, n).astype(np.float32)
        A_np = (A_np + A_np.T) / 2
        check_eigs_and_vecs(A_np)

        # Test with upper triangle
        check_eigs_and_vecs(A_np, {"UPLO": "U"})

        # Test with batched input
        A_np = np.random.randn(3, n, n).astype(np.float32)
        A_np = (A_np + np.transpose(A_np, (0, 2, 1))) / 2
        check_eigs_and_vecs(A_np)

        # Test error cases
        with self.assertRaises(ValueError):
            mx.linalg.eigh(mx.array([1.0, 2.0]))  # 1D array

        with self.assertRaises(ValueError):
            mx.linalg.eigh(
                mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )  # Non-square matrix

        with self.assertRaises(ValueError):
            mx.linalg.eigvalsh(mx.array([1.0, 2.0]))  # 1D array

        with self.assertRaises(ValueError):
            mx.linalg.eigvalsh(
                mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )  # Non-square matrix


if __name__ == "__main__":
    unittest.main()
