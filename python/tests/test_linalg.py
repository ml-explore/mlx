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
        matrix_ords = [None, "fro", "nuc", -1, 1, -2, 2, float("inf"), -float("inf")]

        for shape in [(3,), (2, 3), (2, 3, 3)]:
            x_mx = mx.arange(1, math.prod(shape) + 1, dtype=mx.float32).reshape(shape)
            x_np = np.arange(1, math.prod(shape) + 1, dtype=np.float32).reshape(shape)
            # Test when at least one axis is provided
            for num_axes in range(1, len(shape)):
                if num_axes == 1:
                    ords = vector_ords
                else:
                    ords = matrix_ords
                for axis in itertools.combinations(range(len(shape)), num_axes):
                    for keepdims in [True, False]:
                        for o in ords:
                            stream = (
                                mx.cpu if o in ["nuc", -2, 2] else mx.default_device()
                            )
                            out_np = np.linalg.norm(
                                x_np, ord=o, axis=axis, keepdims=keepdims
                            )
                            out_mx = mx.linalg.norm(
                                x_mx, ord=o, axis=axis, keepdims=keepdims, stream=stream
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
        U, S, Vt = mx.linalg.svd(A, compute_uv=True, stream=mx.cpu)
        self.assertTrue(
            mx.allclose(U[:, : len(S)] @ mx.diag(S) @ Vt, A, rtol=1e-5, atol=1e-7)
        )

        S = mx.linalg.svd(A, compute_uv=False, stream=mx.cpu)
        self.assertTrue(
            mx.allclose(
                mx.linalg.norm(S), mx.linalg.norm(A, ord="fro"), rtol=1e-5, atol=1e-7
            )
        )

        # Multiple matrices
        B = A + 10.0
        AB = mx.stack([A, B])
        Us, Ss, Vts = mx.linalg.svd(AB, compute_uv=True, stream=mx.cpu)
        for M, U, S, Vt in zip([A, B], Us, Ss, Vts):
            self.assertTrue(
                mx.allclose(U[:, : len(S)] @ mx.diag(S) @ Vt, M, rtol=1e-5, atol=1e-7)
            )

        Ss = mx.linalg.svd(AB, compute_uv=False, stream=mx.cpu)
        for M, S in zip([A, B], Ss):
            self.assertTrue(
                mx.allclose(
                    mx.linalg.norm(S),
                    mx.linalg.norm(M, ord="fro"),
                    rtol=1e-5,
                    atol=1e-7,
                )
            )

        # Test float64 - use CPU stream since float64 is not supported on GPU
        with mx.stream(mx.cpu):
            A_f64 = mx.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=mx.float64
            )
            U_f64, S_f64, Vt_f64 = mx.linalg.svd(A_f64, compute_uv=True)
            mx.eval(U_f64, S_f64, Vt_f64)
            self.assertTrue(
                mx.allclose(
                    U_f64[:, : len(S_f64)] @ mx.diag(S_f64) @ Vt_f64,
                    A_f64,
                    rtol=1e-5,
                    atol=1e-7,
                )
            )
            self.assertEqual(S_f64.dtype, mx.float64)

        # Test complex64 - use CPU stream since complex64 is not supported on GPU
        with mx.stream(mx.cpu):
            A_c64 = mx.array(
                [[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]], dtype=mx.complex64
            )
            U_c64, S_c64, Vt_c64 = mx.linalg.svd(A_c64, compute_uv=True)
            mx.eval(U_c64, S_c64, Vt_c64)
            self.assertTrue(
                mx.allclose(
                    U_c64[:, : len(S_c64)] @ mx.diag(S_c64) @ Vt_c64,
                    A_c64,
                    rtol=1e-5,
                    atol=1e-7,
                )
            )
            self.assertEqual(S_c64.dtype, mx.float32)
            self.assertEqual(U_c64.dtype, mx.complex64)
            self.assertEqual(Vt_c64.dtype, mx.complex64)

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

        # Ensure that tri_inv will 0-out the supposedly 0 triangle
        x = mx.random.normal((2, 8, 8))
        y1 = mx.linalg.tri_inv(x, upper=True, stream=mx.cpu)
        y2 = mx.linalg.tri_inv(x, upper=False, stream=mx.cpu)
        self.assertTrue(mx.all(y1 == mx.triu(y1)))
        self.assertTrue(mx.all(y2 == mx.tril(y2)))

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

        # Test singular matrix
        A = mx.array([[4.0, 1.0], [4.0, 1.0]])
        A_plus = mx.linalg.pinv(A, stream=mx.cpu)
        self.assertTrue(mx.allclose(A @ A_plus @ A, A))

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

    def test_eig(self):
        tols = {"atol": 1e-5, "rtol": 1e-5}

        def check_eigs_and_vecs(A_np, kwargs={}):
            A = mx.array(A_np)
            eig_vals, eig_vecs = mx.linalg.eig(A, stream=mx.cpu, **kwargs)
            self.assertTrue(
                mx.allclose(A @ eig_vecs, eig_vals[..., None, :] * eig_vecs, **tols)
            )
            eig_vals_only = mx.linalg.eigvals(A, stream=mx.cpu, **kwargs)
            self.assertTrue(mx.allclose(eig_vals, eig_vals_only, **tols))

        # Test a simple 2x2 matrix
        A_np = np.array([[1.0, 1.0], [3.0, 4.0]], dtype=np.float32)
        check_eigs_and_vecs(A_np)

        # Test complex eigenvalues
        A_np = np.array([[1.0, -1.0], [1.0, 1.0]], dtype=np.float32)
        check_eigs_and_vecs(A_np)

        # Test a larger random symmetric matrix
        n = 5
        np.random.seed(1)
        A_np = np.random.randn(n, n).astype(np.float32)
        check_eigs_and_vecs(A_np)

        # Test with batched input
        A_np = np.random.randn(3, n, n).astype(np.float32)
        check_eigs_and_vecs(A_np)

        # Test float64 - use CPU stream since float64 is not supported on GPU
        with mx.stream(mx.cpu):
            A_np_f64 = np.array([[1.0, 1.0], [3.0, 4.0]], dtype=np.float64)
            A_f64 = mx.array(A_np_f64, dtype=mx.float64)
            eig_vals_f64, eig_vecs_f64 = mx.linalg.eig(A_f64)
            mx.eval(eig_vals_f64, eig_vecs_f64)
            self.assertTrue(
                mx.allclose(
                    A_f64 @ eig_vecs_f64,
                    eig_vals_f64[..., None, :] * eig_vecs_f64,
                    rtol=1e-5,
                    atol=1e-5,
                )
            )
            # Eigenvalues should be complex64 (output dtype)
            self.assertEqual(eig_vals_f64.dtype, mx.complex64)
            self.assertEqual(eig_vecs_f64.dtype, mx.complex64)

        # Test complex64 input - use CPU stream since complex64 is not supported on GPU
        with mx.stream(mx.cpu):
            A_np_c64 = np.array(
                [[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]], dtype=np.complex64
            )
            A_c64 = mx.array(A_np_c64, dtype=mx.complex64)
            eig_vals_c64, eig_vecs_c64 = mx.linalg.eig(A_c64)
            mx.eval(eig_vals_c64, eig_vecs_c64)
            self.assertTrue(
                mx.allclose(
                    A_c64 @ eig_vecs_c64,
                    eig_vals_c64[..., None, :] * eig_vecs_c64,
                    rtol=1e-5,
                    atol=1e-5,
                )
            )
            self.assertEqual(eig_vals_c64.dtype, mx.complex64)
            self.assertEqual(eig_vecs_c64.dtype, mx.complex64)

        # Test error cases
        with self.assertRaises(ValueError):
            mx.linalg.eig(mx.array([1.0, 2.0]))  # 1D array

        with self.assertRaises(ValueError):
            mx.linalg.eig(
                mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )  # Non-square matrix

        with self.assertRaises(ValueError):
            mx.linalg.eigvals(mx.array([1.0, 2.0]))  # 1D array

        with self.assertRaises(ValueError):
            mx.linalg.eigvals(
                mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            )  # Non-square matrix

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

        # Test with complex inputs
        A_np = (
            np.random.randn(8, 8, 2).astype(np.float32).view(np.complex64).squeeze(-1)
        )
        A_np = A_np + A_np.T.conj()
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

    def test_lu(self):
        with self.assertRaises(ValueError):
            mx.linalg.lu(mx.array(0.0), stream=mx.cpu)

        with self.assertRaises(ValueError):
            mx.linalg.lu(mx.array([0.0, 1.0]), stream=mx.cpu)

        with self.assertRaises(ValueError):
            mx.linalg.lu(mx.array([[0, 1], [1, 0]]), stream=mx.cpu)

        # Test 3x3 matrix
        a = mx.array([[3.0, 1.0, 2.0], [1.0, 8.0, 6.0], [9.0, 2.0, 5.0]])
        P, L, U = mx.linalg.lu(a, stream=mx.cpu)
        self.assertTrue(mx.allclose(L[P, :] @ U, a))

        # Test batch dimension
        a = mx.broadcast_to(a, (5, 5, 3, 3))
        P, L, U = mx.linalg.lu(a, stream=mx.cpu)
        L = mx.take_along_axis(L, P[..., None], axis=-2)
        self.assertTrue(mx.allclose(L @ U, a))

        # Test non-square matrix
        a = mx.array([[3.0, 1.0, 2.0], [1.0, 8.0, 6.0]])
        P, L, U = mx.linalg.lu(a, stream=mx.cpu)
        self.assertTrue(mx.allclose(L[P, :] @ U, a))

        a = mx.array([[3.0, 1.0], [1.0, 8.0], [9.0, 2.0]])
        P, L, U = mx.linalg.lu(a, stream=mx.cpu)
        self.assertTrue(mx.allclose(L[P, :] @ U, a))

    def test_lu_factor(self):
        mx.random.seed(7)

        # Test 3x3 matrix
        a = mx.random.uniform(shape=(5, 5))
        LU, pivots = mx.linalg.lu_factor(a, stream=mx.cpu)
        n = a.shape[-1]

        pivots = pivots.tolist()
        perm = list(range(n))
        for i in range(len(pivots)):
            perm[i], perm[pivots[i]] = perm[pivots[i]], perm[i]

        L = mx.add(mx.tril(LU, k=-1), mx.eye(n))
        U = mx.triu(LU)
        self.assertTrue(mx.allclose(L @ U, a[perm, :]))

    def test_solve(self):
        mx.random.seed(7)

        # Test 3x3 matrix with 1D rhs
        a = mx.array([[3.0, 1.0, 2.0], [1.0, 8.0, 6.0], [9.0, 2.0, 5.0]])
        b = mx.array([11.0, 35.0, 28.0])

        result = mx.linalg.solve(a, b, stream=mx.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test symmetric positive-definite matrix
        N = 5
        a = mx.random.uniform(shape=(N, N))
        a = mx.matmul(a, a.T) + N * mx.eye(N)
        b = mx.random.uniform(shape=(N, 1))

        result = mx.linalg.solve(a, b, stream=mx.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test batch dimension
        a = mx.random.uniform(shape=(5, 5, 4, 4))
        b = mx.random.uniform(shape=(5, 5, 4, 1))

        result = mx.linalg.solve(a, b, stream=mx.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected, atol=1e-5))

        # Test large matrix
        N = 1000
        a = mx.random.uniform(shape=(N, N))
        b = mx.random.uniform(shape=(N, 1))

        result = mx.linalg.solve(a, b, stream=mx.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected, atol=1e-3))

        # Test multi-column rhs
        a = mx.random.uniform(shape=(5, 5))
        b = mx.random.uniform(shape=(5, 8))

        result = mx.linalg.solve(a, b, stream=mx.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test batched multi-column rhs
        a = mx.broadcast_to(a, (3, 2, 5, 5))
        b = mx.broadcast_to(b, (3, 1, 5, 8))

        result = mx.linalg.solve(a, b, stream=mx.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected, rtol=1e-5, atol=1e-5))

    def test_solve_triangular(self):
        # Test lower triangular matrix
        a = mx.array([[4.0, 0.0, 0.0], [2.0, 3.0, 0.0], [1.0, -2.0, 5.0]])
        b = mx.array([8.0, 14.0, 3.0])

        result = mx.linalg.solve_triangular(a, b, upper=False, stream=mx.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test upper triangular matrix
        a = mx.array([[3.0, 2.0, 1.0], [0.0, 5.0, 4.0], [0.0, 0.0, 6.0]])
        b = mx.array([13.0, 33.0, 18.0])

        result = mx.linalg.solve_triangular(a, b, upper=True, stream=mx.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected))

        # Test batch multi-column rhs
        a = mx.broadcast_to(a, (3, 4, 3, 3))
        b = mx.broadcast_to(mx.expand_dims(b, -1), (3, 4, 3, 8))

        result = mx.linalg.solve_triangular(a, b, upper=True, stream=mx.cpu)
        expected = np.linalg.solve(a, b)
        self.assertTrue(np.allclose(result, expected))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
