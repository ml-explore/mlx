# Copyright © 2023-2024 Apple Inc.

import math
import unittest
from itertools import permutations

import mlx.core as mx
import mlx_tests
import numpy as np


class TestBlas(mlx_tests.MLXTestCase):
    @property
    def dtypes(self):
        return ["float32", "float16"]

    def __gemm_test(
        self,
        shape_a,
        shape_b,
        np_dtype=np.float32,
        f_np_a=lambda x: x,
        f_np_b=lambda x: x,
        f_mx_a=lambda x: x,
        f_mx_b=lambda x: x,
    ):
        with self.subTest(
            dtype=np.dtype(np_dtype).name, shape_a=shape_a, shape_b=shape_b
        ):
            np.random.seed(42)
            scale = max(np.sum(shape_a), 128)
            a_np = np.random.normal(0.0, 1.0 / scale, shape_a).astype(np_dtype)
            b_np = np.random.normal(0.0, 1.0 / scale, shape_b).astype(np_dtype)

            a_mx = mx.array(a_np)
            b_mx = mx.array(b_np)

            a_np = f_np_a(a_np.astype(np.float32))
            b_np = f_np_b(b_np.astype(np.float32))
            a_mx = f_mx_a(a_mx)
            b_mx = f_mx_b(b_mx)

            out_npy = a_np @ b_np
            out_mlx = a_mx @ b_mx

            self.assertListEqual(list(out_npy.shape), list(out_mlx.shape))
            self.assertTrue(np.allclose(out_mlx, out_npy.astype(np_dtype), atol=1e-5))

    def test_matmul_unaligned(self):
        if not mx.is_available(mx.gpu):
            return

        for dtype in self.dtypes:
            np_dtype = getattr(np, dtype)
            base_shapes = [4, 8, 16, 32, 64, 128]
            perturbations = [-2, -1, 0, 1, 2]

            for dim in base_shapes:
                for p in perturbations:
                    shape_a = (dim + p, dim + p)
                    shape_b = (dim + p, dim + p)
                    self.__gemm_test(shape_a, shape_b, np_dtype)

    def test_matvec_unaligned(self):
        a = mx.random.normal(shape=(4, 128))
        b = mx.random.normal(shape=(129,))[1:]
        out = a @ b
        np_out = np.array(a) @ np.array(b)
        self.assertTrue(np.allclose(out, np_out))

    def test_matmul_shapes(self):
        if not mx.is_available(mx.gpu):
            return

        shapes = [
            (1, 2, 1, 1),
            (1, 1, 2, 1),
            (3, 23, 457, 3),
        ]

        if mx.default_device() == mx.gpu:
            shapes += [
                (16, 768, 768, 128),
                (1, 64, 64, 4096),
            ]

        for dtype in self.dtypes:
            np_dtype = getattr(np, dtype)

            for B, M, N, K in shapes:
                with self.subTest(transpose="nn"):
                    shape_a = (B, M, K)
                    shape_b = (B, K, N)
                    self.__gemm_test(shape_a, shape_b, np_dtype)

                with self.subTest(transpose="nt"):
                    shape_a = (B, M, K)
                    shape_b = (B, N, K)
                    self.__gemm_test(
                        shape_a,
                        shape_b,
                        np_dtype,
                        f_np_b=lambda x: np.transpose(x, (0, 2, 1)),
                        f_mx_b=lambda x: mx.transpose(x, (0, 2, 1)),
                    )

                with self.subTest(transpose="tn"):
                    shape_a = (B, K, M)
                    shape_b = (B, K, N)
                    self.__gemm_test(
                        shape_a,
                        shape_b,
                        np_dtype,
                        f_np_a=lambda x: np.transpose(x, (0, 2, 1)),
                        f_mx_a=lambda x: mx.transpose(x, (0, 2, 1)),
                    )

                with self.subTest(transpose="tt"):
                    shape_a = (B, K, M)
                    shape_b = (B, N, K)
                    self.__gemm_test(
                        shape_a,
                        shape_b,
                        np_dtype,
                        f_np_a=lambda x: np.transpose(x, (0, 2, 1)),
                        f_mx_a=lambda x: mx.transpose(x, (0, 2, 1)),
                        f_np_b=lambda x: np.transpose(x, (0, 2, 1)),
                        f_mx_b=lambda x: mx.transpose(x, (0, 2, 1)),
                    )

    def test_matmul(self):
        # Note: so far, matmul only works with floating-point types
        a = mx.array([[1.0, 2.0], [3.0, 4.0]])

        b = mx.array([[0.0, -1.0], [-3.0, 3.0]])

        expected = [[-6.0, 5.0], [-12.0, 9.0]]

        self.assertEqual((a @ b).tolist(), expected)
        self.assertEqual(mx.matmul(a, b).tolist(), expected)

        # Transposed matmul
        np.random.seed(0)
        a_npy = np.random.normal(0.0, 1.0 / 128, (128, 16)).astype(np.float32)
        b_npy = np.random.normal(0.0, 1.0 / 128, (128, 16)).astype(np.float32)
        c_npy = a_npy @ np.transpose(b_npy, (1, 0))
        d_npy = np.transpose(a_npy, (1, 0)) @ b_npy

        a_mlx = mx.array(a_npy)
        b_mlx = mx.array(b_npy)
        c_mlx = a_mlx @ mx.transpose(b_mlx, (1, 0))
        d_mlx = mx.transpose(a_mlx, (1, 0)) @ b_mlx

        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertListEqual(list(d_npy.shape), list(d_mlx.shape))

        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-6))
        self.assertTrue(np.allclose(d_mlx, d_npy, atol=1e-6))

    def test_matmul_dtypes(self):
        for dt in self.dtypes:
            a_npy = np.random.normal(0.0, 1.0 / 256, (16, 16, 16)).astype(
                getattr(np, dt)
            )
            b_npy = np.random.normal(0.0, 1.0 / 256, (16, 16, 16)).astype(
                getattr(np, dt)
            )
            a_mlx = mx.array(a_npy)
            b_mlx = mx.array(b_npy)

            c_npy = np.matmul(a_npy, b_npy, dtype=getattr(np, dt))
            c_mlx = a_mlx @ b_mlx

            self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-6))

    def test_matmul_batched(self):
        np.random.seed(0)
        # Batched matmul
        a_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
        b_npy = np.random.normal(0.0, 1.0 / 128, (32, 16, 16)).astype(np.float32)
        c_npy = a_npy @ b_npy

        a_mlx = mx.array(a_npy)
        b_mlx = mx.array(b_npy)
        c_mlx = a_mlx @ b_mlx

        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-6))

        # Batched and transposed matmul
        b_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
        c_npy = a_npy @ np.transpose(b_npy, (0, 2, 1))

        b_mlx = mx.array(b_npy)
        c_mlx = a_mlx @ mx.transpose(b_mlx, (0, 2, 1))

        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-6))

        # Batched matmul with simple broadcast
        a_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
        b_npy = np.random.normal(0.0, 1.0 / 128, (16, 16)).astype(np.float32)
        c_npy = a_npy @ b_npy

        a_mlx = mx.array(a_npy)
        b_mlx = mx.array(b_npy)
        c_mlx = a_mlx @ b_mlx

        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-6))

        # Both operands broadcasted
        d_npy = np.broadcast_to(b_npy, (5, 16, 16))
        d_mlx = mx.broadcast_to(b_mlx, (5, 16, 16))

        e_npy = d_npy @ d_npy
        e_mlx = d_mlx @ d_mlx

        self.assertListEqual(list(e_npy.shape), list(e_mlx.shape))
        self.assertTrue(np.allclose(e_mlx, e_npy, atol=1e-6))

        # Batched and transposed matmul with simple broadcast
        a_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
        b_npy = np.random.normal(0.0, 1.0 / 128, (128, 16)).astype(np.float32)
        a_mlx = mx.array(a_npy)
        b_mlx = mx.array(b_npy)

        c_npy = a_npy @ np.transpose(b_npy, (1, 0))
        c_mlx = a_mlx @ mx.transpose(b_mlx, (1, 0))

        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-6))

        # Matmul with vector
        a_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
        b_npy = np.random.normal(0.0, 1.0 / 128, (16,)).astype(np.float32)
        a_mlx = mx.array(a_npy)
        b_mlx = mx.array(b_npy)

        c_npy = a_npy @ b_npy
        c_mlx = a_mlx @ b_mlx

        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-6))

        # Test Multiheaded attention style matmul
        a_npy = np.random.normal(0.0, 1.0 / 128, (64, 16, 4, 32)).astype(np.float32)
        b_npy = np.random.normal(0.0, 1.0 / 128, (64, 16, 4, 32)).astype(np.float32)
        a_mlx = mx.array(a_npy)
        b_mlx = mx.array(b_npy)

        a_npy = np.transpose(a_npy, (0, 2, 1, 3))
        b_npy = np.transpose(b_npy, (0, 2, 1, 3))
        a_mlx = mx.transpose(a_mlx, (0, 2, 1, 3))
        b_mlx = mx.transpose(b_mlx, (0, 2, 1, 3))

        c_npy = a_npy @ np.transpose(b_npy, (0, 1, 3, 2))
        c_mlx = a_mlx @ mx.transpose(b_mlx, (0, 1, 3, 2))
        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-6))

    def __gemv_test(
        self,
        shape_mat,
        shape_vec,
        np_dtype=np.float32,
        mat_first=True,
        np_mat_f=lambda x: x,
        np_vec_f=lambda x: x,
        mlx_mat_f=lambda x: x,
        mlx_vec_f=lambda x: x,
    ):
        with self.subTest(
            shape_mat=shape_mat, shape_vec=shape_vec, mat_first=mat_first
        ):
            np.random.seed(42)
            scale = max(np.sum(shape_mat), 32)
            mat_npy = np.random.normal(0.0, 1.0 / scale, shape_mat).astype(np_dtype)
            vec_npy = np.random.normal(0.0, 1.0 / scale, shape_vec).astype(np_dtype)

            mat_mlx = mx.array(mat_npy)
            vec_mlx = mx.array(vec_npy)

            mat_npy = np_mat_f(mat_npy)
            vec_npy = np_vec_f(vec_npy)
            mat_mlx = mlx_mat_f(mat_mlx)
            vec_mlx = mlx_vec_f(vec_mlx)

            if mat_first:
                out_npy = mat_npy @ vec_npy
                out_mlx = mat_mlx @ vec_mlx
            else:
                out_npy = vec_npy @ mat_npy
                out_mlx = vec_mlx @ mat_mlx

            self.assertListEqual(list(out_npy.shape), list(out_mlx.shape))
            self.assertTrue(np.allclose(out_mlx, out_npy, atol=1e-5))

    def test_matrix_vector(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                np_dtype = getattr(np, dtype)

                # Basic square matrix test
                self.__gemv_test(
                    shape_mat=(64, 64), shape_vec=(64, 1), np_dtype=np_dtype
                )
                self.__gemv_test(
                    shape_mat=(64, 64),
                    shape_vec=(64, 1),
                    np_dtype=np_dtype,
                    mat_first=False,
                    np_vec_f=lambda x: np.transpose(x, (1, 0)),
                    mlx_vec_f=lambda x: mx.transpose(x, (1, 0)),
                )

                # Vector matrix product with aligned and unaligned shapes
                for in_len_base, out_len_base in (
                    (2, 2),
                    (32, 32),
                    (64, 64),
                    (2048, 2048),
                ):
                    for mi in (-1, 0, 1):
                        for mj in (-1, 0, 1):
                            # Vec mat
                            shape_mat = (in_len_base + mi, out_len_base + mj)
                            shape_vec = (1, in_len_base + mi)
                            self.__gemv_test(
                                shape_mat, shape_vec, mat_first=False, np_dtype=np_dtype
                            )

                            # Mat vec
                            shape_mat = (out_len_base + mj, in_len_base + mi)
                            shape_vec = (in_len_base + mi, 1)
                            self.__gemv_test(
                                shape_mat, shape_vec, mat_first=True, np_dtype=np_dtype
                            )

    def test_matrix_vector_batched(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                np_dtype = getattr(np, dtype)

                # Batched mat vec
                for shape_mat, shape_vec in (
                    ((32, 128, 64), (32, 64, 1)),
                    ((128, 64), (32, 64, 1)),
                    ((32, 128, 64), (64, 1)),
                    ((2, 1, 8, 1, 6, 128), (2, 1, 8, 4, 128, 1)),
                ):
                    self.__gemv_test(
                        shape_mat, shape_vec, mat_first=True, np_dtype=np_dtype
                    )

                # Batched vec mat
                for shape_vec, shape_mat in (
                    ((32, 1, 128), (32, 128, 64)),
                    ((32, 1, 128), (128, 64)),
                    ((1, 128), (32, 128, 64)),
                    ((1, 8, 4, 1, 128), (1, 8, 1, 128, 6)),
                ):
                    self.__gemv_test(
                        shape_mat, shape_vec, mat_first=False, np_dtype=np_dtype
                    )

    def test_matrix_vector_broadcast(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                np_dtype = getattr(np, dtype)

                # Different broadcasts mat vec
                for shape_mat, shape_vec in (
                    ((32, 64, 64), (32, 64, 1)),
                    ((64, 64), (32, 64, 1)),
                    ((32, 64, 64), (64, 1)),
                ):
                    self.__gemv_test(
                        shape_mat=(64, 64),
                        shape_vec=(64, 1),
                        np_dtype=np_dtype,
                        np_mat_f=(lambda mat_npy: np.broadcast_to(mat_npy, shape_mat)),
                        np_vec_f=(lambda vec_npy: np.broadcast_to(vec_npy, shape_vec)),
                        mlx_mat_f=(lambda mat_mlx: mx.broadcast_to(mat_mlx, shape_mat)),
                        mlx_vec_f=(lambda vec_mlx: mx.broadcast_to(vec_mlx, shape_vec)),
                    )

                # Different broadcasts vec mat
                for shape_vec, shape_mat in (
                    ((32, 1, 64), (32, 64, 64)),
                    ((32, 1, 64), (64, 64)),
                    ((1, 64), (32, 64, 64)),
                ):
                    self.__gemv_test(
                        shape_mat=(64, 64),
                        shape_vec=(1, 64),
                        np_dtype=np_dtype,
                        mat_first=False,
                        np_mat_f=lambda mat_npy: np.broadcast_to(mat_npy, shape_mat),
                        np_vec_f=lambda vec_npy: np.broadcast_to(vec_npy, shape_vec),
                        mlx_mat_f=lambda mat_mlx: mx.broadcast_to(mat_mlx, shape_mat),
                        mlx_vec_f=lambda vec_mlx: mx.broadcast_to(vec_mlx, shape_vec),
                    )

    def test_matrix_vector_attn(self):
        # Multi-query style attention check
        for dtype in self.dtypes:
            # fmt: off
            for (B,  D, n_kv_heads, factor,  qsl,  ksl) in (
                (1, 16,          8,      4,    1,  256),
                (1, 16,          8,      4,   32,  256),
                (1, 16,          8,      4,  256,    1),
                (4, 16,          8,      4,    1,  256),
                (4, 16,          8,      4,  256,    1),
            ):
            # fmt: on
                with self.subTest(
                        B=B, # Batch size
                        D=D, # Dimension of mm
                        n_kv_heads=n_kv_heads, # key-value heads
                        factor=factor, # factor to get query heads
                        qsl=qsl, # Query sequence length
                        ksl=ksl, # Key sequence length
                        dtype=dtype # Data type
                    ):

                    np_dtype = getattr(np, dtype)

                    # Fix shapes for kqv
                    n_q_heads = n_kv_heads * factor
                    Dk = D * n_kv_heads
                    Dq = D * n_q_heads
                    scale = 1. / math.sqrt(Dk)

                    shape_queries = (B, qsl, Dq)
                    shape_keys = (B, ksl, Dk)
                    shape_values = (B, ksl, Dk)

                    # Prepare numpy arrays
                    q_np = np.random.uniform(-scale, scale, size=shape_queries).astype(np_dtype)
                    k_np = np.random.uniform(-scale, scale, size=shape_keys).astype(np_dtype)
                    v_np = np.random.uniform(-scale, scale, size=shape_values).astype(np_dtype)

                    # Rearrange to move heads up
                    q_np_reshape = q_np.reshape(B, qsl, n_kv_heads, factor, -1).transpose(0, 2, 3, 1, 4)
                    k_np_reshape = k_np.reshape(B, ksl, n_kv_heads, 1, -1).transpose(0, 2, 3, 4, 1)
                    v_np_reshape = v_np.reshape(B, ksl, n_kv_heads, 1, -1).transpose(0, 2, 3, 1, 4)

                    # Do attn style matmul
                    s_np = q_np_reshape @ k_np_reshape
                    o_np = s_np @ v_np_reshape
                    o_np = o_np.transpose(0, 3, 1, 2, 4).reshape(B, qsl, -1)

                    # Test mlx
                    q_mx = mx.array(q_np)
                    k_mx = mx.array(k_np)
                    v_mx = mx.array(v_np)

                    # Rearrange to move heads up
                    q_mx_reshape = q_mx.reshape(B, qsl, n_kv_heads, factor, -1).transpose(0, 2, 3, 1, 4)
                    k_mx_reshape = k_mx.reshape(B, ksl, n_kv_heads, 1, -1).transpose(0, 2, 3, 4, 1)
                    v_mx_reshape = v_mx.reshape(B, ksl, n_kv_heads, 1, -1).transpose(0, 2, 3, 1, 4)

                    # Do attn style matmul
                    s_mx = q_mx_reshape @ k_mx_reshape
                    o_mx = (s_mx @ v_mx_reshape)
                    o_mx = o_mx.transpose(0, 3, 1, 2, 4).reshape(B, qsl, -1)

                    # Check against np
                    self.assertListEqual(list(s_np.shape), list(s_mx.shape))
                    self.assertTrue(np.allclose(s_np, s_mx, atol=1e-4))

                    self.assertListEqual(list(o_np.shape), list(o_mx.shape))
                    self.assertTrue(np.allclose(o_np, o_mx, atol=1e-4))

    def test_matrix_vector_edgecases(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                np_dtype = getattr(np, dtype)

                for in_vec_len in np.arange(1, 5):
                    for out_vec_len in np.arange(1, 5):
                        for batch_size in np.arange(1, 5):
                            with self.subTest(
                                problem_shape=(batch_size, in_vec_len, out_vec_len)
                            ):
                                # Matrix vector
                                with self.subTest(transpose=False):
                                    a_npy = np.ones(
                                        (batch_size, out_vec_len, in_vec_len),
                                        dtype=np_dtype,
                                    )
                                    b_npy = np.ones(
                                        (batch_size, in_vec_len, 1), dtype=np_dtype
                                    )
                                    for i in range(batch_size):
                                        b_npy[i] *= i + 1.0

                                    a_mlx, b_mlx = map(mx.array, [a_npy, b_npy])
                                    c_npy = a_npy @ b_npy
                                    c_mlx = a_mlx @ b_mlx

                                    self.assertListEqual(
                                        list(c_npy.shape), list(c_mlx.shape)
                                    )
                                    self.assertTrue(np.array_equal(c_mlx, c_npy))

                                # Vector matrix
                                with self.subTest(transpose=True):
                                    a_npy = np.ones(
                                        (batch_size, out_vec_len, in_vec_len),
                                        dtype=np_dtype,
                                    )
                                    b_npy = np.ones(
                                        (batch_size, 1, out_vec_len), dtype=np_dtype
                                    )
                                    for i in range(batch_size):
                                        b_npy[i] *= i + 1.0

                                    a_mlx, b_mlx = map(mx.array, [a_npy, b_npy])
                                    c_npy = b_npy @ a_npy
                                    c_mlx = b_mlx @ a_mlx

                                    self.assertListEqual(
                                        list(c_npy.shape), list(c_mlx.shape)
                                    )
                                    self.assertTrue(np.array_equal(c_mlx, c_npy))

    def test_mismatch_stride_mm(self):
        np.random.seed(0)
        a_npy = np.random.normal(0.0, 1.0 / 128, (4, 16, 16)).astype(np.float32)
        b_npy = np.random.normal(0.0, 1.0 / 128, (4, 16, 16)).astype(np.float32)

        a_mlx = mx.array(a_npy)
        b_mlx = mx.array(b_npy)

        # Matmul with batches
        c_npy = a_npy[::2, :, :] @ b_npy[1::2, :, :]
        c_mlx = a_mlx[::2, :, :] @ b_mlx[1::2, :, :]

        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-5))

        # Matvec with batches
        c_npy = a_npy[::2, :, :] @ b_npy[1::2, :, 2:3]
        c_mlx = a_mlx[::2, :, :] @ b_mlx[1::2, :, 2:3]

        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-5))

        # Matmul with slice
        c_npy = a_npy[:, :8, :] @ b_npy[:, :, :8]
        c_mlx = a_mlx[:, :8, :] @ b_mlx[:, :, :8]

        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-5))

        # Matmul with slice
        c_npy = a_npy[:, :, :8] @ b_npy[:, :8, :]
        c_mlx = a_mlx[:, :, :8] @ b_mlx[:, :8, :]

        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-5))

        # Matmul transpose with slice
        c_npy = a_npy[:, :8, :] @ b_npy[:, :8, :].swapaxes(-1, -2)
        c_mlx = a_mlx[:, :8, :] @ b_mlx[:, :8, :].swapaxes(-1, -2)

        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-5))

        # Matmul transpose with slice
        c_npy = a_npy[:, :, :8] @ b_npy[:, :, :8].swapaxes(-1, -2)
        c_mlx = a_mlx[:, :, :8] @ b_mlx[:, :, :8].swapaxes(-1, -2)

        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-5))

        # Matvec with slice
        c_npy = a_npy[:, :8, :] @ b_npy[:, :, 6:7]
        c_mlx = a_mlx[:, :8, :] @ b_mlx[:, :, 6:7]

        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-5))

        # Matvec with slice
        c_npy = a_npy[:, :, :8] @ b_npy[:, 3:11, 2:3]
        c_mlx = a_mlx[:, :, :8] @ b_mlx[:, 3:11, 2:3]

        self.assertListEqual(list(c_npy.shape), list(c_mlx.shape))
        self.assertTrue(np.allclose(c_mlx, c_npy, atol=1e-5))

    def test_addmm(self):
        np.random.seed(0)
        # Batched matmul
        alpha = 0.5
        for beta in (1.0, 2.0):
            # c must broadcast to the output shape
            with self.assertRaises(ValueError):
                mx.addmm(mx.zeros((2, 2, 2)), mx.zeros((2, 2)), mx.zeros((2, 2)))

            # Regular batched case
            a_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
            b_npy = np.random.normal(0.0, 1.0 / 128, (32, 16, 16)).astype(np.float32)

            a_mlx = mx.array(a_npy)
            b_mlx = mx.array(b_npy)

            for c_shape in ((1,), (1, 16), (32, 1, 16), (1, 128, 16)):
                c_npy = np.ones(c_shape).astype(np.float32)
                c_mlx = mx.array(c_npy)

                d_npy = alpha * (a_npy @ b_npy) + beta * c_npy
                d_mlx = mx.addmm(c_mlx, a_mlx, b_mlx, alpha, beta)

                self.assertListEqual(list(d_npy.shape), list(d_mlx.shape))
                self.assertTrue(np.allclose(d_mlx, d_npy, atol=1e-5))

            # Batched and transposed matmul
            b_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
            b_mlx = mx.array(b_npy)

            for c_shape in ((1,), (32, 1, 128), (1, 128)):
                c_npy = np.ones(c_shape).astype(np.float32)
                c_mlx = mx.array(c_npy)

                b_np_t = np.transpose(b_npy, (0, 2, 1))
                b_mx_t = mx.transpose(b_mlx, (0, 2, 1))

                d_npy = alpha * (a_npy @ b_np_t) + beta * c_npy
                d_mlx = mx.addmm(c_mlx, a_mlx, b_mx_t, alpha, beta)

                self.assertListEqual(list(d_npy.shape), list(d_mlx.shape))
                self.assertTrue(np.allclose(d_mlx, d_npy, atol=1e-5))
            # Batched matmul with simple broadcast
            a_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
            b_npy = np.random.normal(0.0, 1.0 / 128, (16, 16)).astype(np.float32)

            a_mlx = mx.array(a_npy)
            b_mlx = mx.array(b_npy)

            for c_shape in ((1,), (1, 16), (32, 1, 16), (1, 128, 16)):
                c_npy = np.ones(c_shape).astype(np.float32)
                c_mlx = mx.array(c_npy)

                d_npy = alpha * (a_npy @ b_npy) + beta * c_npy
                d_mlx = mx.addmm(c_mlx, a_mlx, b_mlx, alpha, beta)

                self.assertListEqual(list(d_npy.shape), list(d_mlx.shape))
                self.assertTrue(np.allclose(d_mlx, d_npy, atol=1e-5))
            # Matmul with vector
            a_npy = np.random.normal(0.0, 1.0 / 128, (16,)).astype(np.float32)
            b_npy = np.random.normal(0.0, 1.0 / 128, (32, 16, 128)).astype(np.float32)
            a_mlx = mx.array(a_npy)
            b_mlx = mx.array(b_npy)

            for c_shape in ((1,), (128,), (32, 128)):
                c_npy = np.ones(c_shape).astype(np.float32)
                c_mlx = mx.array(c_npy)

                d_npy = alpha * (a_npy @ b_npy) + beta * c_npy
                d_mlx = mx.addmm(c_mlx, a_mlx, b_mlx, alpha, beta)

                self.assertListEqual(list(d_npy.shape), list(d_mlx.shape))
                self.assertTrue(np.allclose(d_mlx, d_npy, atol=1e-5))

            # Matmul with vector
            a_npy = np.random.normal(0.0, 1.0 / 128, (32, 128, 16)).astype(np.float32)
            b_npy = np.random.normal(0.0, 1.0 / 128, (16,)).astype(np.float32)
            a_mlx = mx.array(a_npy)
            b_mlx = mx.array(b_npy)

            for c_shape in ((1,), (32, 128)):
                c_npy = np.ones(c_shape).astype(np.float32)
                c_mlx = mx.array(c_npy)

                d_npy = alpha * (a_npy @ b_npy) + beta * c_npy
                d_mlx = mx.addmm(c_mlx, a_mlx, b_mlx, alpha, beta)

                self.assertListEqual(list(d_npy.shape), list(d_mlx.shape))
                self.assertTrue(np.allclose(d_mlx, d_npy, atol=1e-5))

            # Split K specializtion
            a_npy = np.random.normal(0.0, 1.0 / 128, (64, 4096)).astype(np.float32)
            b_npy = np.random.normal(0.0, 1.0 / 128, (4096, 32)).astype(np.float32)

            a_mlx = mx.array(a_npy)
            b_mlx = mx.array(b_npy)

            for c_shape in ((1,), (1, 32), (64, 1), (64, 32)):
                c_npy = np.ones(c_shape).astype(np.float32)
                c_mlx = mx.array(c_npy)

                d_npy = alpha * (a_npy @ b_npy) + beta * c_npy
                d_mlx = mx.addmm(c_mlx, a_mlx, b_mlx, alpha, beta)

                self.assertListEqual(list(d_npy.shape), list(d_mlx.shape))
                self.assertTrue(np.allclose(d_mlx, d_npy, atol=1e-5))

            # Transposed c
            a = mx.ones((10, 5)).T
            b = mx.ones((5, 5))
            out = mx.addmm(a, b, a, beta=beta, alpha=alpha)
            expected = beta * a + alpha * (b @ a)
            self.assertTrue(mx.allclose(expected, out))

            # Broadcast c
            a = mx.ones((5, 5))
            b = mx.ones((5, 5))
            c = mx.ones((1, 5))
            out = mx.addmm(c, a, b, beta=beta, alpha=alpha)
            expected = beta * c + alpha * (a @ b)
            self.assertTrue(mx.allclose(expected, out))

    def test_addmm_grad(self):
        def make_ref_addmm(alpha, beta):
            return lambda c, a, b: alpha * (a @ b) + beta * c

        def make_addmm(alpha, beta):
            return lambda c, a, b: mx.addmm(c, a, b, alpha, beta)

        # B, M, N, K
        shapes = ((1, 64, 32, 128), (4, 28, 24, 47), (1, 1, 24, 47))

        alpha = 2.0
        for beta in (1.0, 0.5):
            f_test = make_addmm(alpha, beta)
            f_ref = make_ref_addmm(alpha, beta)

            for B, M, N, K in shapes:
                cotan = mx.ones((B, M, N))
                c = mx.random.normal((B, M, N))
                a = mx.random.normal((B, M, K))
                b = mx.random.normal((B, K, N))

                out_ref, dout_ref = mx.vjp(
                    f_ref,
                    [c, a, b],
                    [cotan],
                )
                out_test, dout_test = mx.vjp(
                    f_test,
                    [c, a, b],
                    [cotan],
                )

                self.assertTrue(mx.allclose(out_ref[0], out_test[0], atol=1e-4).item())

                for r, t in zip(dout_ref, dout_test):
                    self.assertEqual(r.shape, t.shape)
                    self.assertTrue(mx.allclose(r, t, atol=1e-4).item())

    def test_empty_matmul(self):
        a = mx.array([[], []]).T
        b = mx.array([[1.0, 2.0], [2.0, 3.0]])
        c = a @ b
        mx.eval(c)
        self.assertEqual(c.shape, (0, 2))

        a = mx.array([[1.0, 2.0], [2.0, 3.0]])
        b = mx.array([[], []])
        c = a @ b
        mx.eval(c)
        self.assertEqual(c.shape, (2, 0))

        a = mx.array([[], []]).T
        b = mx.array([[], []])
        c = a @ b
        mx.eval(c)
        self.assertEqual(c.shape, (0, 0))

        c = mx.array(1.0, dtype=mx.float32)
        a = mx.array([], dtype=mx.float32)
        b = mx.array([], dtype=mx.float32)
        out = mx.addmm(c, a, b)
        self.assertEqual(out.item(), 1.0)
        self.assertEqual(out.shape, ())

        a = mx.zeros(shape=(5, 0))
        b = mx.zeros(shape=(0, 5))
        c = mx.random.uniform(shape=(5, 5))
        out = mx.addmm(c, a, b)
        self.assertTrue(mx.allclose(out, c))

    def test_block_masked_matmul(self):
        def ref_block_masked_mm(
            a, b, block_size, out_mask=None, lhs_mask=None, rhs_mask=None
        ):
            # Get mask adjusted shapes
            M = a.shape[-2]
            N = b.shape[-1]
            K = a.shape[-1]

            bsx_shape = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])

            # Expand mask dims
            def expand_mask(mask, block_size, Y, X):
                mask = mx.expand_dims(mask, (-3, -1))
                mask_shape = list(bsx_shape) + list(mask.shape[-4:])
                mask_shape[-1] = block_size
                x = mask_shape[-2] * block_size
                mask_shape[-3] = block_size
                y = mask_shape[-4] * block_size
                mask = mx.broadcast_to(mask, mask_shape)
                mask_shape = mask_shape[:-4] + [y, x]
                return mask.reshape(mask_shape)[..., :Y, :X]

            a_masked = a
            b_masked = b

            if lhs_mask is not None:
                lhs_mask = expand_mask(lhs_mask, block_size, M, K).astype(mx.float32)
                a_masked = lhs_mask * a_masked

            if rhs_mask is not None:
                rhs_mask = expand_mask(rhs_mask, block_size, K, N).astype(mx.float32)
                b_masked = rhs_mask * b_masked

            out = a_masked @ b_masked

            if out_mask is not None:
                out_mask = expand_mask(out_mask, block_size, M, N).astype(mx.float32)
                out = out * out_mask
            return out

        def run_test(a, b, block_size, out_mask, a_mask, b_mask, cotan):
            def f_ref(a_, b_):
                return ref_block_masked_mm(a_, b_, block_size, out_mask, a_mask, b_mask)

            def f_test(a_, b_):
                return mx.block_masked_mm(a_, b_, block_size, out_mask, a_mask, b_mask)

            out_ref, dout_ref = mx.vjp(f_ref, [a, b], [cotan])
            out_test, dout_test = mx.vjp(f_test, [a, b], [cotan])

            self.assertTrue(mx.allclose(out_ref[0], out_test[0], atol=1e-5).item())

            for r, t in zip(dout_ref, dout_test):
                self.assertEqual(r.shape, t.shape)
                self.assertTrue(mx.allclose(r, t, atol=1e-4).item())

        def run_test_mask_vjp(a, b, block_size, out_mask, a_mask, b_mask, cotan):
            def f_ref(a_, b_, a_mask_, b_mask_):
                return ref_block_masked_mm(
                    a_, b_, block_size, out_mask, a_mask_, b_mask_
                )

            def f_test(a_, b_, a_mask_, b_mask_):
                return mx.block_masked_mm(
                    a_, b_, block_size, out_mask, a_mask_, b_mask_
                )

            out_ref, dout_ref = mx.vjp(f_ref, [a, b, a_mask, b_mask], [cotan])
            out_test, dout_test = mx.vjp(f_test, [a, b, a_mask, b_mask], [cotan])

            mx.eval((out_ref, dout_ref, out_test, dout_test))

            self.assertTrue(mx.allclose(out_ref[0], out_test[0], atol=1e-5).item())

            for r, t in zip(dout_ref, dout_test):
                self.assertEqual(r.shape, t.shape)
                self.assertTrue(mx.allclose(r, t, atol=1e-4).item())

        def make_mask(tm_, tn_, batch, np_dtype):
            arr_np_mask = np.random.normal(size=batch + (tm_, tn_)).astype(np_dtype)
            arr_np_bool_mask = arr_np_mask < 0.0
            arr_np_mask[arr_np_bool_mask] = 0.0

            return mx.array(arr_np_bool_mask), mx.array(arr_np_mask)

        def test_shape(
            M,
            N,
            K,
            block_size,
            transpose=False,
            np_dtype=np.float32,
            batch_A=(),
            batch_B=(),
        ):
            with self.subTest(
                M=M,
                N=N,
                K=K,
                block_size=block_size,
                np_dtype=np_dtype,
                transpose=transpose,
                batch_A=batch_A,
                batch_B=batch_B,
            ):
                batch_out = np.broadcast_shapes(batch_A, batch_B)
                cotan = mx.ones(batch_out + (M, N))

                a_np = np.random.normal(size=batch_A + (M, K)).astype(np_dtype)
                b_np = np.random.normal(size=batch_B + (K, N)).astype(np_dtype)

                a_mx = mx.array(a_np)
                b_mx = mx.array(b_np)

                tm = (M + block_size - 1) // block_size
                tn = (N + block_size - 1) // block_size
                tk = (K + block_size - 1) // block_size

                a_mx_bool_mask, a_mx_mask = make_mask(tm, tk, batch_A, np_dtype)
                b_mx_bool_mask, b_mx_mask = make_mask(tk, tn, batch_B, np_dtype)
                out_mx_bool_mask, out_mx_mask = make_mask(tm, tn, batch_out, np_dtype)

                # Boolean block masks
                run_test(
                    a_mx,
                    b_mx,
                    block_size,
                    out_mx_bool_mask,
                    a_mx_bool_mask,
                    b_mx_bool_mask,
                    cotan,
                )
                run_test(a_mx, b_mx, block_size, out_mx_bool_mask, None, None, cotan)
                run_test(
                    a_mx, b_mx, block_size, None, a_mx_bool_mask, b_mx_bool_mask, cotan
                )

                # Float block masks
                run_test(
                    a_mx, b_mx, block_size, out_mx_mask, a_mx_mask, b_mx_mask, cotan
                )
                run_test(a_mx, b_mx, block_size, None, a_mx_mask, b_mx_mask, cotan)
                run_test_mask_vjp(
                    a_mx, b_mx, block_size, out_mx_mask, a_mx_mask, b_mx_mask, cotan
                )
                run_test_mask_vjp(
                    a_mx, b_mx, block_size, None, a_mx_mask, b_mx_mask, cotan
                )

        shapes = (
            (16, 16, 16, 32),
            (64, 64, 16, 32),
            (128, 128, 128, 32),
            (256, 256, 128, 64),
            (1, 128, 128, 32),
            (256, 1, 128, 64),
        )

        for M, N, K, block_size in shapes:
            test_shape(M, N, K, block_size)

        # Test broadcasting
        test_shape(64, 64, 64, 32, batch_A=(1, 2), batch_B=(2, 2))
        test_shape(1, 128, 128, 32, batch_A=(1, 2), batch_B=(2, 2))
        test_shape(128, 1, 128, 32, batch_A=(1, 2), batch_B=(2, 2))

        a_np = np.ones((128, 256)).astype(np.float32)
        b_np = np.ones((128, 1)).astype(np.float32)
        d_np = np.ones((1, 256)).astype(np.float32)
        a_mask_np = np.random.normal(size=(4, 8)).astype(np.float32)
        b_mask_np = np.ones((4, 1)).astype(np.bool_)
        d_mask_np = np.ones((1, 8)).astype(np.bool_)
        c_mask_np = np.random.normal(size=(8, 1)).astype(np.float32)
        e_mask_np = np.random.normal(size=(1, 4)).astype(np.float32)

        a_mask_np[a_mask_np < 0.0] = 0.0
        e_mask_np[e_mask_np < 0.0] = 0.0
        c_mask_np[c_mask_np < 0.0] = 0.0

        a_mx = mx.array(a_np)
        b_mx = mx.array(b_np)
        d_mx = mx.array(d_np)
        a_mask_mx = mx.array(a_mask_np)
        b_mask_mx = mx.array(b_mask_np)
        d_mask_mx = mx.array(d_mask_np)
        e_mask_mx = mx.array(e_mask_np)
        c_mask_mx = mx.array(c_mask_np)

        c_mx = mx.block_masked_mm(a_mx.T, b_mx, 32, c_mask_mx, a_mask_mx.T, b_mask_mx)
        e_mx = mx.block_masked_mm(d_mx, a_mx.T, 32, e_mask_mx, d_mask_mx, a_mask_mx.T)

        a_mask_np = np.broadcast_to(np.expand_dims(a_mask_np, (-3, -1)), (4, 32, 8, 32))
        a_mask_np = a_mask_np.reshape((128, 256))
        a_np *= a_mask_np

        c_np = a_np.T @ b_np
        e_np = d_np @ a_np.T

        c_mask_np = np.broadcast_to(np.expand_dims(c_mask_np, (-2)), (8, 32, 1))
        c_mask_np = c_mask_np.reshape((256, 1))
        c_np *= c_mask_np

        e_mask_np = np.broadcast_to(np.expand_dims(e_mask_np, (-1)), (1, 4, 32))
        e_mask_np = e_mask_np.reshape((1, 128))
        e_np *= e_mask_np

        self.assertTrue(np.allclose(c_mx, c_np, atol=1e-5))
        self.assertTrue(np.allclose(e_mx, e_np, atol=1e-5))

    def test_gather_matmul(self):
        def np_gather_mm(a, b, lhs_indices=None, rhs_indices=None):
            a = a.reshape((-1, a.shape[-2], a.shape[-1]))
            b = b.reshape((-1, b.shape[-2], b.shape[-1]))
            lhs_indices = lhs_indices or np.arange(a.shape[0])
            rhs_indices = rhs_indices or np.arange(b.shape[0])
            a = a[lhs_indices, :, :]
            b = b[rhs_indices, :, :]
            out = a @ b
            return out

        def test_shape(
            M,
            N,
            K,
            np_dtype=np.float32,
            batch_A=(),
            batch_B=(),
            lhs_indices=None,
            rhs_indices=None,
        ):
            with self.subTest(
                M=M,
                N=N,
                K=K,
                np_dtype=np_dtype,
                batch_A=batch_A,
                batch_B=batch_B,
                lhs_indices=lhs_indices,
                rhs_indices=rhs_indices,
            ):
                a_np = np.random.normal(size=batch_A + (M, K)).astype(np_dtype)
                b_np = np.random.normal(size=batch_B + (K, N)).astype(np_dtype)

                a_mx = mx.array(a_np)
                b_mx = mx.array(b_np)

                out_np = np_gather_mm(a_np, b_np, lhs_indices, rhs_indices)

                lhs_indices_mx = None if lhs_indices is None else mx.array(lhs_indices)
                rhs_indices_mx = None if rhs_indices is None else mx.array(rhs_indices)

                out_mx = mx.gather_mm(a_mx, b_mx, lhs_indices_mx, rhs_indices_mx)

                self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5))

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
            test_shape(32, 32, 32, **kwargs)
            test_shape(16, 1, 16, **kwargs)

        # Add tests for broadcasting
        a_np = np.random.normal(size=(5, 32, 32)).astype(np.float32)
        b_np = np.random.normal(size=(3, 32, 32)).astype(np.float32)
        a_mx = mx.array(a_np)
        b_mx = mx.array(b_np)

        # Numpy
        a_np = a_np.reshape((5, 1, 32, 32))
        b_np = b_np.reshape((1, 3, 32, 32))

        a_np = np.broadcast_to(a_np, (5, 4, 32, 32))
        b_np = np.broadcast_to(b_np, (2, 3, 32, 32)).swapaxes(1, 0)

        lhs_indices = [0, 13, 12]
        rhs_indices = [0, 3, 5]

        out_np = np_gather_mm(a_np, b_np, lhs_indices, rhs_indices)

        # MLX
        a_mx = a_mx.reshape((5, 1, 32, 32))
        b_mx = b_mx.reshape((1, 3, 32, 32))

        a_mx = mx.broadcast_to(a_mx, (5, 4, 32, 32))
        b_mx = mx.broadcast_to(b_mx, (2, 3, 32, 32)).swapaxes(1, 0)

        lhs_indices_mx = mx.array(lhs_indices)
        rhs_indices_mx = mx.array(rhs_indices)

        out_mx = mx.gather_mm(a_mx, b_mx, lhs_indices_mx, rhs_indices_mx)

        self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5))

        # Gemv test
        a_np = np.random.normal(size=(5, 1, 32)).astype(np.float32)
        b_np = np.random.normal(size=(3, 16, 32)).astype(np.float32)
        a_mx = mx.array(a_np)
        b_mx = mx.array(b_np)

        lhs_indices = [3, 1]
        rhs_indices = [0, 2]

        b_np_t = np.swapaxes(b_np, -1, -2)
        out_np = np_gather_mm(a_np, b_np_t, lhs_indices, rhs_indices)

        lhs_indices_mx = mx.array(lhs_indices)
        rhs_indices_mx = mx.array(rhs_indices)

        b_mx_t = mx.swapaxes(b_mx, -1, -2)
        out_mx = mx.gather_mm(a_mx, b_mx_t, lhs_indices_mx, rhs_indices_mx)

        self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5))

    def test_gather_matmul_grad(self):
        lhs_indices = mx.array([[7, 6], [4, 1], [0, 2]], dtype=mx.uint32)
        rhs_indices = mx.array([[2], [0], [1]], dtype=mx.uint32)

        def f_ref(a, b):
            lhs_indices_ = mx.broadcast_to(lhs_indices, (3, 2))
            rhs_indices_ = mx.broadcast_to(rhs_indices, (3, 2))
            M = a.shape[-2]
            N = b.shape[-1]
            K = a.shape[-1]

            a = a.reshape((-1, M, K))
            b = b.reshape((-1, K, N))

            a = mx.take(a, lhs_indices_, 0)
            b = mx.take(b, rhs_indices_, 0)

            return a @ b

        def f_test(a, b):
            return mx.gather_mm(a, b, lhs_indices, rhs_indices)

        a_mx = mx.random.normal((4, 2, 32, 32))
        b_mx = mx.random.normal((4, 1, 32, 32))

        out_test = f_test(a_mx, b_mx)
        out_ref = f_ref(a_mx, b_mx)

        self.assertTrue(mx.allclose(out_test, out_ref, atol=1e-5))

        cotan = mx.ones_like(out_test)
        out_ref, dout_ref = mx.vjp(
            f_ref,
            [a_mx, b_mx],
            [cotan],
        )
        out_test, dout_test = mx.vjp(
            f_test,
            [a_mx, b_mx],
            [cotan],
        )

        for r, t in zip(dout_ref, dout_test):
            self.assertEqual(r.shape, t.shape)
            self.assertTrue(mx.allclose(r, t, atol=1e-4).item())

    def test_gather_mm_sorted(self):
        def gather_mm_ref(a, b, rhs):
            b = b[rhs]
            return a @ b

        def gather_mm_test(a, b, rhs):
            return mx.gather_mm(a, b, rhs_indices=rhs, sorted_indices=True)

        a = mx.random.normal((100, 1, 100))
        b = mx.random.normal((8, 100, 100))
        rhs = mx.sort(mx.random.randint(0, 8, shape=(100,)))

        c1 = gather_mm_ref(a, b, rhs)
        c2 = gather_mm_test(a, b, rhs)
        self.assertTrue(mx.allclose(c1, c2, atol=1e-4))

        cotan = mx.random.normal(c1.shape)
        c1, dc1 = mx.vjp(
            lambda a, b: gather_mm_ref(a, b, rhs),
            [a, b],
            [cotan],
        )
        c2, dc2 = mx.vjp(
            lambda a, b: gather_mm_test(a, b, rhs),
            [a, b],
            [cotan],
        )
        self.assertTrue(mx.allclose(c1[0], c2[0], atol=1e-4))
        self.assertTrue(mx.allclose(dc1[0], dc2[0], atol=1e-4))
        self.assertTrue(mx.allclose(dc1[1], dc2[1], atol=1e-4))

    def test_segmented_mm(self):
        def segmented_mm_ref(a, b, s):
            s = s.tolist()
            c = []
            for s1, s2 in s:
                c.append(a[:, s1:s2] @ b[s1:s2, :])
            return mx.stack(c, axis=0)

        shapes = [
            (10, 10, 10),
            (10, 10, 1000),
            (1000, 1000, 1000),
        ]
        all_segments = [[0, 0, 1.0], [0, 0.5, 1.0], [r / 9 for r in range(10)]]

        for M, N, K in shapes:
            for s in all_segments:
                segments = []
                for i in range(len(s) - 1):
                    segments.append([s[i], s[i + 1]])
                segments = mx.array(segments)
                segments = mx.minimum(K - 1, (K * segments).astype(mx.uint32))
                a = mx.random.normal((M, K))
                b = mx.random.normal((K, N))
                c1 = segmented_mm_ref(a, b, segments)
                c2 = mx.segmented_mm(a, b, segments)
                self.assertTrue(mx.allclose(c1, c2, atol=1e-4))

                a = mx.random.normal((K, M))
                b = mx.random.normal((K, N))
                c1 = segmented_mm_ref(a.T, b, segments)
                c2 = mx.segmented_mm(a.T, b, segments)
                self.assertTrue(mx.allclose(c1, c2, atol=1e-4))

                a = mx.random.normal((M, K))
                b = mx.random.normal((N, K))
                c1 = segmented_mm_ref(a, b.T, segments)
                c2 = mx.segmented_mm(a, b.T, segments)
                self.assertTrue(mx.allclose(c1, c2, atol=1e-4))

                a = mx.random.normal((K, M))
                b = mx.random.normal((N, K))
                c1 = segmented_mm_ref(a.T, b.T, segments)
                c2 = mx.segmented_mm(a.T, b.T, segments)
                self.assertTrue(mx.allclose(c1, c2, atol=1e-4))

        with self.assertRaises(ValueError):
            a = mx.ones((2, 10, 10))
            s = mx.array([[0, 5], [5, 10]]).astype(mx.uint32)
            mx.segmented_mm(a, a, s)

        a = mx.ones((10, 1000))
        s = mx.random.randint(0, 16, shape=(1000,))
        s = mx.zeros(16, dtype=s.dtype).at[s].add(1)
        s = mx.sort(s)
        s = mx.cumsum(s)
        s = mx.concatenate([mx.array([0]), s])
        s = mx.as_strided(s, (16, 2), (1, 1))
        s = mx.reshape(s, (2, 2, 4, 2))
        c = mx.segmented_mm(a, a.T, s)
        self.assertEqual(c.shape, (2, 2, 4, 10, 10))

    def test_gemv_gemm_same_precision(self):
        mx.random.seed(0)
        N = 256
        if mx.is_available(mx.gpu):
            t = mx.bfloat16
            a = mx.random.normal([1, N]).astype(t)
            b = mx.concatenate([a, a], axis=0).astype(t)
            c = mx.random.normal([N, 64]).astype(t)
            out_gemv = a @ c
            out_gemm = (b @ c)[0]
            self.assertTrue(mx.allclose(out_gemv, out_gemm))

    def test_complex_gemv(self):
        M = 16
        N = 50

        def rand(shape):
            return mx.random.uniform(shape=shape) + 1j * mx.random.uniform(shape=shape)

        a = rand((M, N))
        b = rand((N, 1))
        c = mx.matmul(a, b)
        c_np = np.matmul(a, b)
        self.assertTrue(np.allclose(c, c_np))

        # Transposed
        a = rand((N, M))
        b = rand((N, 1))
        c = mx.matmul(a.T, b)
        c_np = np.matmul(np.array(a).T, b)
        self.assertTrue(np.allclose(c, c_np))

        # Check shapes
        a = mx.random.normal((2, 3)).astype(mx.complex64)
        b = mx.random.normal((3,))
        self.assertEqual((a @ b).shape, (2,))

        a = mx.random.normal((2, 3)).astype(mx.complex64)
        b = mx.random.normal((3,))
        c = mx.random.normal((2,))
        self.assertEqual(mx.addmm(c, a, b).shape, (2,))

    def test_complex_gemm(self):
        M = 16
        K = 50
        N = 32

        def rand(shape):
            return mx.random.uniform(shape=shape) + 1j * mx.random.uniform(shape=shape)

        a = rand((M, K))
        b = rand((K, N))
        c = mx.matmul(a, b)
        c_np = np.matmul(a, b)
        self.assertTrue(np.allclose(c, c_np))

        # Test addmm
        a = rand((M, K))
        b = rand((K, N))
        c = rand((M, N))
        out = mx.addmm(c, a, b, 2.0, 2.0)
        out_np = 2.0 * np.matmul(a, b) + 2.0 * c
        self.assertTrue(np.allclose(out, out_np))

        # complex with real
        a = rand((M, K)).real
        b = rand((K, N))
        c = mx.matmul(a, b)
        c_np = np.matmul(a, b)
        self.assertTrue(np.allclose(out, out_np))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
