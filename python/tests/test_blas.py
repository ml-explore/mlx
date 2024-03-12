# Copyright © 2023 Apple Inc.

import math
import unittest
from itertools import permutations

import mlx.core as mx
import mlx_tests
import numpy as np


class TestBlas(mlx_tests.MLXTestCase):
    @property
    def dtypes(self):
        return ["float32", "float16"] if mx.metal.is_available() else ["float32"]

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
        if not mx.metal.is_available():
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

    def test_matmul_shapes(self):
        if not mx.metal.is_available():
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
        with self.subTest(shape=shape_mat):
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

    def test_addmm(self):
        np.random.seed(0)
        # Batched matmul
        alpha = 0.5
        beta = 2.0

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

        # # Batched matmul with simple broadcast
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

    def test_addmm_grad(self):
        def make_ref_addmm(alpha, beta):
            return lambda c, a, b: alpha * (a @ b) + beta * c

        def make_addmm(alpha, beta):
            return lambda c, a, b: mx.addmm(c, a, b, alpha, beta)

        # B, M, N, K
        shapes = ((1, 64, 32, 128), (4, 28, 24, 47), (1, 1, 24, 47))

        alpha = 2.0
        beta = 0.5

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


if __name__ == "__main__":
    unittest.main()
