# Copyright © 2023 Apple Inc.

import unittest
from itertools import combinations, permutations

import mlx.core as mx
import mlx_tests
import numpy as np


class TestReduce(mlx_tests.MLXTestCase):
    def test_axis_permutation_sums(self):
        x_npy = np.random.randn(5, 5, 5, 5, 5).astype(np.float32)
        x_mlx = mx.array(x_npy)
        for t in permutations(range(5)):
            with self.subTest(t=t):
                y_npy = np.transpose(x_npy, t)
                y_mlx = mx.transpose(x_mlx, t)
                for n in range(1, 6):
                    for a in combinations(range(5), n):
                        with self.subTest(a=a):
                            z_npy = np.sum(y_npy, axis=a)
                            z_mlx = mx.sum(y_mlx, axis=a)
                            mx.eval(z_mlx)
                            self.assertTrue(
                                np.allclose(z_npy, np.array(z_mlx), atol=1e-4)
                            )

    def test_expand_sums(self):
        x_npy = np.random.randn(5, 1, 5, 1, 5, 1).astype(np.float32)
        x_mlx = mx.array(x_npy)
        for m in range(1, 4):
            for ax in combinations([1, 3, 5], m):
                shape = np.array([5, 1, 5, 1, 5, 1])
                shape[list(ax)] = 5
                shape = shape.tolist()
                with self.subTest(shape=shape):
                    y_npy = np.broadcast_to(x_npy, shape)
                    y_mlx = mx.broadcast_to(x_mlx, shape)
                    for n in range(1, 7):
                        for a in combinations(range(6), n):
                            with self.subTest(a=a):
                                z_npy = np.sum(y_npy, axis=a) / 1000
                                z_mlx = mx.sum(y_mlx, axis=a) / 1000
                                mx.eval(z_mlx)
                                self.assertTrue(
                                    np.allclose(z_npy, np.array(z_mlx), atol=1e-4)
                                )

    def test_dtypes(self):
        int_dtypes = [
            "int8",
            "int16",
            "int32",
            "uint8",
            "uint16",
            "uint32",
        ]
        float_dtypes = ["float32"]

        for dtype in int_dtypes + float_dtypes:
            with self.subTest(dtype=dtype):
                x = np.random.uniform(0, 2, size=(3, 3, 3)).astype(getattr(np, dtype))
                y = mx.array(x)

                for op in ("sum", "prod", "min", "max"):
                    with self.subTest(op=op):

                        np_op = getattr(np, op)
                        mlx_op = getattr(mx, op)

                        for axes in (None, 0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)):
                            with self.subTest(axes=axes):
                                if op in ("sum", "prod"):
                                    r_np = np_op(
                                        x, axis=axes, dtype=(getattr(np, dtype))
                                    )
                                else:
                                    r_np = np_op(x, axis=axes)
                                r_mlx = mlx_op(y, axis=axes)
                                mx.eval(r_mlx)
                                self.assertTrue(np.allclose(r_np, r_mlx, atol=1e-4))

    def test_arg_reduce(self):
        dtypes = [
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
        ]
        for dtype in dtypes:
            with self.subTest(dtype=dtype):

                data = np.random.rand(10, 12, 13).astype(getattr(np, dtype))
                x = mx.array(data)
                for op in ["argmin", "argmax"]:
                    for axis in range(3):
                        for kd in [True, False]:
                            a = getattr(mx, op)(x, axis, kd)
                            b = getattr(np, op)(data, axis, keepdims=kd)
                            self.assertEqual(a.tolist(), b.tolist())

                for op in ["argmin", "argmax"]:
                    a = getattr(mx, op)(x, keepdims=True)
                    b = getattr(np, op)(data, keepdims=True)
                    self.assertEqual(a.tolist(), b.tolist())
                    a = getattr(mx, op)(x)
                    b = getattr(np, op)(data)
                    self.assertEqual(a.item(), b)


if __name__ == "__main__":
    unittest.main(failfast=True)
