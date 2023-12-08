# Copyright © 2023 Apple Inc.

import math
import unittest
from itertools import permutations

import mlx.core as mx
import mlx_tests
import numpy as np

try:
    import torch

    has_torch = True
except ImportError as e:
    has_torch = False


class TestBF16(mlx_tests.MLXTestCase):
    def __test_ops(
        self,
        ref_op,  # Function that outputs array_like
        mlx_op,  # Function that outputs array_like
        np_args,  # Numpy arguments
        ref_transform=lambda x: x,
        mlx_transform=lambda x: mx.array(x),
        atol=1e-5,
    ):
        ref_args = map(ref_transform, np_args)
        mlx_args = map(mlx_transform, np_args)

        r_ref = ref_op(*ref_args)
        r_mlx = mlx_op(*mlx_args)

        self.assertTrue(np.allclose(r_mlx, r_ref, atol=atol))

    def __default_test(
        self,
        op,
        np_args,
        simple_transform=lambda x: x,
        atol_np=1e-3,
        atol_torch=1e-5,
        np_kwargs=dict(),
        mlx_kwargs=dict(),
        torch_kwargs=dict(),
        torch_op=None,
    ):
        with self.subTest(reference="numpy"):

            def np_transform(x):
                x_mx_bf16 = mx.array(x).astype(mx.bfloat16)
                x_mx_fp32 = x_mx_bf16.astype(mx.float32)
                return np.asarray(x_mx_fp32)

            def mlx_fn(*args):
                out_bf16 = getattr(mx, op)(*args, **mlx_kwargs)
                return np.asarray(out_bf16.astype(mx.float32))

            def np_fn(*args):
                out_fp32 = getattr(np, op)(*args, **np_kwargs)
                return np_transform(out_fp32)

            ref_op = np_fn
            mlx_op = mlx_fn

            ref_transform = lambda x: simple_transform(np_transform(x))
            mlx_transform = lambda x: simple_transform(mx.array(x).astype(mx.bfloat16))

            self.__test_ops(
                ref_op,
                mlx_op,
                np_args,
                ref_transform=ref_transform,
                mlx_transform=mlx_transform,
                atol=atol_np,
            )

        if has_torch:
            with self.subTest(reference="torch"):
                torch_op = op if torch_op is None else torch_op

                def torch_fn(*args):
                    out_bf16 = getattr(torch, torch_op)(*args, **torch_kwargs)
                    return out_bf16.to(torch.float32).numpy()

                ref_op = torch_fn
                ref_transform = lambda x: simple_transform(
                    torch.from_numpy(x).to(torch.bfloat16)
                )
                self.__test_ops(
                    ref_op,
                    mlx_op,
                    np_args,
                    ref_transform=ref_transform,
                    mlx_transform=mlx_transform,
                    atol=atol_torch,
                )

    def test_unary_ops(self):
        x = np.random.rand(18, 28, 38)
        for op in ["abs", "exp", "log", "square", "sqrt"]:
            with self.subTest(op=op):
                np_args = (x.astype(np.float32),)
                self.__default_test(op, np_args)

    def test_binary_ops(self):
        x = np.random.rand(18, 28, 38)
        y = np.random.rand(18, 28, 38)
        for op in ["add", "subtract", "multiply", "divide", "maximum", "minimum"]:
            with self.subTest(op=op):
                np_args = (
                    x.astype(np.float32),
                    y.astype(np.float32),
                )
                self.__default_test(op, np_args, simple_transform=lambda x: x)
                self.__default_test(op, np_args, simple_transform=lambda x: x[:1])
                self.__default_test(op, np_args, simple_transform=lambda x: x[:, :1])

    def test_reduction_ops(self):
        x = np.random.rand(18, 28, 38).astype(np.float32)

        for op in ("min", "max"):
            with self.subTest(op=op):

                for axes in (0, 1, 2, (0, 1), (0, 2), (1, 2), (0, 1, 2)):
                    with self.subTest(axes=axes):
                        np_args = (x.astype(np.float32),)
                        self.__default_test(
                            op,
                            np_args,
                            np_kwargs={"axis": axes},
                            mlx_kwargs={"axis": axes},
                            torch_kwargs={"dim": axes},
                            torch_op="a" + op,
                        )

    def test_arg_reduction_ops(self):
        data = np.random.rand(10, 12, 13).astype(np.float32)
        x = mx.array(data).astype(mx.bfloat16)
        data = np.asarray(x.astype(mx.float32))

        for op in ["argmin", "argmax"]:
            for axis in range(3):
                for kd in [True, False]:
                    a = getattr(mx, op)(x, axis, kd)
                    b = getattr(np, op)(data, axis, keepdims=kd)
                    a = a.astype(mx.float32)
                    self.assertEqual(a.tolist(), b.tolist())

        for op in ["argmin", "argmax"]:
            a = getattr(mx, op)(x, keepdims=True)
            b = getattr(np, op)(data, keepdims=True)
            a = a.astype(mx.float32)
            self.assertEqual(a.tolist(), b.tolist())
            a = getattr(mx, op)(x)
            b = getattr(np, op)(data)
            a = a.astype(mx.float32)
            self.assertEqual(a.item(), b)

    def test_blas_ops(self):
        if mx.default_device() != mx.gpu:
            return

        def test_blas(shape_x, shape_y):
            np.random.seed(42)
            with self.subTest(shape_x=shape_x, shape_y=shape_y):
                x = np.random.normal(0.0, 1.0 / shape_x[-1], size=shape_x)
                y = np.random.normal(0.0, 1.0 / shape_x[-1], size=shape_y)

                np_args = (
                    x.astype(np.float32),
                    y.astype(np.float32),
                )
                op = "matmul"

                self.__default_test(op, np_args, atol_np=1e-3, atol_torch=1e-3)

        for shape_x, shape_y in [
            [(32, 32), (32, 32)],
            [(23, 57), (57, 1)],
            [(1, 3), (3, 128)],
            [(8, 128, 768), (768, 16)],
        ]:
            test_blas(shape_x, shape_y)


if __name__ == "__main__":
    unittest.main()
