# Copyright © 2023 Apple Inc.

import operator
import unittest
from itertools import permutations

import mlx.core as mx
import mlx_tests
import numpy as np


class TestVersion(mlx_tests.MLXTestCase):
    def test_version(self):
        v = mx.__version__
        vnums = v.split(".")
        self.assertGreaterEqual(len(vnums), 3)
        v = ".".join(str(int(vn)) for vn in vnums[:3])
        self.assertEqual(v, mx.__version__[: len(v)])


class TestDtypes(mlx_tests.MLXTestCase):
    def test_dtypes(self):
        self.assertEqual(mx.bool_.size, 1)
        self.assertEqual(mx.uint8.size, 1)
        self.assertEqual(mx.uint16.size, 2)
        self.assertEqual(mx.uint32.size, 4)
        self.assertEqual(mx.uint64.size, 8)
        self.assertEqual(mx.int8.size, 1)
        self.assertEqual(mx.int16.size, 2)
        self.assertEqual(mx.int32.size, 4)
        self.assertEqual(mx.int64.size, 8)
        self.assertEqual(mx.float16.size, 2)
        self.assertEqual(mx.float32.size, 4)
        self.assertEqual(mx.bfloat16.size, 2)
        self.assertEqual(mx.complex64.size, 8)

        self.assertEqual(str(mx.bool_), "mlx.core.bool")
        self.assertEqual(str(mx.uint8), "mlx.core.uint8")
        self.assertEqual(str(mx.uint16), "mlx.core.uint16")
        self.assertEqual(str(mx.uint32), "mlx.core.uint32")
        self.assertEqual(str(mx.uint64), "mlx.core.uint64")
        self.assertEqual(str(mx.int8), "mlx.core.int8")
        self.assertEqual(str(mx.int16), "mlx.core.int16")
        self.assertEqual(str(mx.int32), "mlx.core.int32")
        self.assertEqual(str(mx.int64), "mlx.core.int64")
        self.assertEqual(str(mx.float16), "mlx.core.float16")
        self.assertEqual(str(mx.float32), "mlx.core.float32")
        self.assertEqual(str(mx.bfloat16), "mlx.core.bfloat16")
        self.assertEqual(str(mx.complex64), "mlx.core.complex64")

    def test_scalar_conversion(self):
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
            "complex64",
        ]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                x = np.array(2, dtype=getattr(np, dtype))
                y = np.min(x)

                self.assertEqual(x.dtype, y.dtype)
                self.assertTupleEqual(x.shape, y.shape)

                z = mx.array(y)
                self.assertEqual(np.array(z), x)
                self.assertEqual(np.array(z), y)
                self.assertEqual(z.dtype, getattr(mx, dtype))
                self.assertListEqual(list(z.shape), list(x.shape))
                self.assertListEqual(list(z.shape), list(y.shape))


class TestArray(mlx_tests.MLXTestCase):
    def test_array_basics(self):
        x = mx.array(1)
        self.assertEqual(x.size, 1)
        self.assertEqual(x.ndim, 0)
        self.assertEqual(x.shape, [])
        self.assertEqual(x.dtype, mx.int32)
        self.assertEqual(x.item(), 1)
        self.assertTrue(isinstance(x.item(), int))

        with self.assertRaises(TypeError):
            len(x)

        x = mx.array(1, mx.uint32)
        self.assertEqual(x.item(), 1)
        self.assertTrue(isinstance(x.item(), int))

        x = mx.array(1, mx.int64)
        self.assertEqual(x.item(), 1)
        self.assertTrue(isinstance(x.item(), int))

        x = mx.array(1.0)
        self.assertEqual(x.size, 1)
        self.assertEqual(x.ndim, 0)
        self.assertEqual(x.shape, [])
        self.assertEqual(x.dtype, mx.float32)
        self.assertEqual(x.item(), 1.0)
        self.assertTrue(isinstance(x.item(), float))

        x = mx.array(False)
        self.assertEqual(x.size, 1)
        self.assertEqual(x.ndim, 0)
        self.assertEqual(x.shape, [])
        self.assertEqual(x.dtype, mx.bool_)
        self.assertEqual(x.item(), False)
        self.assertTrue(isinstance(x.item(), bool))

        x = mx.array(complex(1, 1))
        self.assertEqual(x.ndim, 0)
        self.assertEqual(x.shape, [])
        self.assertEqual(x.dtype, mx.complex64)
        self.assertEqual(x.item(), complex(1, 1))
        self.assertTrue(isinstance(x.item(), complex))

        x = mx.array([True, False, True])
        self.assertEqual(x.dtype, mx.bool_)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(x.shape, [3])
        self.assertEqual(len(x), 3)

        x = mx.array([True, False, True], mx.float32)
        self.assertEqual(x.dtype, mx.float32)

        x = mx.array([0, 1, 2])
        self.assertEqual(x.dtype, mx.int32)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(x.shape, [3])

        x = mx.array([0, 1, 2], mx.float32)
        self.assertEqual(x.dtype, mx.float32)

        x = mx.array([0.0, 1.0, 2.0])
        self.assertEqual(x.dtype, mx.float32)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(x.shape, [3])

        x = mx.array([1j, 1 + 0j])
        self.assertEqual(x.dtype, mx.complex64)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(x.shape, [2])

        # From tuple
        x = mx.array((1, 2, 3), mx.int32)
        self.assertEqual(x.dtype, mx.int32)
        self.assertEqual(x.tolist(), [1, 2, 3])

    def test_bool_conversion(self):
        x = mx.array(True)
        self.assertTrue(x)
        x = mx.array(False)
        self.assertFalse(x)
        x = mx.array(1.0)
        self.assertTrue(x)
        x = mx.array(0.0)
        self.assertFalse(x)

    def test_construction_from_lists(self):
        x = mx.array([])
        self.assertEqual(x.size, 0)
        self.assertEqual(x.shape, [0])
        self.assertEqual(x.dtype, mx.float32)

        x = mx.array([[], [], []])
        self.assertEqual(x.size, 0)
        self.assertEqual(x.shape, [3, 0])
        self.assertEqual(x.dtype, mx.float32)

        x = mx.array([[[], []], [[], []], [[], []]])
        self.assertEqual(x.size, 0)
        self.assertEqual(x.shape, [3, 2, 0])
        self.assertEqual(x.dtype, mx.float32)

        # Check failure cases
        with self.assertRaises(ValueError):
            x = mx.array([[[], []], [[]], [[], []]])

        with self.assertRaises(ValueError):
            x = mx.array([[[], []], [[1.0, 2.0], []], [[], []]])

        with self.assertRaises(ValueError):
            x = mx.array([[0, 1], [[0, 1], 1]])

        with self.assertRaises(ValueError):
            x = mx.array([[0, 1], ["hello", 1]])

        x = mx.array([True, False, 3])
        self.assertEqual(x.dtype, mx.int32)

        x = mx.array([True, False, 3, 4.0])
        self.assertEqual(x.dtype, mx.float32)

        x = mx.array([[True, False], [1, 3], [2, 4.0]])
        self.assertEqual(x.dtype, mx.float32)

        x = mx.array([[1.0, 2.0], [0.0, 3.9]], mx.bool_)
        self.assertEqual(x.dtype, mx.bool_)
        self.assertTrue(mx.array_equal(x, mx.array([[True, True], [False, True]])))

        x = mx.array([[1.0, 2.0], [0.0, 3.9]], mx.int32)
        self.assertTrue(mx.array_equal(x, mx.array([[1, 2], [0, 3]])))

        x = mx.array([1 + 0j, 2j, True, 0], mx.complex64)
        self.assertEqual(x.tolist(), [1 + 0j, 2j, 1 + 0j, 0j])

    def test_init_from_array(self):
        x = mx.array(3.0)
        y = mx.array(x)

        self.assertTrue(mx.array_equal(x, y))

        y = mx.array(x, mx.int32)
        self.assertEqual(y.dtype, mx.int32)
        self.assertEqual(y.item(), 3)

        y = mx.array(x, mx.bool_)
        self.assertEqual(y.dtype, mx.bool_)
        self.assertEqual(y.item(), True)

        # y = mx.array(x, mx.complex64)
        # self.assertEqual(y.dtype, mx.complex64)
        # self.assertEqual(y.item(), 3.0+0j)

    def test_array_repr(self):
        x = mx.array(True)
        self.assertEqual(str(x), "array(true, dtype=bool)")
        x = mx.array(1)
        self.assertEqual(str(x), "array(1, dtype=int32)")
        x = mx.array(1.0)
        self.assertEqual(str(x), "array(1, dtype=float32)")

        x = mx.array([1, 0, 1])
        self.assertEqual(str(x), "array([1, 0, 1], dtype=int32)")

        x = mx.array([1] * 6)
        expected = "array([1, 1, 1, 1, 1, 1], dtype=int32)"
        self.assertEqual(str(x), expected)

        x = mx.array([1] * 7)
        expected = "array([1, 1, 1, ..., 1, 1, 1], dtype=int32)"
        self.assertEqual(str(x), expected)

        x = mx.array([[1, 2], [1, 2], [1, 2]])
        expected = "array([[1, 2],\n" "       [1, 2],\n" "       [1, 2]], dtype=int32)"
        self.assertEqual(str(x), expected)

        x = mx.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]])
        expected = (
            "array([[[1, 2],\n"
            "        [1, 2]],\n"
            "       [[1, 2],\n"
            "        [1, 2]]], dtype=int32)"
        )
        self.assertEqual(str(x), expected)

        x = mx.array([[1, 2]] * 6)
        expected = (
            "array([[1, 2],\n"
            "       [1, 2],\n"
            "       [1, 2],\n"
            "       [1, 2],\n"
            "       [1, 2],\n"
            "       [1, 2]], dtype=int32)"
        )
        self.assertEqual(str(x), expected)
        x = mx.array([[1, 2]] * 7)
        expected = (
            "array([[1, 2],\n"
            "       [1, 2],\n"
            "       [1, 2],\n"
            "       ...,\n"
            "       [1, 2],\n"
            "       [1, 2],\n"
            "       [1, 2]], dtype=int32)"
        )
        self.assertEqual(str(x), expected)

        x = mx.array([1], dtype=mx.int8)
        expected = "array([1], dtype=int8)"
        self.assertEqual(str(x), expected)
        x = mx.array([1], dtype=mx.int16)
        expected = "array([1], dtype=int16)"
        self.assertEqual(str(x), expected)
        x = mx.array([1], dtype=mx.uint8)
        expected = "array([1], dtype=uint8)"
        self.assertEqual(str(x), expected)

        # Fp16 is not supported in all platforms
        x = mx.array([1.2], dtype=mx.float16)
        expected = "array([1.2002], dtype=float16)"
        self.assertEqual(str(x), expected)

        x = mx.array([1 + 1j], dtype=mx.complex64)
        expected = "array([1+1j], dtype=complex64)"
        self.assertEqual(str(x), expected)
        x = mx.array([1 - 1j], dtype=mx.complex64)
        expected = "array([1-1j], dtype=complex64)"

        x = mx.array([1 + 1j], dtype=mx.complex64)
        expected = "array([1+1j], dtype=complex64)"
        self.assertEqual(str(x), expected)
        x = mx.array([1 - 1j], dtype=mx.complex64)
        expected = "array([1-1j], dtype=complex64)"

    def test_array_to_list(self):
        types = [mx.bool_, mx.uint32, mx.int32, mx.int64, mx.float32]
        for t in types:
            x = mx.array(1, t)
            self.assertEqual(x.tolist(), 1)

        vals = [1, 2, 3, 4]
        x = mx.array(vals)
        self.assertEqual(x.tolist(), vals)

        vals = [[1, 2], [3, 4]]
        x = mx.array(vals)
        self.assertEqual(x.tolist(), vals)

        vals = [[1, 0], [0, 1]]
        x = mx.array(vals, mx.bool_)
        self.assertEqual(x.tolist(), vals)

        vals = [[1.5, 2.5], [3.5, 4.5]]
        x = mx.array(vals)
        self.assertEqual(x.tolist(), vals)

        vals = [[[0.5, 1.5], [2.5, 3.5]], [[4.5, 5.5], [6.5, 7.5]]]
        x = mx.array(vals)
        self.assertEqual(x.tolist(), vals)

        # Empty arrays
        vals = []
        x = mx.array(vals)
        self.assertEqual(x.tolist(), vals)

        vals = [[], []]
        x = mx.array(vals)
        self.assertEqual(x.tolist(), vals)

        # Complex arrays
        vals = [0.5 + 0j, 1.5 + 1j, 2.5 + 0j, 3.5 + 1j]
        x = mx.array(vals)
        self.assertEqual(x.tolist(), vals)

    def test_array_np_conversion(self):
        # Shape test
        a = np.array([])
        x = mx.array(a)
        self.assertEqual(x.size, 0)
        self.assertEqual(x.shape, [0])
        self.assertEqual(x.dtype, mx.float32)

        a = np.array([[], [], []])
        x = mx.array(a)
        self.assertEqual(x.size, 0)
        self.assertEqual(x.shape, [3, 0])
        self.assertEqual(x.dtype, mx.float32)

        a = np.array([[[], []], [[], []], [[], []]])
        x = mx.array(a)
        self.assertEqual(x.size, 0)
        self.assertEqual(x.shape, [3, 2, 0])
        self.assertEqual(x.dtype, mx.float32)

        # Content test
        a = 2.0 * np.ones((3, 5, 4))
        x = mx.array(a)
        self.assertEqual(x.dtype, mx.float32)
        self.assertEqual(x.ndim, 3)
        self.assertEqual(x.shape, [3, 5, 4])

        y = np.asarray(x)
        self.assertTrue(np.allclose(a, y))

        a = np.array(3, dtype=np.int32)
        x = mx.array(a)
        self.assertEqual(x.dtype, mx.int32)
        self.assertEqual(x.ndim, 0)
        self.assertEqual(x.shape, [])
        self.assertEqual(x.item(), 3)

        # mlx to numpy test
        x = mx.array([True, False, True])
        y = np.asarray(x)
        self.assertEqual(y.dtype, np.bool_)
        self.assertEqual(y.ndim, 1)
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y[0], True)
        self.assertEqual(y[1], False)
        self.assertEqual(y[2], True)

        # complex64 mx <-> np
        cvals = [0j, 1, 1 + 1j]
        x = np.array(cvals)
        y = mx.array(x)
        self.assertEqual(y.dtype, mx.complex64)
        self.assertEqual(y.shape, [3])
        self.assertEqual(y.tolist(), cvals)

        y = mx.array([0j, 1, 1 + 1j])
        x = np.asarray(y)
        self.assertEqual(x.dtype, np.complex64)
        self.assertEqual(x.shape, (3,))
        self.assertEqual(x.tolist(), cvals)

    def test_array_np_dtype_conversion(self):
        dtypes_list = [
            (mx.bool_, np.bool_),
            (mx.uint8, np.uint8),
            (mx.uint16, np.uint16),
            (mx.uint32, np.uint32),
            (mx.uint64, np.uint64),
            (mx.int8, np.int8),
            (mx.int16, np.int16),
            (mx.int32, np.int32),
            (mx.int64, np.int64),
            (mx.float16, np.float16),
            (mx.float32, np.float32),
            (mx.complex64, np.complex64),
        ]

        for mlx_dtype, np_dtype in dtypes_list:
            a_npy = np.random.uniform(low=0, high=100, size=(32,)).astype(np_dtype)
            a_mlx = mx.array(a_npy)

            self.assertEqual(a_mlx.dtype, mlx_dtype)
            self.assertTrue(np.allclose(a_mlx, a_npy))

            b_mlx = mx.random.uniform(
                low=0,
                high=10,
                shape=(32,),
            ).astype(mlx_dtype)
            b_npy = np.array(b_mlx)

            self.assertEqual(b_npy.dtype, np_dtype)

    def test_dtype_promotion(self):
        dtypes_list = [
            (mx.bool_, np.bool_),
            (mx.uint8, np.uint8),
            (mx.uint16, np.uint16),
            (mx.uint32, np.uint32),
            (mx.uint64, np.uint64),
            (mx.int8, np.int8),
            (mx.int16, np.int16),
            (mx.int32, np.int32),
            (mx.int64, np.int64),
            (mx.float32, np.float32),
        ]

        promotion_pairs = permutations(dtypes_list, 2)

        for (mlx_dt_1, np_dt_1), (mlx_dt_2, np_dt_2) in promotion_pairs:
            with self.subTest(dtype1=np_dt_1, dtype2=np_dt_2):
                a_npy = np.ones((3,), dtype=np_dt_1)
                b_npy = np.ones((3,), dtype=np_dt_2)

                c_npy = a_npy + b_npy

                a_mlx = mx.ones((3,), dtype=mlx_dt_1)
                b_mlx = mx.ones((3,), dtype=mlx_dt_2)

                c_mlx = a_mlx + b_mlx

                self.assertEqual(c_mlx.dtype, mx.array(c_npy).dtype)

        a_mlx = mx.ones((3,), dtype=mx.float16)
        b_mlx = mx.ones((3,), dtype=mx.float32)
        c_mlx = a_mlx + b_mlx

        self.assertEqual(c_mlx.dtype, mx.float32)

        b_mlx = mx.ones((3,), dtype=mx.int32)
        c_mlx = a_mlx + b_mlx

        self.assertEqual(c_mlx.dtype, mx.float16)

    def test_dtype_python_scalar_promotion(self):
        tests = [
            (mx.bool_, operator.mul, False, mx.bool_),
            (mx.bool_, operator.mul, 0, mx.int32),
            (mx.bool_, operator.mul, 1.0, mx.float32),
            (mx.int8, operator.mul, False, mx.int8),
            (mx.int8, operator.mul, 0, mx.int8),
            (mx.int8, operator.mul, 1.0, mx.float32),
            (mx.int16, operator.mul, False, mx.int16),
            (mx.int16, operator.mul, 0, mx.int16),
            (mx.int16, operator.mul, 1.0, mx.float32),
            (mx.int32, operator.mul, False, mx.int32),
            (mx.int32, operator.mul, 0, mx.int32),
            (mx.int32, operator.mul, 1.0, mx.float32),
            (mx.int64, operator.mul, False, mx.int64),
            (mx.int64, operator.mul, 0, mx.int64),
            (mx.int64, operator.mul, 1.0, mx.float32),
            (mx.uint8, operator.mul, False, mx.uint8),
            (mx.uint8, operator.mul, 0, mx.uint8),
            (mx.uint8, operator.mul, 1.0, mx.float32),
            (mx.uint16, operator.mul, False, mx.uint16),
            (mx.uint16, operator.mul, 0, mx.uint16),
            (mx.uint16, operator.mul, 1.0, mx.float32),
            (mx.uint32, operator.mul, False, mx.uint32),
            (mx.uint32, operator.mul, 0, mx.uint32),
            (mx.uint32, operator.mul, 1.0, mx.float32),
            (mx.uint64, operator.mul, False, mx.uint64),
            (mx.uint64, operator.mul, 0, mx.uint64),
            (mx.uint64, operator.mul, 1.0, mx.float32),
            (mx.float32, operator.mul, False, mx.float32),
            (mx.float32, operator.mul, 0, mx.float32),
            (mx.float32, operator.mul, 1.0, mx.float32),
            (mx.float16, operator.mul, False, mx.float16),
            (mx.float16, operator.mul, 0, mx.float16),
            (mx.float16, operator.mul, 1.0, mx.float16),
        ]

        for dtype_in, f, v, dtype_out in tests:
            x = mx.array(0, dtype_in)
            y = f(x, v)
            self.assertEqual(y.dtype, dtype_out)

    def test_array_comparison(self):
        a = mx.array([0.0, 1.0, 5.0])
        b = mx.array([-1.0, 2.0, 5.0])

        self.assertEqual((a < b).tolist(), [False, True, False])
        self.assertEqual((a <= b).tolist(), [False, True, True])
        self.assertEqual((a > b).tolist(), [True, False, False])
        self.assertEqual((a >= b).tolist(), [True, False, True])

        self.assertEqual((a < 5).tolist(), [True, True, False])
        self.assertEqual((5 < a).tolist(), [False, False, False])
        self.assertEqual((5 <= a).tolist(), [False, False, True])
        self.assertEqual((a > 1).tolist(), [False, False, True])
        self.assertEqual((a >= 1).tolist(), [False, True, True])

    def test_array_neg(self):
        a = mx.array([-1.0, 4.0, 0.0])

        self.assertEqual((-a).tolist(), [1.0, -4.0, 0.0])

    def test_array_type_cast(self):
        a = mx.array([0.1, 2.3, -1.3])
        b = [0, 2, -1]

        self.assertEqual(a.astype(mx.int32).tolist(), b)
        self.assertEqual(a.astype(mx.int32).dtype, mx.int32)

        b = mx.array(b).astype(mx.float32)
        self.assertEqual(b.dtype, mx.float32)

    def test_array_iteration(self):
        a = mx.array([0, 1, 2])

        for i, x in enumerate(a):
            self.assertEqual(x.item(), i)

        a = mx.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        x, y, z = a
        self.assertEqual(x.tolist(), [1.0, 2.0])
        self.assertEqual(y.tolist(), [3.0, 4.0])
        self.assertEqual(z.tolist(), [5.0, 6.0])

    def test_indexing(self):
        # Basic content check, slice indexing
        a_npy = np.arange(64, dtype=np.float32)
        a_mlx = mx.array(a_npy)
        a_sliced_mlx = a_mlx[2:50:4]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[2:50:4]))

        # Basic content check, mlx array indexing
        a_npy = np.arange(64, dtype=np.int32)
        a_npy = a_npy.reshape((8, 8))
        a_mlx = mx.array(a_npy)
        idx_npy = np.array([0, 1, 2, 7, 5], dtype=np.uint32)
        idx_mlx = mx.array(idx_npy)
        a_sliced_mlx = a_mlx[idx_mlx]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[idx_npy]))

        # Basic content check, int indexing
        a_sliced_mlx = a_mlx[5]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[5]))
        self.assertEqual(len(a_sliced_npy.shape), len(a_npy[5].shape))
        self.assertEqual(len(a_sliced_npy.shape), 1)
        self.assertEqual(a_sliced_npy.shape[0], a_npy[5].shape[0])

        # Basic content check, negative indexing
        a_sliced_mlx = a_mlx[-1]
        self.assertTrue(np.array_equal(a_sliced_mlx, a_npy[-1]))

        # Basic content check, empty index
        a_sliced_mlx = a_mlx[()]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[()]))

        # Basic content check, new axis
        a_sliced_mlx = a_mlx[None]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[None]))

        # Multi dim indexing, all ints
        self.assertEqual(a_mlx[0, 0].item(), 0)
        self.assertEqual(a_mlx[0, 0].ndim, 0)

        # Multi dim indexing, all slices
        a_sliced_mlx = a_mlx[2:4, 5:]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[2:4, 5:]))

        a_sliced_mlx = a_mlx[:, 0:5]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[:, 0:5]))

        # Slicing, strides
        a_sliced_mlx = a_mlx[:, ::2]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[:, ::2]))

        # Slicing, -ve index
        a_sliced_mlx = a_mlx[-2:, :-1]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[-2:, :-1]))

        # Slicing, start > end
        a_sliced_mlx = a_mlx[8:3]
        self.assertEqual(a_sliced_mlx.size, 0)

        # Slicing, Clipping past the end
        a_sliced_mlx = a_mlx[7:10]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[7:10]))

        # Multi dim indexing, int and slices
        a_sliced_mlx = a_mlx[0, :5]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[0, :5]))

        a_sliced_mlx = a_mlx[:, -1]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[:, -1]))

        # Multi dim indexing, int and array
        a_sliced_mlx = a_mlx[idx_mlx, 0]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[idx_npy, 0]))

        # Multi dim indexing, array and slices
        a_sliced_mlx = a_mlx[idx_mlx, :5]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[idx_npy, :5]))

        a_sliced_mlx = a_mlx[:, idx_mlx]
        a_sliced_npy = np.asarray(a_sliced_mlx)
        self.assertTrue(np.array_equal(a_sliced_npy, a_npy[:, idx_npy]))

        # Multi dim indexing with multiple arrays
        def check_slices(arr_np, *idx_np):
            arr_mlx = mx.array(arr_np)
            idx_mlx = [
                mx.array(idx) if isinstance(idx, np.ndarray) else idx for idx in idx_np
            ]
            slice_mlx = arr_mlx[tuple(idx_mlx)]
            self.assertTrue(
                np.array_equal(arr_np[tuple(idx_np)], arr_mlx[tuple(idx_mlx)])
            )

        a_np = np.arange(16).reshape(4, 4)
        check_slices(a_np, np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]))
        check_slices(a_np, np.array([0, 1, 2, 3]), np.array([1, 0, 3, 3]))
        check_slices(a_np, np.array([[0, 1]]), np.array([[0], [1], [3]]))

        a_np = np.arange(64).reshape(2, 4, 2, 4)
        check_slices(a_np, 0, np.array([0, 1, 2]))
        check_slices(a_np, slice(0, 1), np.array([0, 1, 2]))
        check_slices(
            a_np, slice(0, 1), np.array([0, 1, 2]), slice(None), slice(0, 4, 2)
        )
        check_slices(
            a_np, slice(0, 1), np.array([0, 1, 2]), slice(None), np.array([1, 2, 0])
        )
        check_slices(a_np, slice(0, 1), np.array([0, 1, 2]), 1, np.array([1, 2, 0]))
        check_slices(
            a_np, slice(0, 1), np.array([0, 1, 2]), np.array([1, 0, 0]), slice(0, 1)
        )
        check_slices(
            a_np,
            slice(0, 1),
            np.array([[0], [1], [2]]),
            np.array([[1, 0, 0]]),
            slice(0, 1),
        )
        check_slices(
            a_np,
            slice(0, 2),
            np.array([[0], [1], [2]]),
            slice(0, 2),
            np.array([[1, 0, 0]]),
        )
        for p in permutations([slice(None), slice(None), 0, np.array([1, 0])]):
            check_slices(a_np, *p)
        for p in permutations(
            [slice(None), slice(None), 0, np.array([1, 0]), None, None]
        ):
            check_slices(a_np, *p)
        for p in permutations([0, np.array([1, 0]), None, Ellipsis, slice(None)]):
            check_slices(a_np, *p)

        # Non-contiguous arrays in slicing
        a_mlx = mx.reshape(mx.arange(128), (16, 8))
        a_mlx = a_mlx[::2, :]
        a_np = np.array(a_mlx)
        idx_np = np.arange(8)[::2]
        idx_mlx = mx.arange(8)[::2]
        self.assertTrue(
            np.array_equal(a_np[idx_np, idx_np], np.array(a_mlx[idx_mlx, idx_mlx]))
        )

        # Slicing with negative indices and integer
        a_np = np.arange(10).reshape(5, 2)
        a_mlx = mx.array(a_np)
        self.assertTrue(np.array_equal(a_np[2:-1, 0], np.array(a_mlx[2:-1, 0])))

    def test_setitem(self):
        a = mx.array(0)
        a[None] = 1
        self.assertEqual(a.item(), 1)

        a = mx.array([1, 2, 3])
        a[0] = 2
        self.assertEqual(a.tolist(), [2, 2, 3])

        a[-1] = 2
        self.assertEqual(a.tolist(), [2, 2, 2])

        a[0] = mx.array([[[1]]])
        self.assertEqual(a.tolist(), [1, 2, 2])

        a[:] = 0
        self.assertEqual(a.tolist(), [0, 0, 0])

        a[None] = 1
        self.assertEqual(a.tolist(), [1, 1, 1])

        a[0:1] = 2
        self.assertEqual(a.tolist(), [2, 1, 1])

        a[0:2] = 3
        self.assertEqual(a.tolist(), [3, 3, 1])

        a[0:3] = 4
        self.assertEqual(a.tolist(), [4, 4, 4])

        a[0:1] = mx.array(0)
        self.assertEqual(a.tolist(), [0, 4, 4])

        a[0:1] = mx.array([1])
        self.assertEqual(a.tolist(), [1, 4, 4])

        with self.assertRaises(ValueError):
            a[0:1] = mx.array([2, 3])

        a[0:2] = mx.array([2, 2])
        self.assertEqual(a.tolist(), [2, 2, 4])

        a[:] = mx.array([[[[1, 1, 1]]]])
        self.assertEqual(a.tolist(), [1, 1, 1])

        # Array slices
        def check_slices(arr_np, update_np, *idx_np):
            arr_mlx = mx.array(arr_np)
            update_mlx = mx.array(update_np)
            idx_mlx = [
                mx.array(idx) if isinstance(idx, np.ndarray) else idx for idx in idx_np
            ]
            if len(idx_np) > 1:
                idx_np = tuple(idx_np)
                idx_mlx = tuple(idx_mlx)
            else:
                idx_np = idx_np[0]
                idx_mlx = idx_mlx[0]
            arr_np[idx_np] = update_np
            arr_mlx[idx_mlx] = update_mlx
            self.assertTrue(np.array_equal(arr_np, arr_mlx))

        check_slices(np.zeros((3, 3)), 1, 0)
        check_slices(np.zeros((3, 3)), 1, -1)
        check_slices(np.zeros((3, 3)), 1, slice(0, 2))
        check_slices(np.zeros((3, 3)), np.array([[0, 1, 2], [3, 4, 5]]), slice(0, 2))

        with self.assertRaises(ValueError):
            a = mx.array(0)
            a[0] = mx.array(1)

        check_slices(np.zeros((3, 3)), 1, np.array([0, 1, 2]))
        check_slices(np.zeros((3, 3)), np.array(3), np.array([0, 1, 2]))
        check_slices(np.zeros((3, 3)), np.array([3]), np.array([0, 1, 2]))
        check_slices(np.zeros((3, 3)), np.array([3]), np.array([0, 1]))
        check_slices(np.zeros((3, 2)), np.array([[3, 3], [4, 4]]), np.array([0, 1]))
        check_slices(np.zeros((3, 2)), np.array([[3, 3], [4, 4]]), np.array([0, 1]))
        check_slices(
            np.zeros((3, 2)), np.array([[3, 3], [4, 4], [5, 5]]), np.array([0, 0, 1])
        )

        # Multiple slices
        a = mx.array(0)
        a[None, None] = 1
        self.assertEqual(a.item(), 1)

        a[None, None] = mx.array(2)
        self.assertEqual(a.item(), 2)

        a[None, None] = mx.array([[[3]]])
        self.assertEqual(a.item(), 3)

        a[()] = 4
        self.assertEqual(a.item(), 4)

        a_np = np.zeros((2, 3, 4, 5))
        check_slices(a_np, 1, np.array([0, 0]), slice(0, 2), slice(0, 3), 4)
        check_slices(
            a_np,
            np.arange(10).reshape(2, 5),
            np.array([0, 0]),
            np.array([0, 1]),
            np.array([2, 3]),
        )
        check_slices(
            a_np,
            np.array([[3], [4]]),
            np.array([0, 0]),
            np.array([0, 1]),
            np.array([2, 3]),
        )
        check_slices(
            a_np, np.arange(5), np.array([0, 0]), np.array([0, 1]), np.array([2, 3])
        )
        check_slices(np.zeros(5), np.arange(2), None, None, np.array([2, 3]))
        check_slices(
            np.zeros((4, 3, 4)),
            np.arange(3),
            np.array([2, 3]),
            slice(0, 3),
            np.array([2, 3]),
        )

        with self.assertRaises(ValueError):
            a = mx.zeros((4, 3, 4))
            a[mx.array([2, 3]), None, mx.array([2, 3])] = mx.arange(2)

        with self.assertRaises(ValueError):
            a = mx.zeros((4, 3, 4))
            a[mx.array([2, 3]), None, mx.array([2, 3])] = mx.arange(3)

        check_slices(np.zeros((4, 3, 4)), 1, np.array([2, 3]), None, np.array([2, 1]))
        check_slices(
            np.zeros((4, 3, 4)), np.arange(4), np.array([2, 3]), None, np.array([2, 1])
        )
        check_slices(
            np.zeros((4, 3, 4)),
            np.arange(2 * 4).reshape(2, 1, 4),
            np.array([2, 3]),
            None,
            np.array([2, 1]),
        )

        check_slices(np.zeros((4, 4)), 1, slice(0, 2), slice(0, 2))
        check_slices(np.zeros((4, 4)), np.arange(2), slice(0, 2), slice(0, 2))
        check_slices(
            np.zeros((4, 4)), np.arange(2).reshape(2, 1), slice(0, 2), slice(0, 2)
        )
        check_slices(
            np.zeros((4, 4)), np.arange(4).reshape(2, 2), slice(0, 2), slice(0, 2)
        )

        with self.assertRaises(ValueError):
            a = mx.zeros((2, 2, 2))
            a[..., ...] = 1

        with self.assertRaises(ValueError):
            a = mx.zeros((2, 2, 2, 2, 2))
            a[0, ..., 0, ..., 0] = 1

        with self.assertRaises(ValueError):
            a = mx.zeros((2, 2))
            a[0, 0, 0] = 1

        check_slices(np.zeros((2, 2, 2, 2)), 1, None, Ellipsis, None)
        check_slices(
            np.zeros((2, 2, 2, 2)), 1, np.array([0, 1]), Ellipsis, np.array([0, 1])
        )
        check_slices(
            np.zeros((2, 2, 2, 2)),
            np.arange(2 * 2 * 2).reshape(2, 2, 2),
            np.array([0, 1]),
            Ellipsis,
            np.array([0, 1]),
        )

        # Check slice assign with negative indices works
        a = mx.zeros((5, 5), mx.int32)
        a[2:-2, 2:-2] = 4
        self.assertEqual(a[2, 2].item(), 4)

    def test_slice_negative_step(self):
        a_np = np.arange(20)
        a_mx = mx.array(a_np)

        # Basic negative slice
        b_np = a_np[::-1]
        b_mx = a_mx[::-1]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Bounds negative slice
        b_np = a_np[-3:3:-1]
        b_mx = a_mx[-3:3:-1]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Bounds negative slice
        b_np = a_np[25:-50:-1]
        b_mx = a_mx[25:-50:-1]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Jumping negative slice
        b_np = a_np[::-3]
        b_mx = a_mx[::-3]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Bounds and negative slice
        b_np = a_np[-3:3:-3]
        b_mx = a_mx[-3:3:-3]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Bounds and negative slice
        b_np = a_np[25:-50:-3]
        b_mx = a_mx[25:-50:-3]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Negatie slice and ascending bounds
        b_np = a_np[0:20:-3]
        b_mx = a_mx[0:20:-3]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Multi-dim negative slices
        a_np = np.arange(3 * 6 * 4).reshape(3, 6, 4)
        a_mx = mx.array(a_np)

        # Flip each dim
        b_np = a_np[..., ::-1]
        b_mx = a_mx[..., ::-1]
        self.assertTrue(np.array_equal(b_np, b_mx))

        b_np = a_np[:, ::-1, :]
        b_mx = a_mx[:, ::-1, :]
        self.assertTrue(np.array_equal(b_np, b_mx))

        b_np = a_np[::-1, ...]
        b_mx = a_mx[::-1, ...]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Flip pairs of dims
        b_np = a_np[::-1, 1:5:2, ::-2]
        b_mx = a_mx[::-1, 1:5:2, ::-2]
        self.assertTrue(np.array_equal(b_np, b_mx))

        b_np = a_np[::-1, ::-2, 1:5:2]
        b_mx = a_mx[::-1, ::-2, 1:5:2]
        self.assertTrue(np.array_equal(b_np, b_mx))

        # Flip all dims
        b_np = a_np[::-1, ::-3, ::-2]
        b_mx = a_mx[::-1, ::-3, ::-2]
        self.assertTrue(np.array_equal(b_np, b_mx))

    def test_api(self):
        x = mx.array(np.random.rand(10, 10, 10))
        ops = [
            ("reshape", (100, -1)),
            "square",
            "sqrt",
            "rsqrt",
            "reciprocal",
            "exp",
            "log",
            "sin",
            "cos",
            "log1p",
            ("all", 1),
            ("any", 1),
            ("transpose", (0, 2, 1)),
            ("sum", 1),
            ("prod", 1),
            ("min", 1),
            ("max", 1),
            ("logsumexp", 1),
            ("mean", 1),
            ("var", 1),
            ("argmin", 1),
            ("argmax", 1),
        ]
        for op in ops:
            if isinstance(op, tuple):
                op, *args = op
            else:
                args = tuple()
            y1 = getattr(mx, op)(x, *args)
            y2 = getattr(x, op)(*args)
            self.assertEqual(y1.dtype, y2.dtype)
            self.assertEqual(y1.shape, y2.shape)
            self.assertTrue(mx.array_equal(y1, y2))

        y1 = mx.split(x, 2)
        y2 = x.split(2)
        self.assertEqual(len(y1), 2)
        self.assertEqual(len(y1), len(y2))
        self.assertTrue(mx.array_equal(y1[0], y2[0]))
        self.assertTrue(mx.array_equal(y1[1], y2[1]))

    def test_memoryless_copy(self):
        a_mx = mx.ones((2, 2))
        b_mx = mx.broadcast_to(a_mx, (5, 2, 2))

        # Make np arrays without copy
        a_np = np.array(a_mx, copy=False)
        b_np = np.array(b_mx, copy=False)

        # Check that we get read-only array that does not own the underlying data
        self.assertFalse(a_np.flags.owndata)
        self.assertFalse(a_np.flags.writeable)

        # Check contents
        self.assertTrue(np.array_equal(np.ones((2, 2), dtype=np.float32), a_np))
        self.assertTrue(np.array_equal(np.ones((5, 2, 2), dtype=np.float32), b_np))

        # Check strides
        self.assertSequenceEqual(b_np.strides, (0, 8, 4))


if __name__ == "__main__":
    unittest.main()
