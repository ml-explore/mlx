# Copyright © 2026 Apple Inc.
"""Tests for the DLPack consumer path in ``mx.array``.

These tests cover scenarios that don't depend on a third-party Metal producer:

1. Round trip through NumPy's DLPack exporter (``kDLCPU`` path).
2. Self round trip via ``mx.array.__dlpack__`` (``kDLMetal`` on Metal hosts,
   ``kDLCPU`` otherwise).
3. Negative paths: used capsule, unsupported strided view, etc.
"""

from __future__ import annotations

import unittest
import ctypes

import numpy as np

try:
    import mlx.core as mx
except ImportError as exc:  # pragma: no cover - import error is environment specific
    raise unittest.SkipTest(f"mlx.core unavailable: {exc}")


class TestArrayDLPackBasic(unittest.TestCase):
    def test_mx_array_accepts_dlpack_capsule(self):
        # Pass a raw PyCapsule rather than the producer object.
        arr_np = np.arange(8, dtype=np.int32).reshape(2, 4)
        capsule = arr_np.__dlpack__()
        arr_mx = mx.array(capsule)
        self.assertEqual(tuple(arr_mx.shape), (2, 4))
        self.assertEqual(arr_mx.dtype, mx.int32)
        self.assertTrue(np.array_equal(np.asarray(arr_mx), arr_np))

    def test_mx_array_accepts_dlpack_producer(self):
        class DLPackProducer:
            def __init__(self, array):
                self.array = array

            def __dlpack__(self):
                return self.array.__dlpack__()

            def __dlpack_device__(self):
                return self.array.__dlpack_device__()

        arr_np = np.arange(12, dtype=np.float32).reshape(3, 4)
        arr_mx = mx.array(DLPackProducer(arr_np))
        self.assertEqual(tuple(arr_mx.shape), (3, 4))
        self.assertEqual(arr_mx.dtype, mx.float32)
        self.assertTrue(np.allclose(np.asarray(arr_mx), arr_np))

    def test_mx_array_accepts_mlx_dlpack_producer(self):
        class DLPackProducer:
            def __init__(self, array):
                self.array = array

            def __dlpack__(self):
                return self.array.__dlpack__()

            def __dlpack_device__(self):
                return self.array.__dlpack_device__()

        x = mx.arange(20, dtype=mx.float32).reshape(4, 5)
        y = mx.array(DLPackProducer(x))
        self.assertTrue(mx.array_equal(x, y).item())

    def test_mx_array_dlpack_dtype_override(self):
        arr_np = np.arange(6, dtype=np.int32).reshape(2, 3)
        arr_mx = mx.array(arr_np.__dlpack__(), dtype=mx.float32)
        self.assertEqual(arr_mx.dtype, mx.float32)
        self.assertTrue(np.array_equal(np.asarray(arr_mx), arr_np.astype(np.float32)))

    def test_mx_array_prefers_mlx_array_protocol_over_dlpack(self):
        class BothProtocols:
            def __mlx_array__(self):
                return mx.array([1, 2, 3], dtype=mx.int32)

            def __dlpack__(self):
                raise AssertionError("__dlpack__ should not be called")

        arr_mx = mx.array(BothProtocols())
        self.assertEqual(arr_mx.dtype, mx.int32)
        self.assertTrue(np.array_equal(np.asarray(arr_mx), np.array([1, 2, 3])))

    def test_dtypes(self):
        cases = [
            (np.bool_, mx.bool_),
            (np.int8, mx.int8),
            (np.int16, mx.int16),
            (np.int32, mx.int32),
            (np.int64, mx.int64),
            (np.uint8, mx.uint8),
            (np.uint16, mx.uint16),
            (np.uint32, mx.uint32),
            (np.uint64, mx.uint64),
            (np.float16, mx.float16),
            (np.float32, mx.float32),
            (np.float64, mx.float64),
            (np.complex64, mx.complex64),
        ]
        for np_dtype, mx_dtype in cases:
            with self.subTest(np_dtype=np_dtype):
                arr = np.zeros((2, 3), dtype=np_dtype)
                if np_dtype is np.bool_:
                    arr[0, 0] = True
                else:
                    arr[0, 0] = 1
                converted = mx.array(arr.__dlpack__())
                # `mx.array` applies the same dtype defaults to DLPack inputs
                # as it does to NumPy inputs, e.g. float64 defaults to float32.
                self.assertEqual(converted.dtype, mx.array(arr).dtype)
                self.assertEqual(tuple(converted.shape), (2, 3))


class TestArrayDLPackErrors(unittest.TestCase):
    def test_rejects_used_capsule(self):
        arr_np = np.arange(4, dtype=np.float32)
        capsule = arr_np.__dlpack__()
        # First call consumes; second must fail because the capsule was
        # renamed to "used_dltensor".
        _ = mx.array(capsule)
        with self.assertRaises(Exception):
            mx.array(capsule)


class TestArrayDLPackNonContiguous(unittest.TestCase):
    def test_strided_view_rejected(self):
        # MLX's first-cut consumer does not support arbitrary DLPack strides.
        # NumPy emits __dlpack__ with explicit strides for slices; producers
        # may or may not encode strides depending on contiguity. We assert
        # that a non-row-contiguous slice is rejected with a clear error
        # rather than silently misinterpreting the layout.
        big = np.arange(16, dtype=np.float32).reshape(4, 4)
        view = big[::2, :]
        try:
            capsule = view.__dlpack__()
        except (TypeError, BufferError):
            self.skipTest(
                "NumPy refused to export a non-contiguous DLPack capsule"
            )
        with self.assertRaises(Exception):
            mx.array(capsule)

    def test_rejected_capsule_is_not_marked_used(self):
        big = np.arange(16, dtype=np.float32).reshape(4, 4)
        view = big[::2, :]
        try:
            capsule = view.__dlpack__()
        except (TypeError, BufferError):
            self.skipTest(
                "NumPy refused to export a non-contiguous DLPack capsule"
            )

        with self.assertRaises(Exception):
            mx.array(capsule)

        get_name = ctypes.pythonapi.PyCapsule_GetName
        get_name.argtypes = [ctypes.py_object]
        get_name.restype = ctypes.c_char_p
        self.assertEqual(get_name(capsule), b"dltensor")


if __name__ == "__main__":
    unittest.main()
