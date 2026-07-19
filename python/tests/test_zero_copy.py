# Copyright © 2024 Apple Inc.

import gc
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestZeroCopy(mlx_tests.MLXTestCase):
    """Tests for zero-copy CPU import: mx.array(host_buffer, copy=False).

    On unified memory (Metal) a page-aligned CPU buffer is adopted via
    newBufferWithBytesNoCopy instead of copied. On backends without Metal, or
    for a non-page-aligned buffer / a dtype conversion, copy=False raises.
    """

    def test_default_copies(self):
        a = np.arange(1_000_000, dtype=np.int32)
        x = mx.array(a)  # default: copy
        a[0] = 12345
        mx.eval(x)
        self.assertNotEqual(int(x[0]), 12345)

    def test_copy_false(self):
        a = np.arange(1_000_000, dtype=np.int32)
        if not mx.metal.is_available():
            with self.assertRaises(Exception):
                mx.array(a, copy=False)
            return
        if a.ctypes.data % 16384 != 0:
            self.skipTest("source buffer not page-aligned; adopt path not taken")
        x = mx.array(a, copy=False)
        self.assertTrue(np.array_equal(np.array(x), a))
        # Zero-copy adoption: a mutation of the source is visible in the array.
        a[1] = 999
        mx.eval(x)
        self.assertEqual(int(x[1]), 999)

    def test_copy_false_dtype_conversion_raises(self):
        a = np.arange(16, dtype=np.float64)
        with self.assertRaises(Exception):
            mx.array(a, dtype=mx.float32, copy=False)

    def test_source_lifetime(self):
        if not mx.metal.is_available():
            self.skipTest("copy=False requires Metal")

        def make():
            a = np.arange(1_000_000, dtype=np.float32) + 0.5
            if a.ctypes.data % 16384 != 0:
                return None
            return mx.array(a, copy=False)

        x = make()
        if x is None:
            self.skipTest("source buffer not page-aligned")
        gc.collect()
        mx.eval(x + 1)
        self.assertAlmostEqual(float(x[10]), 10.5, places=5)


if __name__ == "__main__":
    unittest.main()
