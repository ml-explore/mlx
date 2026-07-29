# Copyright © 2024 Apple Inc.

import gc
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestZeroCopy(mlx_tests.MLXTestCase):
    """Tests for zero-copy CPU import: mx.asarray(host_buffer, copy=False).

    On unified memory (Metal) a page-aligned CPU buffer is adopted instead of
    copied. On backends without Metal, or for a non-page-aligned buffer or a
    dtype conversion, copy=False raises.
    """

    def test_default_shares_or_copies(self):
        # With copy=None MLX adopts the buffer when it can and copies otherwise,
        # so either way the values must match the source at import time.
        a = np.arange(1_000_000, dtype=np.int32)
        x = mx.asarray(a)
        mx.eval(x)
        self.assertTrue(np.array_equal(np.array(x), a))

    def test_copy_true_copies(self):
        a = np.arange(1_000_000, dtype=np.int32)
        x = mx.asarray(a, copy=True)
        a[0] = 12345
        mx.eval(x)
        self.assertNotEqual(int(x[0]), 12345)

    def test_copy_false(self):
        a = np.arange(1_000_000, dtype=np.int32)
        if not mx.metal.is_available():
            with self.assertRaises(Exception):
                mx.asarray(a, copy=False)
            return
        if a.ctypes.data % 16384 != 0:
            self.skipTest("source buffer not page-aligned; adopt path not taken")
        x = mx.asarray(a, copy=False)
        self.assertTrue(np.array_equal(np.array(x), a))
        # Zero-copy adoption: a mutation of the source is visible in the array.
        a[1] = 999
        mx.eval(x)
        self.assertEqual(int(x[1]), 999)

    def test_copy_false_buffer_counts(self):
        """Adopted host buffers enter and leave the public live probes."""
        if not mx.metal.is_available():
            self.skipTest("copy=False requires Metal")
        if mx.default_device() != mx.gpu:
            self.skipTest("the probes track the Metal allocator")

        nbytes = 4 * 1024 * 1024
        raw = np.empty(nbytes + 16384, dtype=np.uint8)
        offset = (-raw.ctypes.data) % 16384
        aligned = raw[offset : offset + nbytes]
        a = aligned.view(np.int32)
        self.assertEqual(a.ctypes.data % 16384, 0)

        gc.collect()
        mx.synchronize()
        baseline_count = mx.get_active_buffer_count()
        baseline_hist = dict(mx.get_buffer_histogram())

        x = mx.asarray(a, copy=False)
        mx.synchronize()
        self.assertEqual(mx.get_active_buffer_count(), baseline_count + 1)
        histogram = dict(mx.get_buffer_histogram())
        self.assertEqual(
            histogram.get(nbytes, 0),
            baseline_hist.get(nbytes, 0) + 1,
        )

        del x, a, aligned, raw
        gc.collect()
        mx.synchronize()
        self.assertEqual(mx.get_active_buffer_count(), baseline_count)
        self.assertEqual(dict(mx.get_buffer_histogram()), baseline_hist)

    def test_copy_false_dtype_conversion_raises(self):
        a = np.arange(16, dtype=np.float64)
        with self.assertRaises(Exception):
            mx.asarray(a, dtype=mx.float32, copy=False)

    def test_source_lifetime(self):
        if not mx.metal.is_available():
            self.skipTest("copy=False requires Metal")

        def make():
            a = np.arange(1_000_000, dtype=np.float32) + 0.5
            if a.ctypes.data % 16384 != 0:
                return None
            return mx.asarray(a, copy=False)

        x = make()
        if x is None:
            self.skipTest("source buffer not page-aligned")
        gc.collect()
        mx.eval(x + 1)
        self.assertAlmostEqual(float(x[10]), 10.5, places=5)

    def test_adopt_in_loop_not_recycled(self):
        # Regression: an adopted buffer must be released (not recycled into the
        # allocator's reuse pool) when freed. Otherwise, over many iterations the
        # pool hands a caller-owned buffer to an unrelated array and corrupts /
        # crashes. Adopt fresh buffers in a loop and compute after each.
        if not mx.metal.is_available():
            self.skipTest("copy=False requires Metal")
        w = mx.random.normal((64, 64))
        for i in range(200):
            a = np.random.rand(256, 64).astype(np.float32)
            x = mx.asarray(a, copy=False)  # adopt; x (and a) freed next iteration
            r = mx.sum(x @ w)
            mx.eval(r)
        self.assertTrue(True)  # reaching here without crashing is the assertion


if __name__ == "__main__":
    unittest.main()
