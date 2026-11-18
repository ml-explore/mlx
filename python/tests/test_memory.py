# Copyright © 2023-2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests


class TestMemory(mlx_tests.MLXTestCase):
    def test_array_buffer_size(self):
        a = mx.array([1.0, 2.0, 3.0, 4.0])
        view = a[:1]
        lazy = a + 1

        with self.assertRaises(ValueError):
            mx.get_array_buffer_size({"lazy": lazy})

        mx.eval(view)
        a_size = mx.get_array_buffer_size(a)
        self.assertGreaterEqual(a_size, a.nbytes)
        self.assertLess(view.nbytes, a.nbytes)
        self.assertEqual(mx.get_array_buffer_size(), 0)
        self.assertEqual(mx.get_array_buffer_size(view), a_size)
        self.assertEqual(
            mx.get_array_buffer_size({"a": a, "nested": (a, None)}), a_size
        )
        self.assertEqual(mx.get_array_buffer_size(mx.array([])), 0)

        b = mx.array([5.0, 6.0])
        self.assertEqual(
            mx.get_array_buffer_size(a, b),
            a_size + mx.get_array_buffer_size(b),
        )

    def test_memory_info(self):
        old_limit = mx.set_cache_limit(0)

        a = mx.zeros((4096,))
        mx.eval(a)
        del a
        self.assertEqual(mx.get_cache_memory(), 0)
        self.assertEqual(mx.set_cache_limit(old_limit), 0)
        self.assertEqual(mx.set_cache_limit(old_limit), old_limit)

        old_limit = mx.set_memory_limit(10)
        self.assertEqual(mx.get_memory_limit(), 10)
        self.assertEqual(mx.set_memory_limit(old_limit), 10)
        self.assertEqual(mx.set_memory_limit(old_limit), old_limit)

        # Query active and peak memory
        a = mx.zeros((4096,))
        mx.eval(a)
        mx.synchronize()
        active_mem = mx.get_active_memory()
        self.assertTrue(active_mem >= 4096 * 4)

        b = mx.zeros((4096,))
        mx.eval(b)
        del b
        mx.synchronize()

        new_active_mem = mx.get_active_memory()
        self.assertEqual(new_active_mem, active_mem)
        peak_mem = mx.get_peak_memory()
        self.assertTrue(peak_mem >= 4096 * 8)

        if mx.metal.is_available():
            cache_mem = mx.get_cache_memory()
            self.assertTrue(cache_mem >= 4096 * 4)

        mx.clear_cache()
        self.assertEqual(mx.get_cache_memory(), 0)

        mx.reset_peak_memory()
        self.assertEqual(mx.get_peak_memory(), 0)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_wired_memory(self):
        old_limit = mx.set_wired_limit(1000)
        old_limit = mx.set_wired_limit(0)
        self.assertEqual(old_limit, 1000)

        max_size = mx.device_info(mx.gpu)["max_recommended_working_set_size"]
        with self.assertRaises(ValueError):
            mx.set_wired_limit(max_size + 10)

    def test_active_memory_count(self):
        mx.synchronize()
        mx.clear_cache()
        init_mem = mx.get_active_memory()
        a = mx.zeros((128, 128))
        mx.eval(a)
        mx.synchronize()
        del a
        a = mx.zeros((90, 128))
        mx.eval(a)
        mx.synchronize()
        del a
        self.assertEqual(init_mem, mx.get_active_memory())


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
