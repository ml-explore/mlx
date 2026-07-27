# Copyright © 2023-2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests


class TestMemory(mlx_tests.MLXTestCase):
    def test_memory_info(self):
        old_limit = mx.set_cache_limit(0)

        a = mx.zeros((4096,))
        mx.eval(a)
        del a
        self.assertEqual(mx.get_cache_memory(), 0)
        self.assertEqual(mx.set_cache_limit(old_limit), 0)
        self.assertEqual(mx.set_cache_limit(old_limit), old_limit)

        old_limit = mx.set_memory_limit(10)
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

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_growing_concatenate_reuses_cache(self):
        # Regression test for #3886: BufferCache.reuse_from_cache anchors on
        # std::multimap::lower_bound(size), so it only ever returns a
        # same-or-larger cached buffer. A monotonically GROWING allocation
        # sequence -- the natural pattern of a per-token KV-cache built on
        # `concatenate` -- freed a buffer smaller than every subsequent
        # request, so it could never be reused: every step missed the cache,
        # and the never-reused buffers piled up in the pool. Measured before
        # this fix, on this exact pattern: ~2129 MB left in the pool (96x the
        # final array) after 100 growth steps. Large buffers are now rounded
        # up to a coarse, scale-relative size class before the cache lookup,
        # so consecutive small growth steps land on the same over-provisioned
        # class and actually get reused.
        mx.synchronize()
        mx.clear_cache()

        base_rows, step_rows, cols = 5000, 4, 1024
        a = mx.zeros((base_rows, cols))
        mx.eval(a)

        n_steps = 100
        for _ in range(n_steps):
            a = mx.concatenate([a, mx.zeros((step_rows, cols))], axis=0)
            mx.eval(a)
        mx.synchronize()

        # Measured with the fix: ~131 MB (5.9x final_bytes). Measured on the
        # unpatched allocator with this identical pattern: ~2129 MB (96x).
        # 20x is a generous margin above the fixed number and decisively
        # below the unpatched one, so this fails loudly if the reuse
        # regresses without being sensitive to small measurement noise.
        self.assertLess(mx.get_cache_memory(), 20 * a.nbytes)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
