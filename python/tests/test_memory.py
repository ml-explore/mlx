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

    def test_buffer_counts(self):
        def check_hist(hist, total):
            # Documented structure: sorted power-of-two size-class upper
            # bounds with positive counts, summing to the total count.
            self.assertEqual(sum(n for _, n in hist), total)
            keys = [k for k, _ in hist]
            self.assertEqual(keys, sorted(keys))
            for k, n in hist:
                self.assertGreater(k, 0)
                self.assertEqual(k & (k - 1), 0)
                self.assertGreater(n, 0)

        mx.synchronize()
        count = mx.get_active_buffer_count()
        cached = mx.get_cache_buffer_count()
        self.assertTrue(count >= 0)
        self.assertTrue(cached >= 0)
        self.assertTrue(cached <= count)
        check_hist(mx.get_buffer_histogram(), count)

        if not mx.metal.is_available():
            # Documented values on backends without a handle limit
            self.assertEqual(count, 0)
            self.assertEqual(cached, 0)
            self.assertEqual(mx.get_buffer_histogram(), [])
            return

        if mx.default_device() != mx.gpu:
            # The probes track the Metal allocator; in CPU mode nothing in
            # this test would land there
            return

        # Warm up one-time allocations so the baseline below is stable
        warm = mx.zeros((4096,))
        mx.eval(warm)
        del warm
        mx.synchronize()
        mx.clear_cache()

        baseline = mx.get_active_buffer_count()
        baseline_hist = mx.get_buffer_histogram()
        self.assertEqual(mx.get_cache_buffer_count(), 0)

        # A fresh 16 KB allocation raises the live count and lands in the
        # 16384 size class
        a = mx.zeros((4096,))  # 4096 x float32 = 16384 bytes
        mx.eval(a)
        mx.synchronize()
        with_alloc = mx.get_active_buffer_count()
        self.assertGreater(with_alloc, baseline)
        hist = mx.get_buffer_histogram()
        check_hist(hist, with_alloc)
        self.assertGreaterEqual(
            dict(hist).get(16384, 0), dict(baseline_hist).get(16384, 0) + 1
        )

        # With `a` live and the cache cleared, active and cached must
        # disagree: live buffers are counted but are not cache entries
        mx.clear_cache()
        self.assertEqual(mx.get_cache_buffer_count(), 0)
        self.assertGreater(mx.get_active_buffer_count(), 0)
        post_clear = mx.get_active_buffer_count()

        # Freeing recycles into the cache: still counted, now visible as
        # cached
        del a
        mx.synchronize()
        self.assertGreater(mx.get_cache_buffer_count(), 0)
        self.assertEqual(mx.get_active_buffer_count(), post_clear)
        check_hist(mx.get_buffer_histogram(), post_clear)

        # Clearing the cache releases the buffers and restores the baseline
        mx.clear_cache()
        self.assertEqual(mx.get_cache_buffer_count(), 0)
        self.assertEqual(mx.get_active_buffer_count(), baseline)
        self.assertEqual(mx.get_buffer_histogram(), baseline_hist)

        # malloc-from-cache: reallocating a freed size reuses the cached
        # buffer instead of creating a new one
        a = mx.zeros((4096,))
        mx.eval(a)
        mx.synchronize()
        total = mx.get_active_buffer_count()
        del a
        mx.synchronize()
        self.assertEqual(mx.get_active_buffer_count(), total)
        a = mx.zeros((4096,))
        mx.eval(a)
        mx.synchronize()
        self.assertLessEqual(mx.get_active_buffer_count(), total)
        check_hist(mx.get_buffer_histogram(), mx.get_active_buffer_count())
        del a
        mx.synchronize()
        mx.clear_cache()

        # free-to-release: with the cache disabled, freeing decrements the
        # count immediately
        old_limit = mx.set_cache_limit(0)
        try:
            a = mx.zeros((4096,))
            mx.eval(a)
            mx.synchronize()
            before_free = mx.get_active_buffer_count()
            del a
            mx.synchronize()
            self.assertLess(mx.get_active_buffer_count(), before_free)
            check_hist(mx.get_buffer_histogram(), mx.get_active_buffer_count())
        finally:
            mx.set_cache_limit(old_limit)

        # cache eviction: buffers resting in the cache are released by a
        # later malloc's cache trim (set_cache_limit does not trim; the
        # 32-byte request cannot reuse the 16 KB entry), so the count
        # and histogram cross the eviction-callback path. The zeros()
        # eval may cache more than one buffer (e.g. its fill scalar), so
        # count relative to the recorded cached population.
        a = mx.zeros((4096,))
        mx.eval(a)
        mx.synchronize()
        del a
        mx.synchronize()
        cached_before = mx.get_cache_buffer_count()
        self.assertGreater(cached_before, 0)
        with_cached = mx.get_active_buffer_count()
        hist_cached = dict(mx.get_buffer_histogram())
        old_limit = mx.set_cache_limit(0)
        try:
            b = mx.zeros((8,))
            mx.eval(b)
            mx.synchronize()
            self.assertEqual(mx.get_cache_buffer_count(), 0)
            after = mx.get_active_buffer_count()
            # every previously cached buffer evicted, one output live
            self.assertEqual(after, with_cached - cached_before + 1)
            check_hist(mx.get_buffer_histogram(), after)
            self.assertEqual(
                dict(mx.get_buffer_histogram()).get(16384, 0),
                hist_cached.get(16384, 0) - 1,
            )
            del b
            mx.synchronize()
        finally:
            mx.set_cache_limit(old_limit)
        mx.clear_cache()

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
