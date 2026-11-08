# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


@unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
class TestMetalStreamReuse(mlx_tests.MLXTestCase):
    def setUp(self):
        super().setUp()
        mx.set_default_device(mx.gpu)
        self.streams = [mx.default_stream(mx.gpu)]

    def tearDown(self):
        for stream in self.streams:
            mx.synchronize(stream)
        super().tearDown()

    def test_matmul_chain_peak_memory(self):
        weight = mx.eye(256)
        mx.eval(weight)
        with mx.config.scoped_update(
            MLX_MAX_OPS_PER_BUFFER=1024, MLX_MAX_MB_PER_BUFFER=1024
        ):
            for depth in (24, 96):
                with self.subTest(depth=depth):
                    hidden = mx.ones((256, 256))
                    hidden = mx.contiguous(hidden)
                    mx.eval(hidden)
                    mx.synchronize()
                    before = mx.get_active_memory()
                    mx.reset_peak_memory()
                    for _ in range(depth):
                        hidden = hidden @ weight
                    mx.eval(hidden)
                    peak = mx.get_peak_memory() - before
                    mx.synchronize()
                    self.assertLessEqual(peak, 4 * hidden.nbytes)
                    self.assertTrue(mx.all(hidden == 1))

    def test_retained_arrays_and_views(self):
        weight = mx.eye(64) * 0.5
        hidden = mx.arange(64 * 64, dtype=mx.float32).reshape(64, 64)
        original = np.array(hidden)
        saved = []
        for i in range(12):
            hidden = hidden @ weight
            if i % 3 == 0:
                saved.append((hidden, hidden[:, ::2], i + 1))
        mx.eval(hidden, saved)
        for value, view, depth in saved:
            expected = original * 0.5**depth
            np.testing.assert_array_equal(np.array(value), expected)
            np.testing.assert_array_equal(np.array(view), expected[:, ::2])
        np.testing.assert_array_equal(np.array(hidden), original * 0.5**12)

    def test_shapeless_mean_after_scalar_reductions(self):
        def fun(x):
            total = mx.sum(x)
            squares = mx.sum(x * x)
            return mx.mean(x + total + squares)

        compiled = mx.compile(fun, shapeless=True)
        # Keep the scalar GPU writes in the same command buffer.
        with mx.config.scoped_update(
            MLX_MAX_OPS_PER_BUFFER=1024, MLX_MAX_MB_PER_BUFFER=1024
        ):
            for length in (16, 128, 1024):
                with self.subTest(length=length):
                    x = mx.arange(length, dtype=mx.float32)
                    mx.eval(x)
                    mx.synchronize()
                    values = np.arange(length, dtype=np.float64)
                    expected = np.mean(
                        values + np.sum(values) + np.sum(values * values)
                    )
                    actual = compiled(x).item()
                    np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_reuse_across_command_buffers(self):
        with mx.config.scoped_update(MLX_MAX_OPS_PER_BUFFER=1, MLX_MAX_MB_PER_BUFFER=1):
            weight = mx.eye(256) * 0.5
            hidden = mx.arange(256 * 256, dtype=mx.float32).reshape(256, 256)
            expected = np.array(hidden) * 0.5**24
            for _ in range(24):
                hidden = hidden @ weight
            mx.eval(hidden)
            np.testing.assert_array_equal(np.array(hidden), expected)

    def test_reuse_with_concurrent_copies(self):
        weight = mx.eye(256)
        hidden = mx.arange(256 * 256, dtype=mx.float32).reshape(256, 256)
        expected = np.array(hidden)
        for _ in range(24):
            hidden = hidden @ weight
            hidden = mx.concatenate((hidden[:128], hidden[128:]))
        mx.eval(hidden)
        np.testing.assert_array_equal(np.array(hidden), expected)

    def test_other_stream_cannot_overwrite_pending_input(self):
        for consumer in (mx.gpu, mx.cpu):
            with self.subTest(consumer=consumer):
                first = mx.new_stream(mx.gpu)
                self.streams.append(first)
                source = mx.arange(1024 * 1024, dtype=mx.float32).reshape(1024, 1024)
                expected = np.array(source)
                weight = mx.eye(1024)
                mx.eval(weight)
                with mx.stream(first):
                    delayed = mx.ones((1024, 1024))
                    for _ in range(24):
                        delayed = delayed @ weight
                    result = delayed + source
                    mx.async_eval(result)
                other = mx.new_stream(consumer)
                self.streams.append(other)
                changed = mx.negative(source, stream=other)
                del source
                mx.async_eval(changed)
                mx.eval(result, changed)
                np.testing.assert_array_equal(np.array(result), expected + 1)
                np.testing.assert_array_equal(np.array(changed), -expected)

    def test_other_stream_cannot_reuse_pending_buffer(self):
        first = mx.new_stream(mx.gpu)
        second = mx.new_stream(mx.gpu)
        self.streams.extend((first, second))
        source = mx.arange(1024 * 1024, dtype=mx.float32).reshape(1024, 1024)
        expected = np.array(source) + 1
        weight = mx.eye(1024)
        mx.eval(weight)
        with mx.stream(first):
            delayed = mx.ones((1024, 1024))
            for _ in range(24):
                delayed = delayed @ weight
            result = delayed + source
            mx.async_eval(result)
        del source
        replacement = mx.contiguous(
            mx.full((1024, 1024), 99.0, stream=second), stream=second
        )
        mx.eval(replacement, result)
        np.testing.assert_array_equal(np.array(result), expected)
        self.assertTrue(mx.all(replacement == 99))

    def test_async_outputs_survive_cache_clear(self):
        previous = mx.set_cache_limit(0)
        try:
            weight = mx.eye(256) * 0.5
            hidden = mx.ones((256, 256))
            for _ in range(24):
                hidden = hidden @ weight
            mx.async_eval(hidden)
            mx.clear_cache()
            mx.eval(hidden)
            self.assertTrue(mx.all(hidden == 0.5**24))
        finally:
            mx.set_cache_limit(previous)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
