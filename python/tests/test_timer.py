# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests


class TestTimer(mlx_tests.MLXTestCase):
    def test_cpu_is_not_supported(self):
        with self.assertRaises(ValueError):
            mx.Timer(mx.cpu)

    @unittest.skipUnless(mx.is_available(mx.gpu), "GPU is not available")
    def test_interval(self):
        x = mx.random.uniform(shape=(256, 256))
        y = mx.random.uniform(shape=(256, 256))
        mx.eval(x, y)

        timer = mx.Timer()
        x, y = timer.start(x, y)
        out = (x @ y) * 2.0
        out = timer.stop(out)

        mx.async_eval(out)
        elapsed = timer.elapsed_time()
        self.assertGreater(elapsed, 0.0)
        self.assertTrue(mx.allclose(out, (x @ y) * 2.0))
        self.assertEqual(timer.elapsed_time(), elapsed)

    @unittest.skipUnless(mx.is_available(mx.gpu), "GPU is not available")
    def test_output_pytree(self):
        x = mx.arange(8)
        mx.eval(x)

        timer = mx.Timer()
        x = timer.start(x)
        out = timer.stop({"a": x + 1, "b": x * 2})

        self.assertGreater(timer.elapsed_time(), 0.0)
        self.assertTrue(mx.array_equal(out["a"], x + 1))
        self.assertTrue(mx.array_equal(out["b"], x * 2))

    @unittest.skipUnless(mx.is_available(mx.gpu), "GPU is not available")
    def test_stream(self):
        stream = mx.new_stream(mx.gpu)
        x = mx.arange(8)
        mx.eval(x)

        timer = mx.Timer(stream)
        x = timer.start(x)
        with mx.stream(stream):
            out = x + 1
        out = timer.stop(out)

        self.assertEqual(timer.stream, stream)
        self.assertGreater(timer.elapsed_time(), 0.0)
        self.assertTrue(mx.array_equal(out, x + 1))

    @unittest.skipUnless(mx.is_available(mx.gpu), "GPU is not available")
    def test_invalid_usage(self):
        timer = mx.Timer()
        with self.assertRaises(RuntimeError):
            timer.elapsed_time()
        with self.assertRaises(ValueError):
            timer.start()

        x = mx.array(1.0)
        with self.assertRaises(RuntimeError):
            timer.stop(x)
        x = timer.start(x)
        with self.assertRaises(RuntimeError):
            timer.start(x)
        with self.assertRaises(ValueError):
            timer.stop()
        out = timer.stop(x + 1)
        with self.assertRaises(RuntimeError):
            timer.stop(out)
        self.assertGreater(timer.elapsed_time(), 0.0)
        self.assertEqual(out.item(), 2.0)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
