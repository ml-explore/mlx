# Copyright © 2023 Apple Inc.

import unittest
from functools import partial

import mlx.core as mx
import mlx_tests


class TestEval(mlx_tests.MLXTestCase):
    def test_eval(self):
        arrs = [mx.ones((2, 2)) for _ in range(4)]
        mx.eval(*arrs)
        for x in arrs:
            self.assertEqual(x.tolist(), [[1, 1], [1, 1]])

    def test_retain_graph(self):
        def fun(x):
            y = 3 * x
            mx.eval(y)
            return 2 * y

        dfun_dx = mx.grad(fun)
        y = dfun_dx(mx.array(1.0))
        self.assertEqual(y.item(), 6.0)

    def test_eval_mixed(self):
        x = mx.array(1) + 1 + 1
        y = 0
        z = "hello"
        state = [x, y, z]
        mx.eval(state)
        self.assertEqual(x.item(), 3)

    def test_async_eval(self):
        x = mx.array(1) + mx.array(1) + mx.array(1)
        mx.async_eval(x)
        self.assertEqual(x.item(), 3)

        # It should be safe to call eval on the array which has been async
        # eval'ed
        x = mx.array(1) + mx.array(1) + mx.array(1)
        self.assertEqual(x.item(), 3)

        x = mx.array([1, 2, 3])
        y = 2 * x
        mx.async_eval(y)
        z = 2 * y
        mx.async_eval(z)
        self.assertTrue(mx.array_equal(y, mx.array([2, 4, 6])))
        self.assertTrue(mx.array_equal(z, mx.array([4, 8, 12])))

    def test_async_eval_twice(self):
        for _ in range(1000):
            x = mx.array(1) + mx.array(1) + mx.array(1)
            mx.async_eval(x)
            y = x + 1
            mx.async_eval(y)
            self.assertEqual(x.item(), 3)
            self.assertEqual(y.item(), 4)

    def test_async_eval_in_trace(self):
        def fun(x):
            y = x + 1.0
            mx.async_eval(y)
            return mx.exp(y)

        # Raises
        with self.assertRaises(ValueError):
            mx.grad(fun)(mx.array(1.0))

        # Also raises
        with self.assertRaises(ValueError):
            mx.vmap(fun)(mx.ones((2, 2)))

    def test_async_eval_into_eval(self):
        x = mx.array(1)
        y = x + 1
        mx.async_eval(y)
        a = y - 10
        b = mx.abs(a)
        self.assertEqual(b.item(), 8)

    def test_async_eval_into_eval_diff_stream(self):
        s = mx.new_stream(mx.cpu)
        x = mx.array(0)
        y = x - 5
        mx.async_eval(y)
        z = mx.abs(y, stream=s)
        self.assertEqual(z.item(), 5)

    def test_eval_slow_fast_multi_stream(self):
        x = mx.ones((8000,))
        y = mx.abs(mx.array(-1.0))
        for _ in range(20):
            x = x + mx.array(1.0)
        z = mx.add(x, y, stream=mx.cpu)
        self.assertTrue(mx.allclose(z, mx.full((8000,), 22.0)))

        # Switch eval order
        x = mx.ones((8000,))
        y = mx.abs(mx.array(-1.0))
        for _ in range(20):
            x = x + mx.array(1.0)
        z = mx.add(y, x, stream=mx.cpu)
        self.assertTrue(mx.allclose(z, mx.full((8000,), 22.0)))

    def test_multi_output_eval_during_transform(self):
        x = mx.random.uniform(shape=(1024,))
        y = mx.ones((1024,))
        mx.eval(x, y)

        def fn(x):
            a, b = mx.divmod(x, x)
            mx.eval(a)
            return a

        out = mx.vjp(fn, (x,), (y,))
        out = mx.vjp(fn, (x,), (y,))
        if mx.metal.is_available():
            peak_mem = mx.metal.get_peak_memory()
            out = mx.vjp(fn, (x,), (y,))
            self.assertEqual(peak_mem, mx.metal.get_peak_memory())


if __name__ == "__main__":
    unittest.main()
