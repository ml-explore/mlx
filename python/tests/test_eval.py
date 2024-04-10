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
        sync = mx.async_eval(x)
        sync.wait()
        self.assertEqual(x.item(), 3)

        # It should be safe to call eval on the array which has been async
        # eval'ed
        x = mx.array(1) + mx.array(1) + mx.array(1)
        sync = mx.async_eval(x)
        self.assertEqual(x.item(), 3)

        x = mx.array([1, 2, 3])
        y = 2 * x
        s1 = mx.async_eval(y)
        z = 2 * y
        s2 = mx.async_eval(z)
        s2.wait()
        self.assertTrue(mx.array_equal(y, mx.array([2, 4, 6])))
        self.assertTrue(mx.array_equal(z, mx.array([4, 8, 12])))


if __name__ == "__main__":
    unittest.main()
