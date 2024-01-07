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


if __name__ == "__main__":
    unittest.main()
