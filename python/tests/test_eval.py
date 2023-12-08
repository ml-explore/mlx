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
        def fun(x, retain_graph):
            y = 3 * x
            mx.eval(y, retain_graph=retain_graph)
            return 2 * y

        dfun_dx_1 = mx.grad(partial(fun, retain_graph=False))
        dfun_dx_2 = mx.grad(partial(fun, retain_graph=True))

        with self.assertRaises(ValueError):
            dfun_dx_1(mx.array(1.0))

        y = dfun_dx_2(mx.array(1.0))
        self.assertEqual(y.item(), 6.0)


if __name__ == "__main__":
    unittest.main()
