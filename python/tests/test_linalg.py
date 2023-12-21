# Copyright © 2023 Apple Inc.

import itertools
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestLinalg(mlx_tests.MLXTestCase):
    def test_norm(self):
        def check_mx_np(a_mx, a_np):
            self.assertTrue(np.allclose(a_np, a_mx, atol=1e-5, rtol=1e-6))

        x_mx = mx.arange(18).reshape((2, 3, 3))
        x_np = np.arange(18).reshape((2, 3, 3))

        for num_axes in range(1, 3):
            for axis in itertools.combinations(range(3), num_axes):
                if num_axes == 1:
                    ords = [None, 0.5, 0, 1, 2, 3, -1, 1]
                else:
                    ords = [None, "fro", -1, 1]
                for o in ords:
                    for keepdims in [True, False]:
                        if o:
                            out_np = np.linalg.norm(
                                x_np, ord=o, axis=axis, keepdims=keepdims
                            )
                            out_mx = mx.linalg.norm(
                                x_mx, ord=o, axis=axis, keepdims=keepdims
                            )
                        else:
                            out_np = np.linalg.norm(x_np, axis=axis, keepdims=keepdims)
                            out_mx = mx.linalg.norm(x_mx, axis=axis, keepdims=keepdims)
                        assert np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
