# Copyright © 2023-2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests


class TestExt(mlx_tests.MLXTestCase):
    def test_rms_norm(self):
        size = 4
        eps = 1e-5
        x = mx.random.uniform(shape=(2, 2, 4))
        weight = mx.random.uniform(shape=(4,))

        def orig_rms(x, weight, eps, precise):
            orig_type = x.dtype
            if precise:
                x = x.astype(mx.float32)
            x = x * mx.rsqrt(x.square().mean(-1, keepdims=True) + eps)
            return weight * x.astype(orig_type)

        for precise in [True, False]:
            expected = orig_rms(x, weight, eps, precise)
            out = mx.ext.rms_norm(x, weight, eps, precise)
            self.assertTrue(mx.allclose(expected, out))


if __name__ == "__main__":
    unittest.main()
