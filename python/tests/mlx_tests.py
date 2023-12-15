# Copyright © 2023 Apple Inc.

import os
import unittest

import mlx.core as mx
import numpy as np


class MLXTestCase(unittest.TestCase):
    def setUp(self):
        self.default = mx.default_device()
        device = os.getenv("DEVICE", None)
        if device is not None:
            device = getattr(mx, device)
            mx.set_default_device(device)

    def tearDown(self):
        mx.set_default_device(self.default)

    def assertEqualArray(
        self,
        mx_res: mx.array,
        expected: mx.array,
        atol=1e-2,
        rtol=1e-2,
        **kwargs,
    ):
        assert tuple(mx_res.shape) == tuple(
            expected.shape
        ), f"shape mismatch expected={expected.shape} got={mx_res.shape}"
        assert (
            mx_res.dtype == expected.dtype
        ), f"dtype mismatch expected={expected.dtype} got={mx_res.dtype}"
        np.testing.assert_allclose(mx_res, expected, rtol=rtol, atol=atol)
