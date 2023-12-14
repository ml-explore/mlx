# Copyright © 2023 Apple Inc.

import os
import unittest
from typing import Callable, List, Tuple, Union

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
        args: List[Union[mx.array, float, int]],
        mlx_func: Callable[..., mx.array],
        expected: mx.array,
        atol=1e-2,
        rtol=1e-2,
    ):
        mx_res = mlx_func(*args)
        assert tuple(mx_res.shape) == tuple(expected.shape), "shape mismatch"
        assert mx_res.dtype == expected.dtype, "dtype mismatch"
        np.testing.assert_allclose(mx_res, expected, rtol=rtol, atol=atol)
