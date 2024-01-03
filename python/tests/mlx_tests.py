# Copyright © 2023 Apple Inc.

import os
import unittest
from typing import Any, Callable, List, Tuple

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

    def assertCmpNumpy(
        self,
        shape: List[Tuple[int] | Any],
        mx_fn: Callable[..., mx.array],
        np_fn: Callable[..., np.array],
        atol=1e-2,
        rtol=1e-2,
        dtype=mx.float32,
        **kwargs,
    ):
        assert dtype != mx.bfloat16, "numpy does not support bfloat16"
        args = [
            mx.random.normal(s, dtype=dtype) if isinstance(s, Tuple) else s
            for s in shape
        ]
        mx_res = mx_fn(*args, **kwargs)
        np_res = np_fn(
            *[np.array(a) if isinstance(a, mx.array) else a for a in args], **kwargs
        )
        return self.assertEqualArray(mx_res, mx.array(np_res), atol=atol, rtol=rtol)

    def assertEqualArray(
        self,
        mx_res: mx.array,
        expected: mx.array,
        atol=1e-2,
        rtol=1e-2,
    ):
        assert tuple(mx_res.shape) == tuple(
            expected.shape
        ), f"shape mismatch expected={expected.shape} got={mx_res.shape}"
        assert (
            mx_res.dtype == expected.dtype
        ), f"dtype mismatch expected={expected.dtype} got={mx_res.dtype}"
        np.testing.assert_allclose(mx_res, expected, rtol=rtol, atol=atol)
