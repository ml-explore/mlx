# Copyright © 2023 Apple Inc.

import os
import platform
import unittest
from typing import Any, Callable, List, Tuple, Union

import mlx.core as mx
import numpy as np


class MLXTestCase(unittest.TestCase):
    @property
    def is_linux(self):
        return platform.system() == "Linux"

    def setUp(self):
        self.default = mx.default_device()
        device = os.getenv("DEVICE", None)
        if device is not None:
            device = getattr(mx, device)
            mx.set_default_device(device)

    def tearDown(self):
        mx.set_default_device(self.default)

    # Note if a tuple is passed into args, it will be considered a shape request and convert to a mx.random.normal with the shape matching the tuple
    def assertCmpNumpy(
        self,
        args: List[Union[Tuple[int], Any]],
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
            for s in args
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
        self.assertEqual(
            tuple(mx_res.shape),
            tuple(expected.shape),
            msg=f"shape mismatch expected={expected.shape} got={mx_res.shape}",
        )
        self.assertEqual(
            mx_res.dtype,
            expected.dtype,
            msg=f"dtype mismatch expected={expected.dtype} got={mx_res.dtype}",
        )
        if not isinstance(mx_res, mx.array) and not isinstance(expected, mx.array):
            np.testing.assert_allclose(mx_res, expected, rtol=rtol, atol=atol)
        elif not isinstance(mx_res, mx.array):
            mx_res = mx.array(mx_res)
            self.assertTrue(mx.allclose(mx_res, expected, rtol=rtol, atol=atol))
        elif not isinstance(expected, mx.array):
            expected = mx.array(expected)
            self.assertTrue(mx.allclose(mx_res, expected, rtol=rtol, atol=atol))
        else:
            self.assertTrue(mx.allclose(mx_res, expected, rtol=rtol, atol=atol))
