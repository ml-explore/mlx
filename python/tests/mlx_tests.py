# Copyright © 2023 Apple Inc.

import os

# Use regular fp32 precision for tests
os.environ["MLX_ENABLE_TF32"] = "0"

import platform
import unittest
from typing import Any, Callable, List, Tuple, Union

import mlx.core as mx
import numpy as np


class MLXTestRunner(unittest.TestProgram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def createTests(self, *args, **kwargs):
        super().createTests(*args, **kwargs)

        # Asume CUDA backend in this case
        device = os.getenv("DEVICE", None)
        if device is not None:
            device = getattr(mx, device)
        else:
            device = mx.default_device()

        if not (device == mx.gpu and not mx.metal.is_available()):
            return

        from cuda_skip import cuda_skip

        filtered_suite = unittest.TestSuite()

        def filter_and_add(t):
            if isinstance(t, unittest.TestSuite):
                for sub_t in t:
                    filter_and_add(sub_t)
            else:
                t_id = ".".join(t.id().split(".")[-2:])
                if t_id in cuda_skip:
                    print(f"Skipping {t_id}")
                else:
                    filtered_suite.addTest(t)

        filter_and_add(self.test)
        self.test = filtered_suite


class MLXTestCase(unittest.TestCase):
    @property
    def is_apple_silicon(self):
        return platform.machine() == "arm64" and platform.system() == "Darwin"

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
            return
        elif not isinstance(mx_res, mx.array):
            mx_res = mx.array(mx_res)
        elif not isinstance(expected, mx.array):
            expected = mx.array(expected)
        self.assertTrue(mx.allclose(mx_res, expected, rtol=rtol, atol=atol))
