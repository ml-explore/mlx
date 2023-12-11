# Copyright © 2023 Apple Inc.

import os
import unittest
from typing import Callable, List, Tuple

import mlx.core as mx
import numpy as np
import torch

dtype_map = {
    torch.half: mx.float16,
    torch.float16: mx.float16,
    torch.float32: mx.float32,
    torch.int32: mx.int32,
    torch.int64: mx.int64,
}


class MLXTestCase(unittest.TestCase):
    def setUp(self):
        self.default = mx.default_device()
        device = os.getenv("DEVICE", None)
        if device is not None:
            device = getattr(mx, device)
            mx.set_default_device(device)

    def tearDown(self):
        mx.set_default_device(self.default)

    def compare_dtype(self, mlx_dtype: mx.Dtype, torch_dtype: torch.dtype):
        assert torch_dtype in dtype_map, "dtype not supported"
        assert (
            dtype_map[torch_dtype] == mlx_dtype
        ), "dtype mismatch got {mlx_dtype} expected {torch_dtype}"

    def compare_torch(
        self,
        shapes: List[Tuple[int]],
        mlx_func: Callable[..., mx.array],
        torch_func: Callable[..., torch.Tensor],
    ):
        mlx_arrs = [mx.random.normal(shape) for shape in shapes]
        torch_arrs = [torch.from_numpy(np.copy(x)) for x in mlx_arrs]
        mx_res = mlx_func(*mlx_arrs)
        torch_res = torch_func(*torch_arrs)
        assert tuple(mx_res.shape) == tuple(torch_res.shape)
        self.compare_dtype(mx_res.dtype, torch_res.dtype)
        np.testing.assert_allclose(mx_res, torch_res.numpy(), rtol=1, atol=1)
