# Copyright © 2023 Apple Inc.

import math
import unittest
from itertools import permutations

import mlx.core as mx
import mlx_tests
import numpy as np

try:
    import torch
    import torch.nn.functional as F

    has_torch = True
except ImportError as e:
    has_torch = False


class TestConv(mlx_tests.MLXTestCase):
    def test_numpy_conv(self):
        for dtype in (
            "float16",
            "float32",
        ):
            np_dtype = getattr(np, dtype)
            for M, N, mode in (
                (1, 1, "full"),
                (25, 5, "full"),
                (24, 5, "same"),
                (24, 4, "same"),
                (24, 4, "valid"),
                (4, 24, "full"),
                (5, 25, "same"),
                (4, 25, "valid"),
            ):
                with self.subTest(dtype=dtype, M=M, N=N, mode=mode):
                    atol = 1e-6 if dtype == "float32" else 1e-5
                    a_np = np.random.rand(M).astype(np_dtype)
                    v_np = np.random.rand(N).astype(np_dtype)
                    a_mx = mx.array(a_np)
                    v_mx = mx.array(v_np)

                    c_np = np.convolve(a_np, v_np, mode=mode)
                    c_mx = mx.convolve(a_mx, v_mx, mode=mode)

                    self.assertListEqual(list(c_mx.shape), list(c_np.shape))
                    self.assertTrue(np.allclose(c_mx, c_np, atol=atol))

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_conv_1D(self):
        def run_conv1D(
            N,
            C,
            O,
            iH,
            kH,
            stride,
            padding,
            dilation=1,
            groups=1,
            dtype="float32",
            atol=1e-5,
        ):
            with self.subTest(
                dtype=dtype,
                N=N,
                C=C,
                O=O,
                iH=iH,
                kH=kH,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ):
                np_dtype = getattr(np, dtype)
                np.random.seed(0)
                in_np = np.random.normal(0, 1.0 / C, (N, iH, C)).astype(np_dtype)
                wt_np = np.random.normal(0, 1.0 / C, (O, kH, C)).astype(np_dtype)

                in_mx, wt_mx = map(mx.array, (in_np, wt_np))
                in_pt, wt_pt = map(
                    lambda x: torch.from_numpy(x.transpose(0, 2, 1)), (in_np, wt_np)
                )

                out_mx = mx.conv1d(
                    in_mx,
                    wt_mx,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                out_pt = torch.conv1d(
                    in_pt,
                    wt_pt,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                out_pt = torch.transpose(out_pt, 2, 1)

                self.assertListEqual(list(out_pt.shape), out_mx.shape)
                self.assertTrue(np.allclose(out_pt.numpy(), out_mx, atol=atol))

        for dtype in ("float32",):
            for N, C, O in (
                (1, 1, 1),
                (1, 6, 1),
                (1, 1, 6),
                (4, 32, 64),
            ):
                for iH, kH, stride, padding in (
                    (1, 1, 1, 0),
                    (3, 3, 1, 0),
                    (31, 5, 5, 2),
                ):
                    run_conv1D(N, C, O, iH, kH, stride, padding, dtype=dtype)

        # Strided inputs tests
        for tpose_in, tpose_wt in (
            ((0, 2, 1), (0, 1, 2)),
            ((0, 2, 1), (0, 2, 1)),
        ):
            with self.subTest(name="strided", tpose_in=tpose_in, tpose_wt=tpose_wt):
                in_np = np.random.normal(0, 1.0 / 16, (16, 16, 16)).astype(np.float32)
                wt_np = np.random.normal(0, 1.0 / 16, (16, 16, 16)).astype(np.float32)

                in_mx, wt_mx = map(mx.array, (in_np, wt_np))
                in_mx_t = mx.transpose(in_mx, tpose_in)
                wt_mx_t = mx.transpose(wt_mx, tpose_wt)
                out_mx = mx.conv1d(in_mx_t, wt_mx_t)

                in_pt, wt_pt = map(
                    lambda x: torch.from_numpy(x.transpose(0, 2, 1)),
                    (in_np.transpose(tpose_in), wt_np.transpose(tpose_wt)),
                )

                out_pt = torch.conv1d(in_pt, wt_pt)
                out_pt = torch.transpose(out_pt, 2, 1)

                self.assertListEqual(list(out_pt.shape), out_mx.shape)
                self.assertTrue(np.allclose(out_pt.numpy(), out_mx, atol=1e-5))

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_conv_1D_grad(self):
        def run_conv1D_grad(
            N,
            C,
            O,
            iH,
            kH,
            stride,
            padding,
            dilation=1,
            groups=1,
            dtype="float32",
            atol=1e-5,
        ):
            with self.subTest(
                dtype=dtype,
                N=N,
                C=C,
                O=O,
                iH=iH,
                kH=kH,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ):
                np_dtype = getattr(np, dtype)
                np.random.seed(0)
                oH = 1 + ((iH + 2 * padding - dilation * (kH - 1) - 1) // stride)

                in_np = np.random.normal(0, 1.0 / C, (N, iH, C)).astype(np_dtype)
                wt_np = np.random.normal(0, 1.0 / C, (O, kH, C)).astype(np_dtype)
                ct_np = np.random.normal(0, 1.0 / C, (N, oH, O)).astype(np_dtype)

                in_mx, wt_mx, ct_mx = map(mx.array, (in_np, wt_np, ct_np))
                in_pt, wt_pt, ct_pt = map(
                    lambda x: torch.from_numpy(x.transpose(0, 2, 1)),
                    (in_np, wt_np, ct_np),
                )

                def f(a, b):
                    return mx.conv1d(
                        a,
                        b,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                    )

                _, outs_mx = mx.vjp(
                    f,
                    [
                        in_mx,
                        wt_mx,
                    ],
                    [
                        ct_mx,
                    ],
                )
                pt_grad_in = F.grad.conv1d_input(
                    in_pt.shape,
                    wt_pt,
                    ct_pt,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                pt_grad_wt = F.grad.conv1d_weight(
                    in_pt,
                    wt_pt.shape,
                    ct_pt,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                pt_grad_in = torch.transpose(pt_grad_in, 2, 1).numpy()
                pt_grad_wt = torch.transpose(pt_grad_wt, 2, 1).numpy()

                mx_grad_in, mx_grad_wt = outs_mx

                self.assertListEqual(list(pt_grad_in.shape), mx_grad_in.shape)
                self.assertListEqual(list(in_mx.shape), mx_grad_in.shape)
                self.assertTrue(np.allclose(pt_grad_in, mx_grad_in, atol=atol))

                self.assertListEqual(list(pt_grad_wt.shape), mx_grad_wt.shape)
                self.assertListEqual(list(wt_mx.shape), mx_grad_wt.shape)
                self.assertTrue(np.allclose(pt_grad_wt, mx_grad_wt, atol=atol))

        for dtype in ("float32",):
            for N, C, O in (
                (1, 1, 1),
                (1, 6, 1),
                (1, 1, 6),
                (4, 32, 64),
            ):
                for iH, kH, stride, padding in (
                    (1, 1, 1, 0),
                    (3, 3, 1, 0),
                    (31, 5, 5, 2),
                ):
                    run_conv1D_grad(N, C, O, iH, kH, stride, padding, dtype=dtype)

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_conv_2D(self):
        def run_conv2D(
            N,
            C,
            O,
            idim,
            kdim,
            stride,
            padding,
            dilation=(1, 1),
            groups=1,
            dtype="float32",
            atol=1e-5,
        ):
            with self.subTest(
                dtype=dtype,
                N=N,
                C=C,
                O=O,
                idim=idim,
                kdim=kdim,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ):
                np_dtype = getattr(np, dtype)
                np.random.seed(0)
                iH, iW = idim
                kH, kW = kdim
                scale = 1.0 / math.sqrt(kH * kW * C)
                in_np = np.random.normal(0.0, scale, (N, iH, iW, C)).astype(np_dtype)
                wt_np = np.random.normal(0.0, 1.0, (O, kH, kW, C)).astype(np_dtype)

                in_mx, wt_mx = map(mx.array, (in_np, wt_np))
                in_pt, wt_pt = map(
                    lambda x: torch.from_numpy(x.transpose(0, 3, 1, 2)).to("cpu"),
                    (in_np, wt_np),
                )

                out_mx = mx.conv2d(
                    in_mx,
                    wt_mx,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                out_pt = torch.conv2d(
                    in_pt,
                    wt_pt,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                out_pt = torch.permute(out_pt, (0, 2, 3, 1)).numpy(force=True)

                self.assertListEqual(list(out_pt.shape), list(out_mx.shape))
                self.assertTrue(np.allclose(out_pt, out_mx, atol=atol))

        for dtype in ("float32",):
            for N, C, O in (
                (1, 1, 1),
                (1, 6, 1),
                (1, 1, 6),
                (4, 32, 64),
            ):
                for idim, kdim, stride, padding in (
                    ((1, 1), (1, 1), (1, 1), (0, 0)),
                    ((3, 3), (3, 1), (1, 1), (0, 0)),
                    ((31, 31), (5, 5), (5, 5), (2, 2)),
                ):
                    run_conv2D(N, C, O, idim, kdim, stride, padding, dtype=dtype)

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_conv_2D_grad(self):
        def run_conv2D_grad(
            N,
            C,
            O,
            idim,
            kdim,
            stride,
            padding,
            dilation=(1, 1),
            groups=1,
            dtype="float32",
            atol=1e-5,
        ):
            with self.subTest(
                dtype=dtype,
                N=N,
                C=C,
                O=O,
                idim=idim,
                kdim=kdim,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
            ):
                np_dtype = getattr(np, dtype)
                np.random.seed(0)
                iH, iW = idim
                kH, kW = kdim
                scale = 1.0 / math.sqrt(kH * kW * C)

                oH = 1 + (
                    (iH + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0]
                )
                oW = 1 + (
                    (iW + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1]
                )

                in_np = np.random.normal(0.0, scale, (N, iH, iW, C)).astype(np_dtype)
                wt_np = np.random.normal(0.0, scale, (O, kH, kW, C)).astype(np_dtype)
                ct_np = np.random.normal(0.0, scale, (N, oH, oW, O)).astype(np_dtype)

                in_mx, wt_mx, ct_mx = map(mx.array, (in_np, wt_np, ct_np))
                in_pt, wt_pt, ct_pt = map(
                    lambda x: torch.from_numpy(x.transpose(0, 3, 1, 2)).to("cpu"),
                    (in_np, wt_np, ct_np),
                )

                def f(a, b):
                    return mx.conv2d(
                        a,
                        b,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                    )

                _, outs_mx = mx.vjp(
                    f,
                    [
                        in_mx,
                        wt_mx,
                    ],
                    [
                        ct_mx,
                    ],
                )
                pt_grad_in = F.grad.conv1d_input(
                    in_pt.shape,
                    wt_pt,
                    ct_pt,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                pt_grad_wt = F.grad.conv1d_weight(
                    in_pt,
                    wt_pt.shape,
                    ct_pt,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                pt_grad_in = torch.permute(pt_grad_in, (0, 2, 3, 1)).numpy()
                pt_grad_wt = torch.permute(pt_grad_wt, (0, 2, 3, 1)).numpy()

                mx_grad_in, mx_grad_wt = outs_mx

                self.assertListEqual(list(pt_grad_in.shape), mx_grad_in.shape)
                self.assertListEqual(list(in_mx.shape), mx_grad_in.shape)
                self.assertTrue(np.allclose(pt_grad_in, mx_grad_in, atol=atol))

                self.assertListEqual(list(pt_grad_wt.shape), mx_grad_wt.shape)
                self.assertListEqual(list(wt_mx.shape), mx_grad_wt.shape)
                self.assertTrue(np.allclose(pt_grad_wt, mx_grad_wt, atol=atol))

        for dtype in ("float32",):
            for N, C, O in (
                (1, 1, 1),
                (1, 6, 1),
                (1, 1, 6),
                (4, 32, 64),
            ):
                for idim, kdim, stride, padding in (
                    ((1, 1), (1, 1), (1, 1), (0, 0)),
                    ((3, 3), (3, 1), (1, 1), (0, 0)),
                    ((31, 31), (5, 5), (5, 5), (2, 2)),
                ):
                    run_conv2D_grad(N, C, O, idim, kdim, stride, padding, dtype=dtype)


if __name__ == "__main__":
    unittest.main()
