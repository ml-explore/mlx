# Copyright © 2023-2024 Apple Inc.

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

                    self.assertEqual(c_mx.shape, c_np.shape)
                    self.assertTrue(np.allclose(c_mx, c_np, atol=atol))

    def test_conv_1d_groups_flipped(self):
        x = mx.broadcast_to(mx.arange(5).astype(mx.float32), (2, 5)).T
        w = mx.broadcast_to(mx.arange(4).astype(mx.float32), (2, 4))
        out = mx.conv_general(x[None], w[..., None], flip=True, groups=2)
        expected = mx.array([4.0, 4.0, 10.0, 10.0]).reshape(1, 2, 2)
        self.assertTrue(mx.allclose(out, expected))

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
                wt_np = np.random.normal(0, 1.0 / C, (O, kH, int(C / groups))).astype(
                    np_dtype
                )

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

                self.assertEqual(out_pt.shape, out_mx.shape)
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

        # Groups tests
        N, C, O = (4, 32, 64)
        for iH, kH, stride, padding in (
            (1, 1, 1, 0),
            (3, 3, 1, 0),
            (31, 5, 5, 2),
        ):
            for group in (1, 2, 4, 8, 16, 32):
                run_conv1D(N, C, O, iH, kH, stride, padding, groups=group, dtype=dtype)

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

                self.assertEqual(out_pt.shape, out_mx.shape)
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

                self.assertEqual(pt_grad_in.shape, mx_grad_in.shape)
                self.assertEqual(in_mx.shape, mx_grad_in.shape)
                self.assertTrue(np.allclose(pt_grad_in, mx_grad_in, atol=atol))

                self.assertEqual(pt_grad_wt.shape, mx_grad_wt.shape)
                self.assertEqual(wt_mx.shape, mx_grad_wt.shape)
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
                np.random.seed(0)
                iH, iW = idim
                kH, kW = kdim
                scale = 1.0 / math.sqrt(kH * kW * C)
                in_np = np.random.normal(0.0, scale, (N, iH, iW, C))
                wt_np = np.random.normal(0.0, 1.0, (O, kH, kW, int(C / groups)))

                mx_dtype = getattr(mx, dtype)
                torch_dtype = getattr(torch, dtype)
                in_mx, wt_mx = map(
                    lambda x: mx.array(x).astype(mx_dtype), (in_np, wt_np)
                )
                in_pt, wt_pt = map(
                    lambda x: torch.from_numpy(x.transpose(0, 3, 1, 2))
                    .to("cpu")
                    .to(torch_dtype),
                    (in_np, wt_np),
                )

                out_mx = mx.conv2d(
                    in_mx,
                    wt_mx,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                ).astype(mx.float32)
                out_pt = torch.conv2d(
                    in_pt,
                    wt_pt,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                out_pt = (
                    torch.permute(out_pt, (0, 2, 3, 1))
                    .to(torch.float32)
                    .numpy(force=True)
                )

                self.assertEqual(out_pt.shape, out_mx.shape)
                if dtype == "bfloat16":
                    atol, rtol = 1e-1, 1e-3
                else:
                    atol, rtol = 1e-5, 1e-6
                self.assertTrue(np.allclose(out_pt, out_mx, atol=atol))

        for dtype in ("float32", "bfloat16"):
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

            # Groups tests
            N, C, O = (4, 32, 64)
            for idim, kdim, stride, padding in (
                ((1, 1), (1, 1), (1, 1), (0, 0)),
                ((3, 3), (3, 1), (1, 1), (0, 0)),
                ((31, 31), (5, 5), (5, 5), (2, 2)),
            ):
                for group in (1, 2, 4, 8, 16, 32):
                    run_conv2D(
                        N, C, O, idim, kdim, stride, padding, groups=group, dtype=dtype
                    )

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
                    [in_mx, wt_mx],
                    [ct_mx],
                )
                pt_grad_in = F.grad.conv2d_input(
                    in_pt.shape,
                    wt_pt,
                    ct_pt,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                pt_grad_wt = F.grad.conv2d_weight(
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

                self.assertEqual(pt_grad_in.shape, mx_grad_in.shape)
                self.assertEqual(in_mx.shape, mx_grad_in.shape)
                self.assertTrue(np.allclose(pt_grad_in, mx_grad_in, atol=atol))

                self.assertEqual(pt_grad_wt.shape, mx_grad_wt.shape)
                self.assertEqual(wt_mx.shape, mx_grad_wt.shape)
                self.assertTrue(np.allclose(pt_grad_wt, mx_grad_wt, atol=atol))

        for dtype in ("float32",):
            for N, C, O in ((1, 1, 1), (1, 6, 1), (1, 1, 6), (4, 32, 64), (4, 16, 32)):
                for idim, kdim, stride, padding, dilation in (
                    ((1, 1), (1, 1), (1, 1), (0, 0), (1, 1)),
                    ((3, 3), (3, 1), (1, 1), (0, 0), (1, 1)),
                    ((31, 31), (5, 5), (5, 5), (2, 2), (1, 1)),
                    ((32, 32), (3, 3), (2, 2), (1, 1), (1, 1)),
                    ((31, 31), (5, 5), (5, 5), (2, 2), (3, 2)),
                    ((32, 32), (3, 3), (2, 2), (1, 1), (3, 2)),
                ):
                    run_conv2D_grad(
                        N, C, O, idim, kdim, stride, padding, dilation, dtype=dtype
                    )

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_conv_3D(self):
        def run_conv3D(
            N,
            C,
            O,
            idim,
            kdim,
            stride,
            padding,
            dilation=(1, 1, 1),
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
                iD, iH, iW = idim
                kD, kH, kW = kdim
                scale = 1.0 / math.sqrt(kD * kH * kW * C)
                in_np = np.random.normal(0.0, scale, (N, iD, iH, iW, C)).astype(
                    np_dtype
                )
                wt_np = np.random.normal(0.0, 1.0, (O, kD, kH, kW, C)).astype(np_dtype)

                in_mx, wt_mx = map(mx.array, (in_np, wt_np))
                in_pt, wt_pt = map(
                    lambda x: torch.from_numpy(x.transpose(0, 4, 1, 2, 3)).to("cpu"),
                    (in_np, wt_np),
                )

                out_mx = mx.conv3d(
                    in_mx,
                    wt_mx,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                out_pt = torch.conv3d(
                    in_pt,
                    wt_pt,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                out_pt = torch.permute(out_pt, (0, 2, 3, 4, 1)).numpy(force=True)

                self.assertEqual(out_pt.shape, out_mx.shape)
                self.assertTrue(np.allclose(out_pt, out_mx, atol=atol))

        for dtype in ("float32",):
            for N, C, O in (
                (1, 1, 1),
                (1, 6, 1),
                (1, 1, 6),
                (4, 16, 32),
            ):
                for idim, kdim, stride, padding in (
                    ((1, 1, 1), (1, 1, 1), (1, 1, 1), (0, 0, 0)),
                    ((3, 3, 3), (3, 1, 1), (1, 1, 1), (0, 0, 0)),
                    ((31, 31, 31), (5, 5, 5), (5, 5, 5), (2, 2, 2)),
                ):
                    run_conv3D(N, C, O, idim, kdim, stride, padding, dtype=dtype)

            N, C, O = (2, 4, 4)
            idim, kdim, stride, padding = (6, 6, 6), (3, 1, 1), (1, 1, 1), (0, 0, 0)
            run_conv3D(
                N, C, O, idim, kdim, stride, padding, dilation=(2, 2, 2), dtype=dtype
            )

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_conv_3D_grad(self):
        def run_conv3D_grad(
            N,
            C,
            O,
            idim,
            kdim,
            stride,
            padding,
            dilation=(1, 1, 1),
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
                iD, iH, iW = idim
                kD, kH, kW = kdim
                scale = 1.0 / math.sqrt(kD * kH * kW * C)

                oD = 1 + (
                    (iD + 2 * padding[0] - dilation[0] * (kD - 1) - 1) // stride[0]
                )
                oH = 1 + (
                    (iH + 2 * padding[1] - dilation[1] * (kH - 1) - 1) // stride[1]
                )
                oW = 1 + (
                    (iW + 2 * padding[2] - dilation[2] * (kW - 1) - 1) // stride[2]
                )

                in_np = np.random.normal(0.0, scale, (N, iD, iH, iW, C)).astype(
                    np_dtype
                )
                wt_np = np.random.normal(0.0, scale, (O, kD, kH, kW, C)).astype(
                    np_dtype
                )
                ct_np = np.random.normal(0.0, scale, (N, oD, oH, oW, O)).astype(
                    np_dtype
                )

                in_mx, wt_mx, ct_mx = map(mx.array, (in_np, wt_np, ct_np))
                in_pt, wt_pt, ct_pt = map(
                    lambda x: torch.from_numpy(x.transpose(0, 4, 1, 2, 3)).to("cpu"),
                    (in_np, wt_np, ct_np),
                )

                def f(a, b):
                    return mx.conv3d(
                        a,
                        b,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                    )

                _, outs_mx = mx.vjp(
                    f,
                    [in_mx, wt_mx],
                    [ct_mx],
                )
                pt_grad_in = F.grad.conv3d_input(
                    in_pt.shape,
                    wt_pt,
                    ct_pt,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                pt_grad_wt = F.grad.conv3d_weight(
                    in_pt,
                    wt_pt.shape,
                    ct_pt,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                pt_grad_in = torch.permute(pt_grad_in, (0, 2, 3, 4, 1)).numpy()
                pt_grad_wt = torch.permute(pt_grad_wt, (0, 2, 3, 4, 1)).numpy()

                mx_grad_in, mx_grad_wt = outs_mx

                self.assertEqual(pt_grad_in.shape, mx_grad_in.shape)
                self.assertEqual(in_mx.shape, mx_grad_in.shape)
                self.assertTrue(np.allclose(pt_grad_in, mx_grad_in, atol=atol))

                self.assertEqual(pt_grad_wt.shape, mx_grad_wt.shape)
                self.assertEqual(wt_mx.shape, mx_grad_wt.shape)
                self.assertTrue(np.allclose(pt_grad_wt, mx_grad_wt, atol=atol))

        for dtype in ("float32",):
            for N, C, O in ((1, 1, 1), (1, 6, 1), (1, 1, 6), (4, 16, 32), (4, 8, 16)):
                for idim, kdim, stride, padding, dilation in (
                    ((1, 1, 1), (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)),
                    ((3, 3, 3), (3, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)),
                    ((15, 15, 15), (5, 5, 5), (5, 5, 5), (2, 2, 2), (1, 1, 1)),
                    ((16, 16, 16), (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1)),
                    ((15, 15, 15), (5, 5, 5), (5, 5, 5), (2, 2, 2), (3, 2, 2)),
                    ((16, 16, 16), (3, 3, 3), (2, 2, 2), (1, 1, 1), (3, 2, 2)),
                ):
                    run_conv3D_grad(
                        N, C, O, idim, kdim, stride, padding, dilation, dtype=dtype
                    )

    def __conv_general_test(
        self,
        in_shape,
        wt_shape,
        stride=1,
        padding=0,
        kernel_dilation=1,
        input_dilation=1,
        groups=1,
        flip=False,
        np_dtype=np.float32,
        atol=1e-5,
    ):
        with self.subTest(
            in_shape=in_shape,
            wt_shape=wt_shape,
            stride=stride,
            padding=padding,
            kernel_dilation=kernel_dilation,
            input_dilation=input_dilation,
            groups=groups,
            flip=flip,
            np_dtype=np_dtype,
        ):
            np.random.seed(0)
            scale = 1.0 / math.sqrt(np.prod(wt_shape[1:]))
            scale = min(0.3, scale)
            in_np = np.random.normal(0, scale, in_shape).astype(np_dtype)
            wt_np = np.random.normal(0, scale, wt_shape).astype(np_dtype)

            in_mx, wt_mx = map(mx.array, (in_np, wt_np))

            in_pt, wt_pt = map(
                lambda x: torch.from_numpy(np.moveaxis(x, -1, 1)).to("cpu"),
                (in_np, wt_np),
            )

            out_mx = mx.conv_general(
                in_mx,
                wt_mx,
                stride=stride,
                padding=padding,
                kernel_dilation=kernel_dilation,
                input_dilation=input_dilation,
                groups=groups,
                flip=flip,
            )

            def conv_general_pt(
                inp, wt, stride, padding, kernel_dilation, input_dilation, groups, flip
            ):
                C = inp.size()[1]
                ndim = inp.ndim - 2
                map_ints = lambda x: [x] * ndim if isinstance(x, int) else x

                stride, padding, kernel_dilation, input_dilation = map(
                    map_ints, (stride, padding, kernel_dilation, input_dilation)
                )

                torch_convt_list = (
                    F.conv_transpose1d,
                    F.conv_transpose2d,
                    F.conv_transpose3d,
                )
                torch_conv_list = (F.conv1d, F.conv2d, F.conv3d)

                conv_f = torch_conv_list[ndim - 1]
                convt_f = torch_convt_list[ndim - 1]

                if flip:
                    wt = torch.flip(wt, tuple(np.arange(2, wt.ndim)))

                if not np.all(input_dilation == 1):
                    ones = torch.ones(
                        [C]
                        + [
                            1,
                        ]
                        * (ndim + 1)
                    ).to(inp.dtype)
                    inp = convt_f(inp, ones, stride=input_dilation, groups=C)

                return conv_f(
                    inp,
                    wt,
                    stride=stride,
                    padding=padding,
                    dilation=kernel_dilation,
                    groups=groups,
                )

            out_pt = conv_general_pt(
                in_pt,
                wt_pt,
                stride=stride,
                padding=padding,
                kernel_dilation=kernel_dilation,
                input_dilation=input_dilation,
                groups=groups,
                flip=flip,
            )

            out_pt = np.moveaxis(out_pt.numpy(), 1, -1)

            self.assertEqual(out_mx.shape, out_pt.shape)
            self.assertTrue(np.allclose(out_mx, out_pt, atol=atol))

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_conv_general(self):
        in_shape = (2, 32, 32, 16)
        wt_shape = (32, 5, 5, 16)
        stride = (1, 1)
        padding = (2, 2)
        kernel_dilation = (2, 3)
        input_dilation = (1, 1)
        flip = False

        self.__conv_general_test(
            in_shape,
            wt_shape,
            stride,
            padding,
            kernel_dilation,
            input_dilation,
            flip=flip,
        )

        in_shape = (2, 32, 32, 16)
        wt_shape = (32, 5, 10, 16)
        stride = (2, 3)
        padding = (0, 0)
        kernel_dilation = (3, 2)
        input_dilation = (2, 4)
        flip = False

        self.__conv_general_test(
            in_shape,
            wt_shape,
            stride,
            padding,
            kernel_dilation,
            input_dilation,
            flip=flip,
        )

        in_shape = (2, 32, 32, 16)
        wt_shape = (32, 5, 10, 16)
        stride = (2, 2)
        padding = (3, 2)
        kernel_dilation = (3, 2)
        input_dilation = (2, 4)
        flip = False

        self.__conv_general_test(
            in_shape,
            wt_shape,
            stride,
            padding,
            kernel_dilation,
            input_dilation,
            flip=flip,
        )

        in_shape = (2, 32, 32, 16)
        wt_shape = (32, 5, 10, 16)
        stride = (2, 3)
        padding = (3, 2)
        kernel_dilation = (3, 2)
        input_dilation = (2, 5)
        flip = False

        self.__conv_general_test(
            in_shape,
            wt_shape,
            stride,
            padding,
            kernel_dilation,
            input_dilation,
            flip=flip,
        )

        in_shape = (2, 32, 32, 16)
        wt_shape = (32, 5, 5, 16)
        stride = (2, 3)
        padding = (0, 0)
        kernel_dilation = (3, 1)
        input_dilation = (2, 5)
        flip = True

        self.__conv_general_test(
            in_shape,
            wt_shape,
            stride,
            padding,
            kernel_dilation,
            input_dilation,
            flip=flip,
        )

    def test_conv_general_flip_grad(self):
        for s in (1, 2):
            w = mx.random.normal(shape=(1, 2, 2, 1))
            x = mx.random.normal(shape=(1, 2, 2, 1))

            def conv_t(w):
                return mx.conv_general(
                    x,
                    w,
                    stride=1,
                    padding=(1, 1),
                    kernel_dilation=1,
                    input_dilation=s,
                    flip=True,
                )

            cotan = mx.random.normal(shape=(1, 2 + s, 2 + s, 1))

            dw = mx.vjp(conv_t, (w,), (cotan,))[1][0]

            x = x.squeeze()
            cotan = cotan.squeeze()
            dw = dw.squeeze()

            dw00 = (cotan[:-1:s, :-1:s] * x).sum()
            dw01 = (cotan[:-1:s, 1::s] * x).sum()
            dw10 = (cotan[1::s, :-1:s] * x).sum()
            dw11 = (cotan[1::s, 1::s] * x).sum()
            expected = mx.array([[dw00, dw01], [dw10, dw11]])
            self.assertTrue(mx.allclose(dw, expected, rtol=1e-5, atol=1e-5))

        # Test with input dilation
        inputs = mx.random.normal((1, 14, 14, 2))
        kernel = mx.random.normal((2, 7, 7, 2))

        def conv_flip(kernel):
            return mx.conv_general(
                inputs,
                kernel,
                stride=1,
                padding=([6, 6], [15, 15]),
                kernel_dilation=(1, 1),
                input_dilation=(16, 16),
                groups=1,
                flip=True,
            ).sum()

        def reverse_sequence(xs, axis=0):
            indices = mx.arange(xs.shape[axis] - 1, -1, -1)
            return mx.take(xs, indices, axis=axis)

        def conv_manual_flip(kernel):
            for ax in range(1, kernel.ndim - 1):
                kernel = reverse_sequence(kernel, axis=ax)
            return mx.conv_general(
                inputs,
                kernel,
                stride=1,
                padding=([6, 6], [15, 15]),
                kernel_dilation=(1, 1),
                input_dilation=(16, 16),
                groups=1,
                flip=False,
            ).sum()

        grad = mx.grad(conv_flip)(kernel)
        expected_grad = mx.grad(conv_manual_flip)(kernel)
        self.assertTrue(mx.allclose(grad, expected_grad))

    def test_conv_groups_grad(self):
        def fn(x, w):
            num_groups = x.shape[-1] // w.shape[-1]
            return mx.conv1d(x, w, groups=num_groups)

        def fn_gt(x, w):
            num_groups = x.shape[-1] // w.shape[-1]
            group_size = w.shape[-1]
            ws = w.reshape(num_groups, -1, *w.shape[1:]).split(num_groups)
            xs = x.reshape(*x.shape[:-1], num_groups, -1).split(num_groups, axis=-2)
            return mx.concatenate(
                [mx.conv_general(x.squeeze(-2), w.squeeze(0)) for x, w in zip(xs, ws)],
                axis=-1,
            )

        mx.random.seed(3)

        w = mx.random.normal(shape=(2, 3, 1))
        x = mx.random.normal(shape=(1, 5, 2))
        cotans = (mx.ones(shape=(1, 3, 2)),)
        grads = mx.vjp(fn, (x, w), cotans)[1]
        expected = mx.vjp(fn_gt, (x, w), cotans)[1]
        self.assertTrue(mx.allclose(expected[0], grads[0]))
        self.assertTrue(mx.allclose(expected[1], grads[1]))

        w = mx.random.normal(shape=(2, 3, 2))
        x = mx.random.normal(shape=(1, 5, 4))
        cotans = (mx.ones(shape=(1, 3, 2)),)
        grads = mx.vjp(fn, (x, w), cotans)[1]
        expected = mx.vjp(fn_gt, (x, w), cotans)[1]
        self.assertTrue(mx.allclose(expected[0], grads[0]))
        self.assertTrue(mx.allclose(expected[1], grads[1]))

        w = mx.random.normal(shape=(6, 3, 2))
        x = mx.random.normal(shape=(1, 5, 4))
        cotans = (mx.ones(shape=(1, 3, 6)),)
        grads = mx.vjp(fn, (x, w), cotans)[1]
        expected = mx.vjp(fn_gt, (x, w), cotans)[1]
        self.assertTrue(mx.allclose(expected[0], grads[0]))
        self.assertTrue(mx.allclose(expected[1], grads[1]))

        # Test 2D
        w = mx.random.normal(shape=(2, 3, 3, 1))
        x = mx.random.normal(shape=(1, 5, 5, 2))
        cotans = (mx.ones(shape=(1, 3, 3, 2)),)
        grads = mx.vjp(fn, (x, w), cotans)[1]
        expected = mx.vjp(fn_gt, (x, w), cotans)[1]
        self.assertTrue(mx.allclose(expected[0], grads[0]))
        self.assertTrue(mx.allclose(expected[1], grads[1]))

        # Test with flip
        def fn(x, w):
            num_groups = x.shape[-1] // w.shape[-1]
            return mx.conv_general(x, w, groups=num_groups, flip=True)

        def fn_gt(x, w):
            num_groups = x.shape[-1] // w.shape[-1]
            group_size = w.shape[-1]
            ws = w.reshape(num_groups, -1, *w.shape[1:]).split(num_groups)
            xs = x.reshape(*x.shape[:-1], num_groups, -1).split(num_groups, axis=-2)
            return mx.concatenate(
                [
                    mx.conv_general(x.squeeze(-2), w.squeeze(0), flip=True)
                    for x, w in zip(xs, ws)
                ],
                axis=-1,
            )

        w = mx.random.normal(shape=(2, 3, 1))
        x = mx.random.normal(shape=(1, 5, 2))
        cotans = (mx.ones(shape=(1, 3, 2)),)
        grads = mx.vjp(fn, (x, w), cotans)[1]
        expected = mx.vjp(fn_gt, (x, w), cotans)[1]
        self.assertTrue(mx.allclose(expected[0], grads[0]))
        self.assertTrue(mx.allclose(expected[1], grads[1]))

        w = mx.random.normal(shape=(2, 3, 2))
        x = mx.random.normal(shape=(1, 5, 4))
        cotans = (mx.ones(shape=(1, 3, 2)),)
        grads = mx.vjp(fn, (x, w), cotans)[1]
        expected = mx.vjp(fn_gt, (x, w), cotans)[1]
        self.assertTrue(mx.allclose(expected[0], grads[0]))
        self.assertTrue(mx.allclose(expected[1], grads[1]))

        # Test 2D
        w = mx.random.normal(shape=(2, 3, 3, 1))
        x = mx.random.normal(shape=(1, 5, 5, 2))
        cotans = (mx.ones(shape=(1, 3, 3, 2)),)
        grads = mx.vjp(fn, (x, w), cotans)[1]
        expected = mx.vjp(fn_gt, (x, w), cotans)[1]
        self.assertTrue(mx.allclose(expected[0], grads[0]))
        self.assertTrue(mx.allclose(expected[1], grads[1]))

    def test_repeated_conv(self):
        x = mx.random.normal((1, 3, 3, 320))
        w = mx.random.normal((320, 3, 3, 320))
        for i in range(8):
            y1 = mx.conv2d(x, w, (1, 1), (1, 1), (1, 1), 1)
            y2 = mx.conv2d(x, w, (1, 1), (1, 1), (1, 1), 1)
            self.assertTrue(mx.allclose(y1, y2))

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_conv_depthwise(self):

        # fmt: off
        shapes = (
            # N,   H,   W,    C   kH,  kW,   O, strides, padding,  groups
            ( 2,  16,  16,   32,   1,   1,  32,  (2, 2),  (1, 1),    32),
            ( 1,  16,  16,   32,   3,   3,  32,  (2, 2),  (1, 1),    32),
            ( 1,  32,  32,   32,   7,   7,  32,  (1, 1),  (3, 3),    32),
            ( 3,  32,  32,   32,   5,   5,  32,  (1, 2),  (0, 0),    32),
            ( 1,  32,  32,   32,   7,   7,  32,  (2, 1),  (1, 3),    32),
        )
        # fmt: on

        dtypes = [np.float32]
        if mx.default_device() == mx.gpu:
            dtypes += [np.float16]

        for N, H, W, C, kH, kW, O, strides, padding, groups in shapes:
            for dtype in dtypes:
                for flip in [False, True]:
                    Cw = C // groups

                    self.__conv_general_test(
                        (N, H, W, C),
                        (O, kH, kW, Cw),
                        strides,
                        padding,
                        kernel_dilation=1,
                        input_dilation=1,
                        groups=groups,
                        flip=flip,
                        np_dtype=dtype,
                        atol=2e-5 if dtype == np.float32 else 5e-4,
                    )

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_asymmetric_padding(self):
        inputs = np.random.normal(size=(2, 8, 8, 8, 3)).astype(np.float32)
        kernel = np.random.normal(size=(2, 3, 3, 3, 3)).astype(np.float32)
        strides = (2, 2, 2)

        pt_out = torch.conv3d(
            torch.permute(torch.tensor(inputs), (0, 4, 1, 2, 3)),
            torch.permute(torch.tensor(kernel), (0, 4, 1, 2, 3)),
            stride=strides,
            padding=2,
        )
        pt_out = torch.permute(pt_out, (0, 2, 3, 4, 1))[:, 1:, 1:, 1:, :].numpy()

        mx_out = mx.conv_general(
            mx.array(inputs),
            mx.array(kernel),
            stride=strides,
            padding=([0, 0, 0], [1, 1, 1]),
        )

        self.assertTrue(mx.allclose(mx_out, mx.array(pt_out), atol=1e-3, rtol=1e-3))

        inputs = np.random.normal(size=(2, 10, 10, 3)).astype(np.float32)
        kernel = np.random.normal(size=(2, 2, 2, 3)).astype(np.float32)

        pt_out = torch.conv2d(
            torch.permute(torch.tensor(inputs), (0, 3, 1, 2)),
            torch.permute(torch.tensor(kernel), (0, 3, 1, 2)),
            stride=1,
            padding=(1, 0),
        )
        pt_out = torch.permute(pt_out, (0, 2, 3, 1))[:, 1:].numpy()

        mx_out = mx.conv_general(
            mx.array(inputs),
            mx.array(kernel),
            stride=1,
            padding=([0, 0], [1, 0]),
        )
        self.assertTrue(mx.allclose(mx_out, mx.array(pt_out), atol=1e-3, rtol=1e-3))

    def test_basic_grad_shapes(self):
        def loss_fn(kernel, inputs, strides, groups):
            return mx.sum(
                mx.conv_general(
                    inputs,
                    kernel,
                    stride=strides,
                    groups=groups,
                )
            )

        for in_shape, k_shape, strides, groups in [
            ((3, 5, 4), (6, 2, 2), (2,), 2),
            ((3, 5, 4), (24, 2, 1), (2,), 4),
            ((3, 5, 5, 4), (6, 2, 2, 2), (2, 1), 2),
            ((3, 5, 5, 4), (24, 2, 2, 1), (2, 2), 4),
        ]:
            grads = mx.grad(loss_fn)(
                mx.zeros(k_shape), mx.zeros(in_shape), strides, groups
            )
            self.assertEqual(grads.shape, k_shape)

    def test_1d_conv_with_2d(self):
        x = mx.random.uniform(shape=(2, 10, 16))
        y = mx.random.normal(shape=(16, 3, 16))

        out = mx.conv1d(x, y, padding=1)
        out_2d = mx.conv2d(
            mx.expand_dims(x, axis=2), mx.expand_dims(y, axis=2), padding=(1, 0)
        )

        self.assertTrue(mx.allclose(out, out_2d.squeeze(2)))

        x = mx.random.uniform(shape=(2, 10, 4))
        y = mx.random.normal(shape=(4, 3, 4))

        out = mx.conv1d(x, y, padding=1)
        out_2d = mx.conv2d(
            mx.expand_dims(x, axis=2), mx.expand_dims(y, axis=2), padding=(1, 0)
        )

        self.assertTrue(mx.allclose(out, out_2d.squeeze(2)))

    def test_conv2d_unaligned_channels(self):
        x = mx.random.uniform(shape=(2, 16, 16, 21))
        w = mx.random.uniform(shape=(32, 3, 3, 21))
        y = mx.conv2d(x, w, stream=mx.cpu)
        y_hat = mx.conv2d(x, w)
        self.assertTrue(mx.allclose(y, y_hat))

        x = mx.random.uniform(shape=(2, 16, 16, 21))
        w = mx.random.uniform(shape=(21, 3, 3, 21))
        y = mx.conv2d(x, w, stream=mx.cpu)
        y_hat = mx.conv2d(x, w)
        self.assertTrue(mx.allclose(y, y_hat))

    def test_conv2d_large_filter_small_channels(self):
        x = mx.random.normal(shape=(1, 181, 181, 1))
        w = mx.random.normal(shape=(1, 182, 182, 1))
        y = mx.conv2d(x, w, (1, 1), (1, 1), stream=mx.cpu)
        y_hat = mx.conv2d(x, w, (1, 1), (1, 1))
        self.assertTrue(mx.allclose(y, y_hat, rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
