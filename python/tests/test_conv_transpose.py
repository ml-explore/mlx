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


class TestConvTranspose(mlx_tests.MLXTestCase):
    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_conv_transpose_1D(self):
        def run_conv_transpose_1D(
            N,
            C,
            O,
            iH,
            kH,
            stride,
            padding,
            output_padding=0,
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
                in_pt = torch.from_numpy(in_np.transpose(0, 2, 1))
                wt_pt = torch.from_numpy(wt_np.transpose(2, 0, 1))

                out_mx = mx.conv_transpose1d(
                    in_mx,
                    wt_mx,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                out_pt = torch.conv_transpose1d(
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
                    run_conv_transpose_1D(N, C, O, iH, kH, stride, padding, dtype=dtype)

        # Groups tests
        N, C, O = (4, 32, 64)
        for iH, kH, stride, padding in (
            (1, 1, 1, 0),
            (3, 3, 1, 0),
            (31, 5, 5, 2),
        ):
            for group in (1,):
                run_conv_transpose_1D(
                    N, C, O, iH, kH, stride, padding, groups=group, dtype=dtype
                )

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
                out_mx = mx.conv_transpose1d(in_mx_t, wt_mx_t)

                in_pt = torch.from_numpy(in_np.transpose(tpose_in).transpose(0, 2, 1))
                wt_pt = torch.from_numpy(wt_np.transpose(tpose_wt).transpose(2, 0, 1))

                out_pt = torch.conv_transpose1d(in_pt, wt_pt)
                out_pt = torch.transpose(out_pt, 2, 1)

                self.assertEqual(out_pt.shape, out_mx.shape)
                self.assertTrue(np.allclose(out_pt.numpy(), out_mx, atol=1e-5))

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_conv_transpose_1D_grad(self):
        def run_conv_transpose1D_grad(
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
                # oH = 1 + ((iH + 2 * padding - dilation * (kH - 1) - 1) // stride)

                in_np = np.random.normal(0, 1.0 / C, (N, iH, C)).astype(np_dtype)
                wt_np = np.random.normal(0, 1.0 / C, (O, kH, C)).astype(np_dtype)

                in_mx, wt_mx = map(mx.array, (in_np, wt_np))
                in_pt = torch.from_numpy(in_np.transpose(0, 2, 1)).requires_grad_(True)
                wt_pt = torch.from_numpy(wt_np.transpose(2, 0, 1)).requires_grad_(True)

                out_pt = F.conv_transpose1d(
                    in_pt, wt_pt, stride=stride, padding=padding, dilation=dilation
                )

                # use torch to compute ct
                out_pt.retain_grad()
                out_pt.sum().backward()

                pt_grad_in = in_pt.grad.permute(0, 2, 1).numpy()
                pt_grad_wt = wt_pt.grad.permute(1, 2, 0).numpy()

                ct_mx = mx.array(out_pt.grad.numpy().transpose(0, 2, 1))

                def f(a, b):
                    return mx.conv_transpose1d(
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
                    run_conv_transpose1D_grad(
                        N, C, O, iH, kH, stride, padding, dtype=dtype
                    )

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_conv_transpose_2D(self):
        def run_conv_transpose2D(
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
                wt_np = np.random.normal(0.0, 1.0, (O, kH, kW, int(C / groups))).astype(
                    np_dtype
                )

                in_mx, wt_mx = map(mx.array, (in_np, wt_np))
                in_pt = torch.from_numpy(in_np.transpose(0, 3, 1, 2)).to("cpu")
                wt_pt = torch.from_numpy(wt_np.transpose(3, 0, 1, 2)).to("cpu")

                out_mx = mx.conv_transpose2d(
                    in_mx,
                    wt_mx,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                out_pt = torch.conv_transpose2d(
                    in_pt,
                    wt_pt,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                out_pt = torch.permute(out_pt, (0, 2, 3, 1)).numpy(force=True)

                self.assertEqual(out_pt.shape, out_mx.shape)
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
                    run_conv_transpose2D(
                        N, C, O, idim, kdim, stride, padding, dtype=dtype
                    )

            # Groups tests
            N, C, O = (4, 32, 64)
            for idim, kdim, stride, padding in (
                ((1, 1), (1, 1), (1, 1), (0, 0)),
                ((3, 3), (3, 1), (1, 1), (0, 0)),
                ((31, 31), (5, 5), (5, 5), (2, 2)),
            ):
                for group in (1,):
                    run_conv_transpose2D(
                        N, C, O, idim, kdim, stride, padding, groups=group, dtype=dtype
                    )

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_conv_transpose_2D_grad(self):
        def run_conv_transpose2D_grad(
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
                scale = 1.0 / math.sqrt(kH * kW * C * O)

                in_np = np.random.normal(0.0, scale, (N, iH, iW, C)).astype(np_dtype)
                wt_np = np.random.normal(0.0, scale, (O, kH, kW, C)).astype(np_dtype)

                in_mx, wt_mx = map(mx.array, (in_np, wt_np))
                in_pt = torch.from_numpy(in_np.transpose(0, 3, 1, 2)).requires_grad_(
                    True
                )
                wt_pt = torch.from_numpy(wt_np.transpose(3, 0, 1, 2)).requires_grad_(
                    True
                )

                out_pt = F.conv_transpose2d(
                    in_pt, wt_pt, stride=stride, padding=padding, dilation=dilation
                )

                # use torch to compute ct
                out_pt.retain_grad()
                out_pt.sum().backward()

                pt_grad_in = in_pt.grad.permute(0, 2, 3, 1).numpy()
                pt_grad_wt = wt_pt.grad.permute(1, 2, 3, 0).numpy()

                ct_mx = mx.array(out_pt.grad.numpy().transpose(0, 2, 3, 1))

                def f(a, b):
                    return mx.conv_transpose2d(
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
                    run_conv_transpose2D_grad(
                        N, C, O, idim, kdim, stride, padding, dilation, dtype=dtype
                    )

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_conv_transpose_3D(self):
        def run_conv_transpose3D(
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
                scale = 1.0 / math.sqrt(kD * kH * kW * C * O)
                in_np = np.random.normal(0.0, scale, (N, iD, iH, iW, C)).astype(
                    np_dtype
                )
                wt_np = np.random.normal(0.0, 1.0, (O, kD, kH, kW, C)).astype(np_dtype)

                in_mx, wt_mx = map(mx.array, (in_np, wt_np))
                in_pt = torch.from_numpy(in_np.transpose(0, 4, 1, 2, 3))
                wt_pt = torch.from_numpy(wt_np.transpose(4, 0, 1, 2, 3))

                out_mx = mx.conv_transpose3d(
                    in_mx,
                    wt_mx,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )
                out_pt = torch.conv_transpose3d(
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
                (2, 8, 16),
            ):
                for idim, kdim, stride, padding in (
                    ((1, 1, 1), (1, 1, 1), (1, 1, 1), (0, 0, 0)),
                    ((3, 3, 3), (3, 1, 1), (1, 1, 1), (0, 0, 0)),
                    ((15, 15, 15), (3, 3, 3), (3, 3, 3), (2, 2, 2)),
                ):
                    run_conv_transpose3D(
                        N, C, O, idim, kdim, stride, padding, dtype=dtype
                    )

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_conv_transpose_3D_grad(self):
        def run_conv_transpose3D_grad(
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
            atol=1e-4,
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
                scale = 1.0 / math.sqrt(kD * kH * kW * C * O)

                in_np = np.random.normal(0.0, scale, (N, iD, iH, iW, C)).astype(
                    np_dtype
                )
                wt_np = np.random.normal(0.0, scale, (O, kD, kH, kW, C)).astype(
                    np_dtype
                )

                in_mx, wt_mx = map(mx.array, (in_np, wt_np))
                in_pt = torch.from_numpy(in_np.transpose(0, 4, 1, 2, 3)).requires_grad_(
                    True
                )
                wt_pt = torch.from_numpy(wt_np.transpose(4, 0, 1, 2, 3)).requires_grad_(
                    True
                )

                out_pt = F.conv_transpose3d(
                    in_pt,
                    wt_pt,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                )

                # use torch to compute ct
                out_pt.retain_grad()
                out_pt.sum().backward()

                pt_grad_in = in_pt.grad.permute(0, 2, 3, 4, 1).numpy()
                pt_grad_wt = wt_pt.grad.permute(1, 2, 3, 4, 0).numpy()

                ct_mx = mx.array(out_pt.grad.numpy().transpose(0, 2, 3, 4, 1))

                def f(a, b):
                    return mx.conv_transpose3d(
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

                mx_grad_in, mx_grad_wt = outs_mx

                self.assertEqual(pt_grad_in.shape, mx_grad_in.shape)
                self.assertEqual(in_mx.shape, mx_grad_in.shape)
                self.assertTrue(np.allclose(pt_grad_in, mx_grad_in, atol=atol))

                self.assertEqual(pt_grad_wt.shape, mx_grad_wt.shape)
                self.assertEqual(wt_mx.shape, mx_grad_wt.shape)
                self.assertTrue(np.allclose(pt_grad_wt, mx_grad_wt, atol=atol))

        for dtype in ("float32",):
            for N, C, O in ((1, 1, 1), (1, 6, 1), (1, 1, 6), (2, 4, 8), (2, 8, 16)):
                for idim, kdim, stride, padding, dilation in (
                    ((1, 1, 1), (1, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)),
                    ((3, 3, 3), (3, 1, 1), (1, 1, 1), (0, 0, 0), (1, 1, 1)),
                    ((15, 15, 15), (5, 5, 5), (5, 5, 5), (2, 2, 2), (1, 1, 1)),
                    ((16, 16, 16), (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1)),
                    ((15, 15, 15), (5, 5, 5), (5, 5, 5), (2, 2, 2), (3, 2, 2)),
                    ((16, 16, 16), (3, 3, 3), (2, 2, 2), (1, 1, 1), (3, 2, 2)),
                ):
                    run_conv_transpose3D_grad(
                        N, C, O, idim, kdim, stride, padding, dilation, dtype=dtype
                    )


if __name__ == "__main__":
    unittest.main()
