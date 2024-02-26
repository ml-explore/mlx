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
                for idim, kdim, stride, padding in (
                    ((1, 1), (1, 1), (1, 1), (0, 0)),
                    ((3, 3), (3, 1), (1, 1), (0, 0)),
                    ((31, 31), (5, 5), (5, 5), (2, 2)),
                ):
                    run_conv2D_grad(N, C, O, idim, kdim, stride, padding, dtype=dtype)

    def __convNd_test(
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

            scale = 1.0 / math.sqrt(np.prod(wt_shape[1:]))
            in_np = np.random.normal(0.0, scale, in_shape).astype(np_dtype)
            wt_np = np.random.normal(0.0, scale, wt_shape).astype(np_dtype)

            in_mx, wt_mx = map(mx.array, (in_np, wt_np))

            in_pt, wt_pt = map(
                lambda x: torch.from_numpy(np.moveaxis(x, -1, 1)).to("cpu"),
                (in_np, wt_np),
            )

            out_mx = mx.convNd(
                in_mx,
                wt_mx,
                stride=stride,
                padding=padding,
                kernel_dilation=kernel_dilation,
                input_dilation=input_dilation,
                groups=groups,
                flip=flip,
            )

            def convNd_pt(
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
                    wt = torch.flip(wt, np.arange(2, wt.ndim()))

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

            out_pt = convNd_pt(
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
    def test_torch_conv_Nd(self):
        in_shape = (2, 32, 32, 16)
        wt_shape = (32, 5, 5, 16)
        stride = (1, 1)
        padding = (2, 2)
        kernel_dilation = (1, 1)
        input_dilation = (1, 1)
        flip = False

        self.__convNd_test(
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
        flip = False

        self.__convNd_test(
            in_shape,
            wt_shape,
            stride,
            padding,
            kernel_dilation,
            input_dilation,
            flip=flip,
        )


if __name__ == "__main__":
    unittest.main()
