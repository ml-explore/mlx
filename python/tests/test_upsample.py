# Copyright © 2023-2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx_tests
import numpy as np

try:
    import torch
    import torch.nn.functional as F

    has_torch = True
except ImportError as e:
    has_torch = False


class TestUpsample(mlx_tests.MLXTestCase):
    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_upsample(self):
        def run_upsample(
            N,
            C,
            idim,
            scale_factor,
            mode,
            align_corner,
            dtype="float32",
            atol=1e-5,
        ):
            with self.subTest(
                N=N,
                C=C,
                idim=idim,
                scale_factor=scale_factor,
                mode=mode,
                align_corner=align_corner,
            ):
                np_dtype = getattr(np, dtype)
                np.random.seed(0)
                iH, iW = idim
                in_np = np.random.normal(-1.0, 1.0, (N, iH, iW, C)).astype(np_dtype)

                in_mx = mx.array(in_np)
                in_pt = torch.from_numpy(in_np.transpose(0, 3, 1, 2)).to("cpu")

                out_mx = nn.Upsample(
                    scale_factor=scale_factor,
                    mode=mode,
                    align_corners=align_corner,
                )(in_mx)
                mode_pt = {
                    "linear": "bilinear",
                    "cubic": "bicubic",
                }[mode]
                out_pt = F.interpolate(
                    in_pt,
                    scale_factor=scale_factor,
                    mode=mode_pt,
                    align_corners=align_corner,
                )
                out_pt = torch.permute(out_pt, (0, 2, 3, 1)).numpy(force=True)
                self.assertEqual(out_pt.shape, out_mx.shape)
                self.assertTrue(np.allclose(out_pt, out_mx, atol=atol))

        for dtype in ("float32",):
            for N, C in ((1, 1), (2, 3)):
                # only test cases in which target sizes are intergers
                # if not, there will be numerical difference between mlx
                # and torch due to different indices selection.
                for idim, scale_factor in (
                    ((2, 2), (1.0, 1.0)),
                    ((2, 2), (1.5, 1.5)),
                    ((2, 2), (2.0, 2.0)),
                    ((4, 4), (0.5, 0.5)),
                    ((7, 7), (2.0, 2.0)),
                    ((10, 10), (0.2, 0.2)),
                    ((11, 21), (3.0, 3.0)),
                    ((11, 21), (3.0, 2.0)),
                ):
                    # only test linear and cubic interpolation
                    # there will be numerical difference in nearest
                    # due to different indices selection.
                    for mode in ("cubic", "linear"):
                        for align_corner in (False, True):
                            run_upsample(
                                N,
                                C,
                                idim,
                                scale_factor,
                                mode,
                                align_corner,
                                dtype=dtype,
                            )


if __name__ == "__main__":
    unittest.main()
