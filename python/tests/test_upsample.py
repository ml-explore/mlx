# Copyright © 2023-2024 Apple Inc.

import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import mlx_tests
import numpy as np
from mlx.nn.layers.upsample import upsample_cubic, upsample_linear

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
                    "nearest": "nearest",
                    "linear": "bilinear",
                    "cubic": "bicubic",
                }[mode]
                out_pt = F.interpolate(
                    in_pt,
                    scale_factor=scale_factor,
                    mode=mode_pt,
                    align_corners=align_corner if mode != "nearest" else None,
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
                    ((10, 10), (0.3, 0.3)),
                    ((11, 21), (3.0, 3.0)),
                    ((11, 21), (3.0, 2.0)),
                ):
                    for mode in ("cubic", "linear", "nearest"):
                        for align_corner in (False, True):
                            if mode == "nearest" and align_corner:
                                continue
                            run_upsample(
                                N,
                                C,
                                idim,
                                scale_factor,
                                mode,
                                align_corner,
                                dtype=dtype,
                            )

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_torch_upsample_antialias(self):
        """Test antialiased downsampling matches PyTorch F.interpolate(antialias=True)."""

        def run_antialias(
            N,
            C,
            idim,
            scale_factor,
            mode,
            align_corners=False,
            dtype="float32",
            atol=1e-5,
        ):
            with self.subTest(
                N=N,
                C=C,
                idim=idim,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners,
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
                    align_corners=align_corners,
                    antialias=True,
                )(in_mx)
                mode_pt = {
                    "linear": "bilinear",
                    "cubic": "bicubic",
                }[mode]
                out_pt = F.interpolate(
                    in_pt,
                    scale_factor=scale_factor,
                    mode=mode_pt,
                    align_corners=align_corners,
                    antialias=True,
                )
                out_pt = torch.permute(out_pt, (0, 2, 3, 1)).numpy(force=True)
                self.assertEqual(out_pt.shape, out_mx.shape)
                self.assertTrue(
                    np.allclose(out_pt, out_mx, atol=atol),
                    f"antialias {mode} ac={align_corners} scale={scale_factor} max_diff="
                    f"{np.abs(out_pt - np.array(out_mx)).max():.2e}",
                )

        for dtype in ("float32",):
            for N, C in ((1, 1), (2, 3)):
                # Test downscale with antialias — use integer-ratio scales
                # to avoid the pre-existing _scaled_indices step divergence
                # for non-integer ratios (see issue #2186).
                for idim, scale_factor in (
                    ((4, 4), (0.5, 0.5)),
                    ((8, 8), (0.5, 0.5)),
                    ((8, 8), (0.25, 0.25)),
                    ((16, 16), (0.5, 0.5)),
                    ((16, 16), (0.25, 0.25)),
                    ((32, 32), (0.5, 0.5)),
                    ((32, 32), (0.25, 0.25)),
                    ((64, 64), (0.5, 0.5)),
                    ((10, 10), (0.5, 0.5)),
                    ((12, 12), (0.25, 0.25)),
                    ((8, 16), (0.5, 0.5)),  # non-square
                    ((16, 8), (0.5, 0.25)),  # different scales per dim
                ):
                    for mode in ("linear", "cubic"):
                        # align_corners=True + antialias has a known
                        # interaction with _scaled_indices that requires
                        # further work. Test align_corners=False for now.
                        run_antialias(
                            N,
                            C,
                            idim,
                            scale_factor,
                            mode,
                            align_corners=False,
                            dtype=dtype,
                        )

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_antialias_upscale_linear_is_noop(self):
        """For linear mode, antialias has no effect on upscaling."""
        np.random.seed(0)
        in_np = np.random.normal(-1.0, 1.0, (1, 4, 4, 3)).astype(np.float32)
        in_mx = mx.array(in_np)

        for scale in (2.0, 3.0):
            with self.subTest(scale=scale):
                out_aa = nn.Upsample(
                    scale_factor=scale,
                    mode="linear",
                    align_corners=False,
                    antialias=True,
                )(in_mx)
                out_no = nn.Upsample(
                    scale_factor=scale,
                    mode="linear",
                    align_corners=False,
                    antialias=False,
                )(in_mx)
                self.assertTrue(
                    np.allclose(np.array(out_aa), np.array(out_no), atol=1e-7),
                    "linear antialias should be no-op for upscaling",
                )

    @unittest.skipIf(not has_torch, "requires Torch")
    def test_antialias_upscale_cubic_matches_pytorch(self):
        """For cubic mode, antialias changes a from -0.75 to -0.5 even on upscale."""
        np.random.seed(0)
        in_np = np.random.normal(-1.0, 1.0, (1, 8, 8, 3)).astype(np.float32)
        in_mx = mx.array(in_np)
        in_pt = torch.from_numpy(in_np.transpose(0, 3, 1, 2))

        for scale in (2.0, 3.0):
            with self.subTest(scale=scale):
                out_mx = nn.Upsample(
                    scale_factor=scale,
                    mode="cubic",
                    align_corners=False,
                    antialias=True,
                )(in_mx)
                out_pt = F.interpolate(
                    in_pt,
                    scale_factor=scale,
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
                out_pt_np = out_pt.permute(0, 2, 3, 1).numpy(force=True)
                self.assertTrue(
                    np.allclose(np.array(out_mx), out_pt_np, atol=1e-5),
                    f"cubic antialias upscale {scale}x max_diff="
                    f"{np.abs(np.array(out_mx) - out_pt_np).max():.2e}",
                )

    def test_antialias_non_integer_scale_smoke(self):
        """Smoke test for non-integer scale factors (no PyTorch comparison)."""
        np.random.seed(42)
        in_np = np.random.normal(0, 1, (1, 32, 32, 3)).astype(np.float32)
        in_mx = mx.array(in_np)

        for scale in (0.3, 0.7, 0.6):
            for mode in ("linear", "cubic"):
                with self.subTest(scale=scale, mode=mode):
                    out = nn.Upsample(
                        scale_factor=scale,
                        mode=mode,
                        align_corners=False,
                        antialias=True,
                    )(in_mx)
                    mx.eval(out)
                    out_np = np.array(out)

                    # Correct shape
                    expected = int(32 * scale)
                    self.assertEqual(out_np.shape, (1, expected, expected, 3))

                    self.assertTrue(np.all(np.isfinite(out_np)))
                    if mode == "linear":
                        # Linear AA uses non-negative triangle weights, so it is
                        # bounded by the input range. Cubic interpolation can
                        # overshoot because the Keys kernel has negative lobes.
                        self.assertLessEqual(
                            out_np.max(),
                            in_np.max() + 0.01,
                            "linear AA output should not exceed input range",
                        )
                        self.assertGreaterEqual(
                            out_np.min(),
                            in_np.min() - 0.01,
                            "linear AA output should not go below input range",
                        )

    def test_antialias_1d_smoke(self):
        """Test that antialias works on 1D spatial input (3D tensor).

        PyTorch does not support antialias on 1D tensors, so this is a
        smoke test only (correct shape, non-trivial, smoother than non-AA).
        """
        np.random.seed(0)
        for length, scale in ((16, 0.5), (32, 0.25)):
            for mode in ("linear", "cubic"):
                with self.subTest(length=length, scale=scale, mode=mode):
                    in_np = np.random.normal(0, 1, (1, length, 3)).astype(np.float32)
                    in_mx = mx.array(in_np)

                    out_aa = nn.Upsample(
                        scale_factor=scale,
                        mode=mode,
                        align_corners=False,
                        antialias=True,
                    )(in_mx)
                    out_no = nn.Upsample(
                        scale_factor=scale,
                        mode=mode,
                        align_corners=False,
                        antialias=False,
                    )(in_mx)
                    mx.eval(out_aa, out_no)

                    expected_len = int(length * scale)
                    self.assertEqual(out_aa.shape, (1, expected_len, 3))
                    # AA should differ from non-AA
                    self.assertGreater(
                        float(mx.abs(out_aa - out_no).max()),
                        1e-6,
                    )

    def test_antialias_uses_separable_path(self):
        """AA interpolation should avoid cartesian product gather expansion."""
        in_mx = mx.zeros((1, 16, 16, 1))
        for mode in ("linear", "cubic"):
            with self.subTest(mode=mode):
                with patch(
                    "mlx.nn.layers.upsample.product",
                    side_effect=AssertionError("cartesian interpolation path used"),
                ):
                    out = nn.Upsample(
                        scale_factor=0.25,
                        mode=mode,
                        align_corners=False,
                        antialias=True,
                    )(in_mx)
                    mx.eval(out)
                    self.assertEqual(out.shape, (1, 4, 4, 1))

    def test_antialias_nearest_raises(self):
        """Antialias + nearest should raise ValueError."""
        with self.assertRaises(ValueError):
            nn.Upsample(scale_factor=0.5, mode="nearest", antialias=True)

    def test_antialias_align_corners_raises(self):
        """Antialias + align_corners is unsupported and should raise."""
        for mode in ("linear", "cubic"):
            for scale in (0.5, 2.0):
                with self.subTest(mode=mode, scale=scale):
                    with self.assertRaises(ValueError):
                        nn.Upsample(
                            scale_factor=scale,
                            mode=mode,
                            align_corners=True,
                            antialias=True,
                        )

    def test_antialias_align_corners_direct_functions_raise(self):
        """Direct interpolation helpers should enforce the same AA contract."""
        x = mx.zeros((1, 4, 4, 1))
        for fn in (upsample_linear, upsample_cubic):
            for scale in ((0.5, 0.5), (2.0, 2.0)):
                with self.subTest(fn=fn.__name__, scale=scale):
                    with self.assertRaises(ValueError):
                        fn(x, scale, align_corners=True, antialias=True)

    def test_antialias_differs_from_non_antialias(self):
        """Antialiased downscale should produce different output than non-AA."""
        np.random.seed(42)
        in_np = np.random.normal(0, 1, (1, 32, 32, 3)).astype(np.float32)
        in_mx = mx.array(in_np)

        out_no = nn.Upsample(
            scale_factor=0.5, mode="linear", align_corners=False, antialias=False
        )(in_mx)
        out_aa = nn.Upsample(
            scale_factor=0.5, mode="linear", align_corners=False, antialias=True
        )(in_mx)

        # Outputs should differ (AA applies a wider filter)
        diff = float(mx.abs(out_aa - out_no).max())
        self.assertGreater(
            diff,
            1e-6,
            "AA and non-AA downscale should produce different results",
        )
        # AA output should have lower variance (wider filter = more averaging)
        std_no = float(mx.std(out_no))
        std_aa = float(mx.std(out_aa))
        self.assertLess(
            std_aa,
            std_no,
            f"AA output should have lower variance: std_aa={std_aa:.4f} >= std_no={std_no:.4f}",
        )


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
