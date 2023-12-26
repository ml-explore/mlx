# Copyright © 2023 Apple Inc.

import itertools
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestFFT(mlx_tests.MLXTestCase):
    def check_mx_np(self, op, a_np, axes, s):
        with self.subTest(op=op, axes=axes, s=s):
            op_np = getattr(np.fft, op)
            op_mx = getattr(mx.fft, op)
            out_np = op_np(a_np, s=s, axes=axes)
            a_mx = mx.array(a_np)
            out_mx = op_mx(a_mx, s=s, axes=axes)
            self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6))

    def test_fft(self):
        default = mx.default_device()
        mx.set_default_device(mx.cpu)

        def check_mx_np(op_mx, op_np, a_np, **kwargs):
            out_np = op_np(a_np, **kwargs)
            a_mx = mx.array(a_np)
            out_mx = op_mx(a_mx, **kwargs)
            self.assertTrue(np.allclose(out_np, out_mx, atol=1e-5, rtol=1e-6))

        r = np.random.rand(100).astype(np.float32)
        i = np.random.rand(100).astype(np.float32)
        a_np = r + 1j * i
        check_mx_np(mx.fft.fft, np.fft.fft, a_np)

        # Check with slicing and padding
        r = np.random.rand(100).astype(np.float32)
        i = np.random.rand(100).astype(np.float32)
        a_np = r + 1j * i
        check_mx_np(mx.fft.fft, np.fft.fft, a_np, n=80)
        check_mx_np(mx.fft.fft, np.fft.fft, a_np, n=120)

        # Check different axes
        r = np.random.rand(100, 100).astype(np.float32)
        i = np.random.rand(100, 100).astype(np.float32)
        a_np = r + 1j * i
        check_mx_np(mx.fft.fft, np.fft.fft, a_np, axis=0)
        check_mx_np(mx.fft.fft, np.fft.fft, a_np, axis=1)

        # Check real fft
        a_np = np.random.rand(100).astype(np.float32)
        check_mx_np(mx.fft.rfft, np.fft.rfft, a_np)
        check_mx_np(mx.fft.rfft, np.fft.rfft, a_np, n=80)
        check_mx_np(mx.fft.rfft, np.fft.rfft, a_np, n=120)

        # Check real inverse
        r = np.random.rand(100, 100).astype(np.float32)
        i = np.random.rand(100, 100).astype(np.float32)
        a_np = r + 1j * i
        check_mx_np(mx.fft.ifft, np.fft.ifft, a_np)
        check_mx_np(mx.fft.ifft, np.fft.ifft, a_np, n=80)
        check_mx_np(mx.fft.ifft, np.fft.ifft, a_np, n=120)
        check_mx_np(mx.fft.irfft, np.fft.irfft, a_np)
        check_mx_np(mx.fft.irfft, np.fft.irfft, a_np, n=80)
        check_mx_np(mx.fft.irfft, np.fft.irfft, a_np, n=120)

        mx.set_default_device(default)

    def test_fftn(self):
        default = mx.default_device()
        mx.set_default_device(mx.cpu)

        r = np.random.randn(8, 8, 8).astype(np.float32)
        i = np.random.randn(8, 8, 8).astype(np.float32)
        a = r + 1j * i

        axes = [None, (1, 2), (2, 1), (0, 2)]
        shapes = [None, (10, 5), (5, 10)]
        ops = ["fft2", "ifft2", "rfft2", "irfft2", "fftn", "ifftn", "rfftn", "irfftn"]

        for op, ax, s in itertools.product(ops, axes, shapes):
            x = a
            if op in ["rfft2", "rfftn"]:
                x = r
            self.check_mx_np(op, x, axes=ax, s=s)

        mx.set_default_device(default)


if __name__ == "__main__":
    unittest.main()
