# Copyright © 2023 Apple Inc.

import itertools
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestFFT(mlx_tests.MLXTestCase):
    def check_mx_np(self, op_mx, op_np, a_np, atol=1e-5, rtol=1e-6, **kwargs):
        out_np = op_np(a_np, **kwargs)
        a_mx = mx.array(a_np)
        out_mx = op_mx(a_mx, **kwargs)
        np.testing.assert_allclose(out_np, out_mx, atol=atol, rtol=rtol)

    def test_fft(self):
        with mx.stream(mx.cpu):
            r = np.random.rand(100).astype(np.float32)
            i = np.random.rand(100).astype(np.float32)
            a_np = r + 1j * i
            self.check_mx_np(mx.fft.fft, np.fft.fft, a_np)

            # Check with slicing and padding
            r = np.random.rand(100).astype(np.float32)
            i = np.random.rand(100).astype(np.float32)
            a_np = r + 1j * i
            self.check_mx_np(mx.fft.fft, np.fft.fft, a_np, n=80)
            self.check_mx_np(mx.fft.fft, np.fft.fft, a_np, n=120)

            # Check different axes
            r = np.random.rand(100, 100).astype(np.float32)
            i = np.random.rand(100, 100).astype(np.float32)
            a_np = r + 1j * i
            self.check_mx_np(mx.fft.fft, np.fft.fft, a_np, axis=0)
            self.check_mx_np(mx.fft.fft, np.fft.fft, a_np, axis=1)

            # Check real fft
            a_np = np.random.rand(100).astype(np.float32)
            self.check_mx_np(mx.fft.rfft, np.fft.rfft, a_np)
            self.check_mx_np(mx.fft.rfft, np.fft.rfft, a_np, n=80)
            self.check_mx_np(mx.fft.rfft, np.fft.rfft, a_np, n=120)

            # Check real inverse
            r = np.random.rand(100, 100).astype(np.float32)
            i = np.random.rand(100, 100).astype(np.float32)
            a_np = r + 1j * i
            self.check_mx_np(mx.fft.ifft, np.fft.ifft, a_np)
            self.check_mx_np(mx.fft.ifft, np.fft.ifft, a_np, n=80)
            self.check_mx_np(mx.fft.ifft, np.fft.ifft, a_np, n=120)
            self.check_mx_np(mx.fft.irfft, np.fft.irfft, a_np)
            self.check_mx_np(mx.fft.irfft, np.fft.irfft, a_np, n=80)
            self.check_mx_np(mx.fft.irfft, np.fft.irfft, a_np, n=120)

    def test_fftn(self):
        with mx.stream(mx.cpu):
            r = np.random.randn(8, 8, 8).astype(np.float32)
            i = np.random.randn(8, 8, 8).astype(np.float32)
            a = r + 1j * i

            axes = [None, (1, 2), (2, 1), (0, 2)]
            shapes = [None, (10, 5), (5, 10)]
            ops = [
                "fft2",
                "ifft2",
                "rfft2",
                "irfft2",
                "fftn",
                "ifftn",
                "rfftn",
                "irfftn",
            ]

            for op, ax, s in itertools.product(ops, axes, shapes):
                x = a
                if op in ["rfft2", "rfftn"]:
                    x = r
                mx_op = getattr(mx.fft, op)
                np_op = getattr(np.fft, op)
                self.check_mx_np(mx_op, np_op, x, axes=ax, s=s)

    def test_fft_powers_of_two(self):
        shape = (16, 4, 8)
        # np.fft.fft always uses double precision complex128
        # mx.fft.fft only supports single precision complex64
        # hence the fairly tolerant equality checks.
        atol = 1e-4
        rtol = 1e-4
        np.random.seed(7)
        for k in range(4, 12):
            r = np.random.rand(*shape, 2**k).astype(np.float32)
            i = np.random.rand(*shape, 2**k).astype(np.float32)
            a_np = r + 1j * i
            self.check_mx_np(mx.fft.fft, np.fft.fft, a_np, atol=atol, rtol=rtol)

        r = np.random.rand(*shape, 32).astype(np.float32)
        i = np.random.rand(*shape, 32).astype(np.float32)
        a_np = r + 1j * i
        for axis in range(4):
            self.check_mx_np(
                mx.fft.fft, np.fft.fft, a_np, atol=atol, rtol=rtol, axis=axis
            )

        r = np.random.rand(4, 8).astype(np.float32)
        i = np.random.rand(4, 8).astype(np.float32)
        a_np = r + 1j * i
        a_mx = mx.array(a_np)

        # non-contiguous in the FFT dim
        out_mx = mx.fft.fft(a_mx[:, ::2])
        out_np = np.fft.fft(a_np[:, ::2])
        np.testing.assert_allclose(out_np, out_mx, atol=1e-5, rtol=1e-5)

        # non-contiguous not in the FFT dim
        out_mx = mx.fft.fft(a_mx[::2])
        out_np = np.fft.fft(a_np[::2])
        np.testing.assert_allclose(out_np, out_mx, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
