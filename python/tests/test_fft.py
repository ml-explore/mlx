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

        x = np.fft.rfft(a_np)
        self.check_mx_np(mx.fft.irfft, np.fft.irfft, x)

    def test_fftn(self):
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
            elif op == "irfft2":
                x = np.ascontiguousarray(np.fft.rfft2(x, axes=ax, s=s))
            elif op == "irfftn":
                x = np.ascontiguousarray(np.fft.rfftn(x, axes=ax, s=s))
            mx_op = getattr(mx.fft, op)
            np_op = getattr(np.fft, op)
            self.check_mx_np(mx_op, np_op, x, axes=ax, s=s)

    def _run_ffts(self, shape, atol=1e-4, rtol=1e-4):
        np.random.seed(9)

        r = np.random.rand(*shape).astype(np.float32)
        i = np.random.rand(*shape).astype(np.float32)
        a_np = r + 1j * i
        self.check_mx_np(mx.fft.fft, np.fft.fft, a_np, atol=atol, rtol=rtol)
        self.check_mx_np(mx.fft.ifft, np.fft.ifft, a_np, atol=atol, rtol=rtol)

        self.check_mx_np(mx.fft.rfft, np.fft.rfft, r, atol=atol, rtol=rtol)

        ia_np = np.fft.rfft(a_np)
        self.check_mx_np(
            mx.fft.irfft, np.fft.irfft, ia_np, atol=atol, rtol=rtol, n=shape[-1]
        )
        self.check_mx_np(mx.fft.irfft, np.fft.irfft, ia_np, atol=atol, rtol=rtol)

    def test_fft_shared_mem(self):
        nums = np.concatenate(
            [
                # small radix
                np.arange(2, 14),
                # powers of 2
                [2**k for k in range(4, 13)],
                # stockham
                [3 * 3 * 3, 3 * 11, 11 * 13 * 2, 7 * 4 * 13 * 11, 13 * 13 * 11],
                # rader
                [17, 23, 29, 17 * 8 * 3, 23 * 2, 1153, 1982],
                # bluestein
                [47, 83, 17 * 17],
                # large stockham
                [3159, 3645, 3969, 4004],
            ]
        )
        for batch_size in (1, 3, 32):
            for num in nums:
                atol = 1e-4 if num < 1025 else 1e-3
                self._run_ffts((batch_size, num), atol=atol)

    @unittest.skip("Too slow for CI but useful for local testing.")
    def test_fft_exhaustive(self):
        nums = range(2, 4097)
        for batch_size in (1, 3, 32):
            for num in nums:
                print(num)
                atol = 1e-4 if num < 1025 else 1e-3
                self._run_ffts((batch_size, num), atol=atol)

    def test_fft_big_powers_of_two(self):
        # TODO: improve precision on big powers of two on GPU
        for k in range(12, 17):
            self._run_ffts((3, 2**k), atol=1e-3)

        for k in range(17, 20):
            self._run_ffts((3, 2**k), atol=1e-2)

    def test_fft_large_numbers(self):
        numbers = [
            1037,  # prime > 2048
            18247,  # medium size prime factors
            1259 * 11,  # large prime factors
            7883,  # large prime
            3**8,  # large stockham decomposable
            3109,  # bluestein
            4006,  # large rader
        ]
        for large_num in numbers:
            self._run_ffts((1, large_num), atol=1e-4)

    def test_fft_contiguity(self):
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

        out_mx = mx.broadcast_to(mx.reshape(mx.transpose(a_mx), (4, 8, 1)), (4, 8, 16))
        out_np = np.broadcast_to(np.reshape(np.transpose(a_np), (4, 8, 1)), (4, 8, 16))
        np.testing.assert_allclose(out_np, out_mx, atol=1e-5, rtol=1e-5)

        out2_mx = mx.fft.fft(mx.abs(out_mx) + 4)
        out2_np = np.fft.fft(np.abs(out_np) + 4)
        np.testing.assert_allclose(out2_mx, out2_np, atol=1e-5, rtol=1e-5)

        b_np = np.array([[0, 1, 2, 3]])
        out_mx = mx.abs(mx.fft.fft(mx.tile(mx.reshape(mx.array(b_np), (1, 4)), (4, 1))))
        out_np = np.abs(np.fft.fft(np.tile(np.reshape(np.array(b_np), (1, 4)), (4, 1))))
        np.testing.assert_allclose(out_mx, out_np, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
