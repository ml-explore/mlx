# Copyright © 2026 Apple Inc.

"""Targeted tests for the multithreaded / vectorized CPU kernels.

The sizes are chosen to cross the parallelization thresholds (so that the
work is split into several chunks with ragged boundaries) and the results are
compared against NumPy.
"""

import os
import subprocess
import sys
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np

# Sizes around and above the elementwise grain (32768 float32 elements per
# chunk), including odd sizes so chunk boundaries are ragged.
SIZES = [1, 7, 4097, 65537, 262147, 1048583]


class TestCPUParallel(mlx_tests.MLXTestCase):
    def setUp(self):
        super().setUp()
        self.rng = np.random.default_rng(0)

    def rand(self, *shape, dtype=np.float32):
        return self.rng.standard_normal(shape).astype(dtype)

    @unittest.skipIf(not mx.is_available(mx.cpu), "CPU only")
    def test_unary_contiguous_and_strided(self):
        with mx.stream(mx.cpu):
            for n in SIZES:
                x = self.rand(n)
                for mf, nf in [(mx.exp, np.exp), (mx.abs, np.abs), (mx.sin, np.sin)]:
                    out = np.array(mf(mx.array(x)))
                    np.testing.assert_allclose(out, nf(x), rtol=2e-6, atol=1e-6)
            # Non-contiguous input: a transposed view and a strided slice
            x = self.rand(1031, 517)
            xt = mx.array(x).T
            np.testing.assert_allclose(
                np.array(mx.exp(xt)), np.exp(x.T), rtol=2e-6, atol=1e-6
            )
            xs = mx.array(x)[::3, 1::2]
            np.testing.assert_allclose(
                np.array(mx.negative(xs)), -x[::3, 1::2], rtol=0, atol=0
            )

    def test_exp_special_values(self):
        with mx.stream(mx.cpu):
            v = np.array(
                [
                    0.0,
                    -0.0,
                    1.0,
                    -1.0,
                    88.0,
                    88.5,
                    -88.0,
                    -88.5,
                    100.0,
                    -100.0,
                    np.inf,
                    -np.inf,
                    np.nan,
                    0.5,
                    -0.5,
                    1e-30,
                ],
                dtype=np.float32,
            )
            out = np.array(mx.exp(mx.array(v)))
            self.assertTrue(np.isnan(out[12]))
            self.assertEqual(out[10], np.inf)
            self.assertEqual(out[11], 0.0)
            self.assertEqual(out[8], np.inf)
            self.assertEqual(out[9], 0.0)
            finite = np.isfinite(v) & (np.abs(v) < 88)
            np.testing.assert_allclose(out[finite], np.exp(v[finite]), rtol=2e-6)
            # Dense sweep over the useful range
            x = np.linspace(-87, 87, 1000003, dtype=np.float32)
            out = np.array(mx.exp(mx.array(x)))
            # 4e-6 is the accuracy of the (unchanged) polynomial approximation.
            np.testing.assert_allclose(out, np.exp(x.astype(np.float64)), rtol=5e-6)

    def test_floor_exact(self):
        with mx.stream(mx.cpu):
            v = np.array(
                [
                    -0.0,
                    0.0,
                    -0.5,
                    0.5,
                    -1.0,
                    1.0,
                    -1.5,
                    2.5,
                    0.99999994,
                    -0.99999994,
                    1e7 + 0.5,
                    -1e7 - 0.5,
                    8388607.5,
                    -8388607.5,
                    8388608.0,
                    -8388609.0,
                    3e9,
                    -3e9,
                    3.4e38,
                    -3.4e38,
                    np.inf,
                    -np.inf,
                    np.nan,
                    1e-40,
                    -1e-40,
                ],
                dtype=np.float32,
            )
            v = np.tile(v, 5000)  # large enough to be split across threads
            out = np.array(mx.floor(mx.array(v)))
            ref = np.floor(v)
            np.testing.assert_array_equal(out, ref)
            np.testing.assert_array_equal(np.signbit(out), np.signbit(ref))

    def test_binary(self):
        with mx.stream(mx.cpu):
            for n in SIZES:
                a, b = self.rand(n), self.rand(n)
                np.testing.assert_array_equal(
                    np.array(mx.array(a) + mx.array(b)), a + b
                )
                np.testing.assert_array_equal(np.array(mx.array(a) * 3.0), a * 3.0)
                np.testing.assert_array_equal(np.array(2.0 - mx.array(b)), 2.0 - b)
                np.testing.assert_array_equal(
                    np.array(mx.array(a) < mx.array(b)), a < b
                )
            # Broadcasting (general path, split over the outer axis)
            for shape_a, shape_b in [
                ((1027, 513), (513,)),
                ((1027, 513), (1027, 1)),
                ((9, 131, 257), (1, 131, 1)),
                ((4, 7, 33, 129), (7, 1, 129)),
                ((3, 5, 7, 11, 13), (5, 1, 11, 1)),
            ]:
                a, b = self.rand(*shape_a), self.rand(*shape_b)
                np.testing.assert_array_equal(
                    np.array(mx.array(a) - mx.array(b)), a - b
                )
            # Non-contiguous operands
            a = self.rand(777, 1201)
            b = self.rand(1201, 777)
            out = mx.array(a) + mx.array(b).T
            np.testing.assert_array_equal(np.array(out), a + b.T)
            # Mixed integer types
            ai = self.rng.integers(-1000, 1000, (300001,), dtype=np.int32)
            bi = self.rng.integers(1, 1000, (300001,), dtype=np.int32)
            np.testing.assert_array_equal(
                np.array(mx.array(ai) * mx.array(bi)), ai * bi
            )

    def test_where(self):
        with mx.stream(mx.cpu):
            for shape in [(1048583,), (1031, 1029)]:
                c = self.rng.random(shape) > 0.5
                a, b = self.rand(*shape), self.rand(*shape)
                out = mx.where(mx.array(c), mx.array(a), mx.array(b))
                np.testing.assert_array_equal(np.array(out), np.where(c, a, b))
            c = self.rng.random((1031, 1)) > 0.5
            a, b = self.rand(1031, 1029), self.rand(1029)
            out = mx.where(mx.array(c), mx.array(a), mx.array(b))
            np.testing.assert_array_equal(np.array(out), np.where(c, a, b))

    def test_where_dtypes_bit_exact(self):
        # Contiguous where() is a bitwise blend: check every element size,
        # and that NaN payloads, -0.0 and inf are copied exactly.
        with mx.stream(mx.cpu):
            for n in [7, 65537, 262147]:
                c = self.rng.random(n) > 0.5
                for dt in [
                    np.bool_,
                    np.uint8,
                    np.int16,
                    np.int32,
                    np.uint64,
                    np.float16,
                    np.float32,
                ]:
                    a = (self.rng.standard_normal(n) * 100).astype(dt)
                    b = (self.rng.standard_normal(n) * 100).astype(dt)
                    out = np.array(mx.where(mx.array(c), mx.array(a), mx.array(b)))
                    np.testing.assert_array_equal(out, np.where(c, a, b))
                a = self.rand(n)
                a[::3] = -0.0
                a[1::5] = np.inf
                a.view(np.uint32)[2::7] = 0x7FC00123  # NaN with a payload
                b = self.rand(n)
                out = np.array(mx.where(mx.array(c), mx.array(a), mx.array(b)))
                expected = np.where(c, a, b)
                np.testing.assert_array_equal(
                    out.view(np.uint32), expected.view(np.uint32)
                )
                x = mx.array(self.rand(n)).astype(mx.bfloat16)
                y = mx.array(self.rand(n)).astype(mx.bfloat16)
                out = mx.where(mx.array(c), x, y)
                ref = np.where(
                    c, np.array(x.astype(mx.float32)), np.array(y.astype(mx.float32))
                )
                np.testing.assert_array_equal(np.array(out.astype(mx.float32)), ref)
                xc = mx.array(self.rand(n) + 1j * self.rand(n))
                yc = mx.array(self.rand(n) + 1j * self.rand(n))
                out = np.array(mx.where(mx.array(c), xc, yc))
                np.testing.assert_array_equal(
                    out, np.where(c, np.array(xc), np.array(yc))
                )

    def test_copies_and_transposes(self):
        with mx.stream(mx.cpu):
            for shape in [(1, 1), (3, 70001), (1023, 1025), (4096, 33), (33, 4096)]:
                x = self.rand(*shape)
                np.testing.assert_array_equal(
                    np.array(mx.contiguous(mx.array(x).T)), x.T
                )
            x = self.rand(5, 37, 129, 65)
            for perm in [(0, 1, 3, 2), (3, 2, 1, 0), (0, 2, 1, 3), (2, 0, 3, 1)]:
                np.testing.assert_array_equal(
                    np.array(mx.contiguous(mx.transpose(mx.array(x), perm))),
                    np.transpose(x, perm),
                )
            # Strided slices and dtype conversion in the copy
            x = self.rand(2049, 1025)
            np.testing.assert_array_equal(
                np.array(mx.contiguous(mx.array(x)[::2, ::3])), x[::2, ::3]
            )
            y = mx.array(x).T.astype(mx.float16)
            np.testing.assert_array_equal(np.array(y), x.T.astype(np.float16))
            # Broadcast fill
            out = mx.broadcast_to(mx.array(3.5), (1031, 1029)) + 0
            np.testing.assert_array_equal(np.array(out), np.full((1031, 1029), 3.5))
            # Slice update (copy into a strided destination)
            a = mx.zeros((1024, 1024))
            b = mx.array(self.rand(512, 1024)).T
            a[:, 256:768] = b
            ref = np.zeros((1024, 1024), np.float32)
            ref[:, 256:768] = np.array(b)
            np.testing.assert_array_equal(np.array(a), ref)

    def test_reductions(self):
        with mx.stream(mx.cpu):
            for n in SIZES:
                x = self.rand(n)
                np.testing.assert_allclose(
                    mx.sum(mx.array(x)).item(),
                    x.astype(np.float64).sum(),
                    rtol=1e-5,
                    atol=1e-3,
                )
                self.assertEqual(mx.max(mx.array(x)).item(), x.max())
                self.assertEqual(mx.min(mx.array(x)).item(), x.min())
            x = self.rand(1031, 2053)
            for axis in [0, 1, None, (0, 1)]:
                np.testing.assert_allclose(
                    np.array(mx.sum(mx.array(x), axis=axis)),
                    x.astype(np.float64).sum(axis=axis),
                    rtol=1e-4,
                    atol=1e-3,
                )
                np.testing.assert_array_equal(
                    np.array(mx.max(mx.array(x), axis=axis)), x.max(axis=axis)
                )
                np.testing.assert_array_equal(
                    np.array(mx.min(mx.array(x), axis=axis)), x.min(axis=axis)
                )
            # NaN propagation in max/min over large arrays
            x = self.rand(300007)
            x[123457] = np.nan
            self.assertTrue(np.isnan(mx.max(mx.array(x)).item()))
            self.assertTrue(np.isnan(mx.min(mx.array(x)).item()))
            # Integer and boolean reductions are exact
            xi = self.rng.integers(-50, 50, (1000003,), dtype=np.int32)
            self.assertEqual(mx.sum(mx.array(xi)).item(), int(xi.sum()))
            xb = self.rng.random(1000003) > 0.0001
            self.assertEqual(mx.all(mx.array(xb)).item(), bool(xb.all()))
            self.assertEqual(mx.any(mx.array(~xb)).item(), bool((~xb).any()))
            self.assertTrue(mx.all(mx.ones((1000003,), dtype=mx.bool_)).item())
            self.assertFalse(mx.any(mx.zeros((1000003,), dtype=mx.bool_)).item())
            xp = self.rng.uniform(0.5, 2.0, (100003,)).astype(np.float32)
            xp = xp / np.exp(np.log(xp.astype(np.float64)).mean()).astype(np.float32)
            np.testing.assert_allclose(
                mx.prod(mx.array(xp[:1003])).item(),
                np.prod(xp[:1003].astype(np.float64)),
                rtol=1e-4,
            )
            # Strided reduction with several outer blocks
            x = self.rand(7, 513, 1029)
            np.testing.assert_allclose(
                np.array(mx.sum(mx.array(x), axis=1)),
                x.astype(np.float64).sum(axis=1),
                rtol=1e-4,
                atol=1e-3,
            )

    def test_softmax(self):
        with mx.stream(mx.cpu):
            for shape in [(1, 7), (3, 4097), (1031, 1029), (4, 8, 129, 257)]:
                x = self.rand(*shape) * 4
                ref = np.exp(x - x.max(-1, keepdims=True))
                ref /= ref.sum(-1, keepdims=True)
                out = np.array(mx.softmax(mx.array(x), axis=-1))
                np.testing.assert_allclose(out, ref, rtol=2e-5, atol=1e-7)
            x = self.rand(1031, 1029)
            out = mx.softmax(mx.array(x).astype(mx.bfloat16), axis=-1, precise=True)
            ref = np.exp(x - x.max(-1, keepdims=True))
            ref /= ref.sum(-1, keepdims=True)
            np.testing.assert_allclose(
                np.array(out.astype(mx.float32)), ref, rtol=2e-2, atol=1e-4
            )

    def test_sort_argsort(self):
        with mx.stream(mx.cpu):
            for shape, axis in [
                ((1031, 517), -1),
                ((1031, 517), 0),
                ((7, 129, 65), 1),
                ((100003,), 0),
            ]:
                x = self.rand(*shape)
                np.testing.assert_array_equal(
                    np.array(mx.sort(mx.array(x), axis=axis)), np.sort(x, axis=axis)
                )
                np.testing.assert_array_equal(
                    np.array(mx.argsort(mx.array(x), axis=axis)),
                    np.argsort(x, axis=axis, kind="stable"),
                )
            # Ties must keep a stable order, NaNs go last
            x = self.rng.integers(0, 5, (517, 1031)).astype(np.float32)
            x[::7, ::11] = np.nan
            np.testing.assert_array_equal(
                np.array(mx.argsort(mx.array(x), axis=-1)),
                np.argsort(x, axis=-1, kind="stable"),
            )
            np.testing.assert_array_equal(
                np.array(mx.sort(mx.array(x), axis=-1)), np.sort(x, axis=-1)
            )

    def test_single_thread_matches(self):
        # Outputs with MLX_CPU_NUM_THREADS=1 must match the default threading
        # bit for bit for elementwise ops, copies and sorts.
        script = (
            "import mlx.core as mx, numpy as np, sys\n"
            "mx.set_default_device(mx.cpu)\n"
            "x = mx.random.normal((1031, 2053), key=mx.random.key(3))\n"
            "y = mx.random.normal((2053,), key=mx.random.key(4))\n"
            "outs = [mx.exp(x), x * y, mx.contiguous(x.T), mx.sort(x, axis=-1),\n"
            "        mx.argsort(x, axis=0), mx.where(x > 0, x, y), mx.max(x, axis=0),\n"
            "        mx.softmax(x, axis=-1)]\n"
            "np.savez(sys.argv[1], *[np.array(o) for o in outs])\n"
        )
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            paths = []
            for nt in ["1", "4"]:
                env = dict(os.environ, MLX_CPU_NUM_THREADS=nt)
                p = os.path.join(d, f"out{nt}.npz")
                subprocess.run([sys.executable, "-c", script, p], env=env, check=True)
                paths.append(p)
            a, b = np.load(paths[0]), np.load(paths[1])
            for k in a.files:
                np.testing.assert_array_equal(a[k], b[k])

    def test_concurrent_streams(self):
        # Two CPU streams issuing parallel kernels at the same time.
        s1 = mx.new_stream(mx.cpu)
        s2 = mx.new_stream(mx.cpu)
        x = mx.array(self.rand(1031, 2053))
        outs = []
        for i in range(20):
            s = s1 if i % 2 == 0 else s2
            with mx.stream(s):
                outs.append(mx.sum(mx.exp(x) * (i + 1), axis=-1))
        mx.eval(outs)
        ref = np.array(mx.sum(mx.exp(x), axis=-1, stream=mx.cpu))
        for i, o in enumerate(outs):
            np.testing.assert_allclose(np.array(o), ref * (i + 1), rtol=1e-5)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
