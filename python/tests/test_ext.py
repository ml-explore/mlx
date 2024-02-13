# Copyright © 2023-2024 Apple Inc.

import math
import unittest

import mlx.core as mx
import mlx_tests


def rope_orig(x, dims, traditional, base, scale, offset):
    N = x.shape[1] + offset
    dtype = x.dtype
    half_D = dims // 2
    positions = mx.arange(offset, N, dtype=dtype) * scale
    freqs = mx.exp(-mx.arange(0.0, half_D, dtype=dtype) * (math.log(base) / half_D))
    theta = mx.reshape(positions, (-1, 1)) * mx.reshape(freqs, (1, -1))
    costheta, sintheta = mx.cos(theta), mx.sin(theta)
    if traditional:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta
        rx = mx.concatenate([rx1[..., None], rx2[..., None]], axis=-1)
        return mx.reshape(rx, x.shape)
    else:
        x1 = x[..., : dims // 2]
        x2 = x[..., dims // 2 : dims]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta
        if dims < x.shape[-1]:
            rx = mx.concatenate([rx1, rx2, x[..., dims:]], axis=-1)
        else:
            rx = mx.concatenate([rx1, rx2], axis=-1)
        return rx


class TestExt(mlx_tests.MLXTestCase):
    def test_rope(self):
        T = 4

        # Defaults: dims, dtype, base, scale, offset, traditional
        defaults = (8, mx.float32, 10000.0, 1.0, 0, False)

        # Test cases:
        dtypes = [mx.float32, mx.float16, mx.bfloat16]
        bases = [10000.0, 1000000.0]
        scales = [1.0, 2.0]
        offsets = [0, 3]
        traditional = [True, False]

        dims, dtype, _, scale, offset, traditional = defaults
        for base in bases:
            x = mx.random.uniform(shape=(2, T, dims)).astype(dtype)
            rx = rope_orig(x, dims, traditional, base, scale, offset)
            rx_ext = mx.ext.rope(x, dims, traditional, base, scale, offset)
            self.assertTrue(mx.allclose(rx, rx_ext))

        dims, _, base, scale, offset, traditional = defaults
        for dtype in dtypes:
            x = mx.random.uniform(shape=(2, T, dims)).astype(dtype)
            rx = rope_orig(x, dims, traditional, base, scale, offset)
            rx_ext = mx.ext.rope(x, dims, traditional, base, scale, offset)
            self.assertTrue(mx.allclose(rx, rx_ext))

        dims, dtype, base, scale, _, traditional = defaults
        for offset in offsets:
            x = mx.random.uniform(shape=(2, T, dims)).astype(dtype)
            rx = rope_orig(x, dims, traditional, base, scale, offset)
            rx_ext = mx.ext.rope(x, dims, traditional, base, scale, offset)
            rx = rope_orig(x, dims, base, scale, traditional, offset)
            rx_ext = mx.ext.rope(x, dims, base, scale, traditional, offset)
            self.assertTrue(mx.allclose(rx, rx_ext))

        dims, dtype, base, _, offset, traditional = defaults
        for scale in scales:
            x = mx.random.uniform(shape=(2, T, dims)).astype(dtype)
            rx = rope_orig(x, dims, traditional, base, scale, offset)
            rx_ext = mx.ext.rope(x, dims, traditional, base, scale, offset)
            self.assertTrue(mx.allclose(rx, rx_ext))

        dims, dtype, base, scale, offset, _ = defaults
        traditional = True
        x = mx.random.uniform(shape=(2, T, dims)).astype(dtype)
        rx = rope_orig(x, dims, traditional, base, scale, offset)
        rx_ext = mx.ext.rope(x, dims, traditional, base, scale, offset)
        self.assertTrue(mx.allclose(rx, rx_ext))

    def test_ext_transforms(self):
        x = mx.random.uniform(shape=(2, 2, 8))

        # Defaults: dims, traditional, base, scale, offset
        defaults = (8, False, 10000.0, 1.0, 0)

        # VJP
        _, vjp_out = mx.vjp(lambda x: rope_orig(x, *defaults), (x,), (mx.ones_like(x),))
        _, vjp_ext_out = mx.vjp(
            lambda x: rope_orig(x, *defaults), (x,), (mx.ones_like(x),)
        )
        self.assertTrue(mx.allclose(vjp_out[0], vjp_ext_out[0]))

        # JVP
        _, jvp_out = mx.jvp(lambda x: rope_orig(x, *defaults), (x,), (mx.ones_like(x),))
        _, jvp_ext_out = mx.jvp(
            lambda x: rope_orig(x, *defaults), (x,), (mx.ones_like(x),)
        )
        self.assertTrue(mx.allclose(jvp_out[0], jvp_ext_out[0]))

        # VMAP
        x = mx.random.uniform(shape=(2, 2, 2, 8))
        vmap_out = mx.vmap(lambda x: rope_orig(x, *defaults))(x)
        vmap_ext_out = mx.vmap(lambda x: rope_orig(x, *defaults))(x)
        self.assertTrue(mx.allclose(vmap_out, vmap_ext_out))


if __name__ == "__main__":
    unittest.main()
