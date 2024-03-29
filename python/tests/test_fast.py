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
        x1 = x[..., :dims:2]
        x2 = x[..., 1:dims:2]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta
        rx = mx.concatenate([rx1[..., None], rx2[..., None]], axis=-1)
        if dims < x.shape[-1]:
            rx = mx.reshape(rx, (*x.shape[:-1], dims))
            rx = mx.concatenate([rx, x[..., dims:]], axis=-1)
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


def rms_norm(x, weight, eps):
    x = x.astype(mx.float32)
    x = x * mx.rsqrt(x.square().mean(-1, keepdims=True) + eps)
    return weight * x.astype(weight.dtype)


def layer_norm(x, weight, bias, eps):
    ot = x.dtype
    x = x.astype(mx.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mean) * mx.rsqrt(var + eps)
    x = x.astype(ot)
    if weight is not None:
        x = x * weight
    if bias is not None:
        x = x + bias
    return x


class TestFast(mlx_tests.MLXTestCase):
    def test_rope(self):
        T = 4

        # Defaults: dims, dtype, base, scale, offset, traditional
        defaults = (8, mx.float32, 10000.0, 1.0, 0, False)

        # Per dtype absolute tolerance
        tolerances = {mx.float32: 1e-6, mx.float16: 1e-3, mx.bfloat16: 1e-2}

        # Test cases:
        dtypes = [mx.float32, mx.float16, mx.bfloat16]
        bases = [10000.0, 1000000.0]
        scales = [1.0, 2.0]
        offsets = [0, 3]
        traditional = [True, False]

        for traditional in [True, False]:
            dims, dtype, _, scale, offset, _ = defaults
            for base in bases:
                x = mx.random.uniform(shape=(2, T, dims)).astype(dtype)
                rx = rope_orig(x, dims, traditional, base, scale, offset)
                rx_fast = mx.fast.rope(
                    x,
                    dims,
                    traditional=traditional,
                    base=base,
                    scale=scale,
                    offset=offset,
                )
                self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])

            dims, _, base, scale, offset, _ = defaults
            for dtype in dtypes:
                x = mx.random.uniform(shape=(2, T, dims)).astype(dtype)
                ry = rope_orig(
                    x.astype(mx.float32), dims, traditional, base, scale, offset
                )
                rx = rope_orig(x, dims, traditional, base, scale, offset)
                rx_fast = mx.fast.rope(
                    x,
                    dims,
                    traditional=traditional,
                    base=base,
                    scale=scale,
                    offset=offset,
                )
                if dtype != mx.float32:
                    self.assertLessEqual(
                        mx.abs(ry - rx_fast).max(), mx.abs(ry - rx).max()
                    )
                self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])

            dims, dtype, base, scale, _, _ = defaults
            for offset in offsets:
                x = mx.random.uniform(shape=(2, T, dims)).astype(dtype)
                rx = rope_orig(x, dims, traditional, base, scale, offset)
                rx_fast = mx.fast.rope(
                    x,
                    dims,
                    traditional=traditional,
                    base=base,
                    scale=scale,
                    offset=offset,
                )
                self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])

            dims, dtype, base, _, offset, _ = defaults
            for scale in scales:
                x = mx.random.uniform(shape=(2, T, dims)).astype(dtype)
                rx = rope_orig(x, dims, traditional, base, scale, offset)
                rx_fast = mx.fast.rope(
                    x,
                    dims,
                    traditional=traditional,
                    base=base,
                    scale=scale,
                    offset=offset,
                )
                self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])

    def test_rope_grad(self):
        D = 32
        defaults = (D, 10000.0, 1.0, 0, False)
        for dims in (D, D // 2):
            for traditional in (True, False):
                _, base, scale, offset, _ = defaults
                f1 = lambda x, y: (
                    rope_orig(x, dims, traditional, base, scale, offset) * y
                ).sum()
                f2 = lambda x, y: (
                    mx.fast.rope(
                        x,
                        dims,
                        traditional=traditional,
                        base=base,
                        scale=scale,
                        offset=offset,
                    )
                    * y
                ).sum()

                x = mx.random.uniform(shape=(2, 100, D))
                y = mx.random.uniform(shape=(2, 100, D))
                g1 = mx.grad(f1)(x, y)
                g2 = mx.grad(f2)(x, y)
                self.assertLess(mx.abs(g1 - g2).max(), 1e-5)

    def test_rms_norm(self):
        # Per dtype absolute tolerance
        tolerances = {mx.float32: 1e-6, mx.float16: 1e-3, mx.bfloat16: 1e-2}

        dtypes = [mx.float32, mx.float16, mx.bfloat16]
        epss = [1e-3, 1e-5]
        dimss = [31, 32, 33]
        defaults = (mx.float32, 1e-5, 32)

        for dtype in dtypes:
            _, eps, dims = defaults
            x = mx.random.uniform(
                shape=(
                    2,
                    dims,
                )
            ).astype(dtype)
            weight = mx.random.uniform(shape=(dims,)).astype(dtype)
            rx = rms_norm(x, weight, eps)
            rx_fast = mx.fast.rms_norm(x, weight, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])

        for eps in epss:
            dtype, _, dims = defaults
            x = mx.random.uniform(shape=(2, dims)).astype(dtype)
            weight = mx.random.uniform(shape=(dims,)).astype(dtype)
            rx = rms_norm(x, weight, eps)
            rx_fast = mx.fast.rms_norm(x, weight, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])

        for dims in dimss:
            dtype, eps, _ = defaults
            x = mx.random.uniform(shape=(2, dims)).astype(dtype)
            weight = mx.random.uniform(shape=(dims,)).astype(dtype)
            rx = rms_norm(x, weight, eps)
            rx_fast = mx.fast.rms_norm(x, weight, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])

        # Test > 4096
        dims, dtype, eps = 4099, mx.float32, 1e-5
        x = mx.random.uniform(shape=(dims,)).astype(dtype)
        weight = mx.random.uniform(shape=(dims,)).astype(dtype)
        rx = rms_norm(x, weight, eps)
        rx_fast = mx.fast.rms_norm(x, weight, eps)
        self.assertLess(mx.abs(rx - rx_fast).max(), 1e-6)

    def test_rms_norm_grad(self):
        D = 32
        eps = 1e-5
        f1 = lambda x, w, y: (rms_norm(x, w, eps) * y).sum()
        f2 = lambda x, w, y: (mx.fast.rms_norm(x, w, eps) * y).sum()

        x = mx.random.uniform(shape=(8, 100, D))
        w = mx.random.uniform(shape=(D,))
        y = mx.random.uniform(shape=(8, 100, D))
        gx1, gw1 = mx.grad(f1, argnums=(0, 1))(x, w, y)
        gx2, gw2 = mx.grad(f2, argnums=(0, 1))(x, w, y)
        self.assertLess(mx.abs(gx1 - gx2).max(), 1e-5)
        self.assertLess(mx.abs(gw1 - gw2).max() / mx.abs(gw1).mean(), 1e-5)

        D = 8192
        x = mx.random.uniform(shape=(2, 2, D))
        w = mx.random.uniform(shape=(D,))
        y = mx.random.uniform(shape=(2, 2, D))
        gx1, gw1 = mx.grad(f1, argnums=(0, 1))(x, w, y)
        gx2, gw2 = mx.grad(f2, argnums=(0, 1))(x, w, y)
        self.assertLess(mx.abs(gx1 - gx2).max(), 1e-5)
        self.assertLess(mx.abs(gw1 - gw2).max() / mx.abs(gw1).mean(), 1e-5)

        def gf(f):
            def inner(x, w, y):
                gx, gw = mx.grad(f, argnums=(0, 1))(x, w, y)
                return (gx + gw).sum()

            return inner

        gx1, gw1 = mx.grad(gf(f1), argnums=(0, 1))(x, w, y)
        gx2, gw2 = mx.grad(gf(f2), argnums=(0, 1))(x, w, y)
        self.assertLess(mx.abs(gx1 - gx2).max(), 1e-5)
        self.assertLess(mx.abs(gw1 - gw2).max() / mx.abs(gw1).mean(), 1e-5)

    def test_layer_norm(self):
        # Per dtype absolute tolerance
        tolerances = {mx.float32: 3e-6, mx.float16: 3e-3, mx.bfloat16: 3e-2}

        dtypes = [mx.float32, mx.float16, mx.bfloat16]
        epss = [1e-3, 1e-5]
        dimss = [31, 32, 33]
        defaults = (mx.float32, 1e-5, 32)

        for dtype in dtypes:
            _, eps, dims = defaults
            x = mx.random.uniform(
                shape=(
                    2,
                    dims,
                )
            ).astype(dtype)
            weight = mx.random.uniform(shape=(dims,)).astype(dtype)
            bias = mx.random.uniform(shape=(dims,)).astype(dtype)
            rx = layer_norm(x, weight, bias, eps)
            rx_fast = mx.fast.layer_norm(x, weight, bias, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, weight, None, eps)
            rx_fast = mx.fast.layer_norm(x, weight, None, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, None, bias, eps)
            rx_fast = mx.fast.layer_norm(x, None, bias, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, None, None, eps)
            rx_fast = mx.fast.layer_norm(x, None, None, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])

        for eps in epss:
            dtype, _, dims = defaults
            x = mx.random.uniform(shape=(2, dims)).astype(dtype)
            weight = mx.random.uniform(shape=(dims,)).astype(dtype)
            bias = mx.random.uniform(shape=(dims,)).astype(dtype)
            rx = layer_norm(x, weight, bias, eps)
            rx_fast = mx.fast.layer_norm(x, weight, bias, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, weight, None, eps)
            rx_fast = mx.fast.layer_norm(x, weight, None, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, None, bias, eps)
            rx_fast = mx.fast.layer_norm(x, None, bias, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, None, None, eps)
            rx_fast = mx.fast.layer_norm(x, None, None, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])

        for dims in dimss:
            dtype, eps, _ = defaults
            x = mx.random.uniform(shape=(2, dims)).astype(dtype)
            weight = mx.random.uniform(shape=(dims,)).astype(dtype)
            bias = mx.random.uniform(shape=(dims,)).astype(dtype)
            rx = layer_norm(x, weight, bias, eps)
            rx_fast = mx.fast.layer_norm(x, weight, bias, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, weight, None, eps)
            rx_fast = mx.fast.layer_norm(x, weight, None, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, None, bias, eps)
            rx_fast = mx.fast.layer_norm(x, None, bias, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = layer_norm(x, None, None, eps)
            rx_fast = mx.fast.layer_norm(x, None, None, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])

        # Test > 4096
        dims, dtype, eps = 4099, mx.float32, 1e-5
        x = mx.random.uniform(shape=(dims,)).astype(dtype)
        weight = mx.random.uniform(shape=(dims,)).astype(dtype)
        bias = mx.random.uniform(shape=(dims,)).astype(dtype)
        rx = layer_norm(x, weight, bias, eps)
        rx_fast = mx.fast.layer_norm(x, weight, bias, eps)
        self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
        rx = layer_norm(x, weight, None, eps)
        rx_fast = mx.fast.layer_norm(x, weight, None, eps)
        self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
        rx = layer_norm(x, None, bias, eps)
        rx_fast = mx.fast.layer_norm(x, None, bias, eps)
        self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
        rx = layer_norm(x, None, None, eps)
        rx_fast = mx.fast.layer_norm(x, None, None, eps)
        self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])

    def test_layer_norm_grad(self):
        D = 32
        eps = 1e-5
        f1 = lambda x, w, b, y: (layer_norm(x, w, b, eps) * y).sum()
        f2 = lambda x, w, b, y: (mx.fast.layer_norm(x, w, b, eps) * y).sum()

        x = mx.random.uniform(shape=(8, 100, D))
        w = mx.random.uniform(shape=(D,))
        b = mx.random.uniform(shape=(D,))
        y = mx.random.uniform(shape=(8, 100, D))

        gx1, gw1, gb1 = mx.grad(f1, argnums=(0, 1, 2))(x, w, b, y)
        gx2, gw2, gb2 = mx.grad(f2, argnums=(0, 1, 2))(x, w, b, y)
        self.assertLess(mx.abs(gx1 - gx2).max(), 1e-5)
        self.assertLess(mx.abs(gw1 - gw2).max() / mx.abs(gw1).mean(), 1e-5)
        self.assertLess(mx.abs(gb1 - gb2).max() / mx.abs(gb1).mean(), 1e-5)

        D = 8192
        x = mx.random.uniform(shape=(8, 100, D))
        w = mx.random.uniform(shape=(D,))
        b = mx.random.uniform(shape=(D,))
        y = mx.random.uniform(shape=(8, 100, D))

        gx1, gw1, gb1 = mx.grad(f1, argnums=(0, 1, 2))(x, w, b, y)
        gx2, gw2, gb2 = mx.grad(f2, argnums=(0, 1, 2))(x, w, b, y)
        self.assertLess(mx.abs(gx1 - gx2).max(), 1e-5)
        self.assertLess(mx.abs(gw1 - gw2).max() / mx.abs(gw1).mean(), 1e-5)
        self.assertLess(mx.abs(gb1 - gb2).max() / mx.abs(gb1).mean(), 1e-5)

        def gf(f):
            def inner(x, w, b, y):
                gx, gw, gb = mx.grad(f, argnums=(0, 1, 2))(x, w, b, y)
                return ((gx + gw + gb) * y).sum()

            return inner

        gx1, gw1, gb1 = mx.grad(gf(f1), argnums=(0, 1, 2))(x, w, b, y)
        gx2, gw2, gb2 = mx.grad(gf(f2), argnums=(0, 1, 2))(x, w, b, y)
        self.assertLess(mx.abs(gx1 - gx2).max() / mx.abs(gx1).mean(), 1e-5)
        self.assertLess(mx.abs(gw1 - gw2).max() / mx.abs(gw1).mean(), 1e-5)
        self.assertLess(mx.abs(gb1).max(), 1e-9)
        self.assertLess(mx.abs(gb2).max(), 1e-9)

    def test_fast_transforms(self):
        x = mx.random.uniform(shape=(2, 2, 8))

        defaults = (8, False, 10000.0, 1.0, 0)
        dims, traditional, base, scale, offset = defaults

        # VJP
        _, vjp_out = mx.vjp(lambda x: rope_orig(x, *defaults), (x,), (mx.ones_like(x),))
        _, vjp_fast_out = mx.vjp(
            lambda x: mx.fast.rope(
                x, dims, traditional=traditional, base=base, scale=scale, offset=offset
            ),
            (x,),
            (mx.ones_like(x),),
        )
        self.assertTrue(mx.allclose(vjp_out[0], vjp_fast_out[0]))

        # JVP
        _, jvp_out = mx.jvp(lambda x: rope_orig(x, *defaults), (x,), (mx.ones_like(x),))
        _, jvp_fast_out = mx.jvp(
            lambda x: mx.fast.rope(
                x, dims, traditional=traditional, base=base, scale=scale, offset=offset
            ),
            (x,),
            (mx.ones_like(x),),
        )
        self.assertTrue(mx.allclose(jvp_out[0], jvp_fast_out[0]))

        # VMAP
        x = mx.random.uniform(shape=(2, 2, 2, 8))
        vmap_out = mx.vmap(lambda x: rope_orig(x, *defaults))(x)
        vmap_fast_out = mx.vmap(
            lambda x: mx.fast.rope(
                x, dims, traditional=traditional, base=base, scale=scale, offset=offset
            )
        )(x)
        self.assertTrue(mx.allclose(vmap_out, vmap_fast_out))


if __name__ == "__main__":
    unittest.main()
