# Copyright © 2023-2024 Apple Inc.

import math
import unittest

import mlx.core as mx
import mlx_tests


def rope_orig(x, dims, traditional, base, scale, offset, freqs=None):
    N = x.shape[-2]
    dtype = x.dtype
    half_D = dims // 2
    positions = mx.arange(N, dtype=dtype)
    if isinstance(offset, mx.array) and offset.size > 1:
        expand = tuple(range(1, x.ndim - 1))
        positions = mx.expand_dims(offset, expand) + positions
    else:
        positions = offset + positions
    positions = positions * scale
    if freqs is None:
        inv_freqs = mx.exp(
            -mx.arange(0.0, half_D, dtype=dtype) * (math.log(base) / half_D)
        )
    else:
        inv_freqs = (1 / freqs).astype(x.dtype)
    theta = mx.expand_dims(positions, -1) * inv_freqs
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
        offsets = [0, 3, mx.array(3)]
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
                    ry = rope_orig(
                        x.astype(mx.float32), dims, traditional, base, scale, offset
                    )
                    self.assertLess(mx.abs(ry - rx_fast).max(), tolerances[dtype])
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

        # Test transpose into rope
        dims, _, base, scale, offset, traditional = defaults
        x = mx.random.uniform(shape=(1, 1, 4, dims)).swapaxes(1, 2)
        rx = rope_orig(x, dims, traditional, base, scale, offset)
        rx_fast = mx.fast.rope(
            1.0 * x,  # multiply here to allow donation
            dims,
            traditional=traditional,
            base=base,
            scale=scale,
            offset=offset,
        )
        self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[mx.float32])

        # Test raises with integer inputs
        dims, _, base, scale, offset, traditional = defaults
        x = (mx.random.uniform(shape=(2, T, dims)) * 10).astype(mx.int32)
        with self.assertRaises(ValueError):
            y = mx.fast.rope(
                x, dims, traditional=traditional, base=base, scale=scale, offset=offset
            )

    def test_rope_with_freqs(self):
        mx.random.seed(0)

        # Check throws
        T = 4
        dims = 8
        x = mx.random.uniform(shape=(2, T, dims))

        with self.assertRaises(ValueError):
            freqs = mx.random.uniform(shape=(dims - 1,))
            mx.fast.rope(
                x,
                dims,
                traditional=False,
                base=None,
                scale=1.0,
                offset=0,
                freqs=freqs,
            )
        with self.assertRaises(ValueError):
            freqs = mx.random.uniform(shape=(1, dims))
            mx.fast.rope(
                x,
                dims,
                traditional=False,
                base=None,
                scale=1.0,
                offset=0,
                freqs=freqs,
            )

        freqs = mx.random.uniform(shape=(dims // 2,))

        tolerances = {mx.float32: 1e-5, mx.float16: 1e-2}
        for dtype in [mx.float32, mx.float16]:
            x_ = x.astype(dtype)
            rx = rope_orig(x_, dims, False, None, 1.0, 0, freqs)
            rx_fast = mx.fast.rope(
                x_,
                dims,
                traditional=False,
                base=None,
                scale=1.0,
                offset=0,
                freqs=freqs,
            )
            self.assertEqual(dtype, rx.dtype)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
            return

        # Test single vector
        x = mx.random.uniform(shape=(1, 1, dims))
        rx = rope_orig(x, dims, False, None, 1.0, 0, freqs)
        rx_fast = mx.fast.rope(
            x,
            dims,
            traditional=False,
            base=None,
            scale=1.0,
            offset=0,
            freqs=freqs,
        )
        self.assertLess(mx.abs(rx - rx_fast).max(), 1e-5)

        # Test grad with freqs
        f1 = lambda x, y: (rope_orig(x, dims, False, None, 1.0, 0, freqs) * y).sum()
        f2 = lambda x, y: (
            mx.fast.rope(
                x,
                dims,
                traditional=False,
                base=None,
                scale=1.0,
                offset=0,
                freqs=freqs,
            )
            * y
        ).sum()

        x = mx.random.uniform(shape=(2, 4, dims))
        y = mx.random.uniform(shape=(2, 4, dims))
        g1 = mx.grad(f1)(x, y)
        g2 = mx.grad(f2)(x, y)
        self.assertLess(mx.abs(g1 - g2).max(), 1e-5)

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

    def test_rope_batch(self):
        T = 4
        base = 10000.0
        scale = 1.0
        traditional = True
        batch_sizes = [3, 8, 11]
        num_heads = [1, 3, 5]
        dims = 32

        x = mx.random.uniform(shape=(8, 4, T, dims))

        offset = mx.array([1, 2, 3])
        with self.assertRaises(ValueError):
            mx.fast.rope(
                x,
                dims,
                traditional=traditional,
                base=base,
                scale=scale,
                offset=offset,
            )

        for batch_size in batch_sizes:
            for n_head in num_heads:
                x = mx.random.uniform(shape=(batch_size, n_head, T, dims))
                offset = mx.arange(batch_size)
                rx = rope_orig(x, dims, traditional, base, scale, offset)
                rx_fast = mx.fast.rope(
                    x,
                    dims,
                    traditional=traditional,
                    base=base,
                    scale=scale,
                    offset=offset,
                )
                self.assertLess(mx.abs(rx - rx_fast).max(), 1e-5)
        x = mx.random.normal(shape=(2, 6, 8, 64)).transpose(0, 2, 1, 3)
        dims = 64
        offset = 0
        rx_fast = mx.fast.rope(
            x, dims, traditional=traditional, scale=scale, base=base, offset=offset
        )
        rx_fast_single = mx.fast.rope(
            x[0:1], dims, traditional=traditional, scale=scale, base=base, offset=offset
        )

        rx = rope_orig(x, dims, traditional, base, scale, offset)
        self.assertLess(mx.abs(rx - rx_fast).max(), 1e-5)

    def test_rope_with_large_offset(self):
        x = mx.random.normal(shape=(1, 1, 1024, 32))
        rx_fp32 = mx.fast.rope(
            x,
            32,
            traditional=False,
            scale=1.0,
            base=10000,
            offset=4000,
        )
        rx_bf16 = mx.fast.rope(
            x.astype(mx.bfloat16),
            32,
            traditional=False,
            scale=1.0,
            base=10000,
            offset=4000,
        )
        self.assertLess((rx_fp32 - rx_bf16).abs().max(), 1e-1)

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
            rx = rms_norm(x, mx.ones_like(weight), eps)
            rx_fast = mx.fast.rms_norm(x, None, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])

        for eps in epss:
            dtype, _, dims = defaults
            x = mx.random.uniform(shape=(2, dims)).astype(dtype)
            weight = mx.random.uniform(shape=(dims,)).astype(dtype)
            rx = rms_norm(x, weight, eps)
            rx_fast = mx.fast.rms_norm(x, weight, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = rms_norm(x, mx.ones_like(weight), eps)
            rx_fast = mx.fast.rms_norm(x, None, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])

        for dims in dimss:
            dtype, eps, _ = defaults
            x = mx.random.uniform(shape=(2, dims)).astype(dtype)
            weight = mx.random.uniform(shape=(dims,)).astype(dtype)
            rx = rms_norm(x, weight, eps)
            rx_fast = mx.fast.rms_norm(x, weight, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])
            rx = rms_norm(x, mx.ones_like(weight), eps)
            rx_fast = mx.fast.rms_norm(x, None, eps)
            self.assertLess(mx.abs(rx - rx_fast).max(), tolerances[dtype])

        # Test > 4096
        dims, dtype, eps = 4099, mx.float32, 1e-5
        x = mx.random.uniform(shape=(dims,)).astype(dtype)
        weight = mx.random.uniform(shape=(dims,)).astype(dtype)
        rx = rms_norm(x, weight, eps)
        rx_fast = mx.fast.rms_norm(x, weight, eps)
        self.assertLess(mx.abs(rx - rx_fast).max(), 1e-6)

        # Wrong size w raises
        with self.assertRaises(ValueError):
            x = mx.random.uniform(shape=(1, 5))
            mx.fast.rms_norm(x, mx.ones((4,)), 1e-5)

    def test_rms_norm_grad(self):
        D = 32
        eps = 1e-5
        f1 = lambda x, w, y: (rms_norm(x, w, eps) * y).sum()
        f2 = lambda x, w, y: (mx.fast.rms_norm(x, w, eps) * y).sum()
        f3 = lambda x, y: (rms_norm(x, mx.ones((x.shape[-1],)), eps) * y).sum()
        f4 = lambda x, y: (mx.fast.rms_norm(x, None, eps) * y).sum()

        x = mx.random.uniform(shape=(8, 100, D))
        w = mx.random.uniform(shape=(D,))
        y = mx.random.uniform(shape=(8, 100, D))
        gx1, gw1 = mx.grad(f1, argnums=(0, 1))(x, w, y)
        gx2, gw2 = mx.grad(f2, argnums=(0, 1))(x, w, y)
        self.assertLess(mx.abs(gx1 - gx2).max(), 1e-5)
        self.assertLess(mx.abs(gw1 - gw2).max() / mx.abs(gw1).mean(), 1e-5)
        gx1 = mx.grad(f3, argnums=(0,))(x, y)
        gx2 = mx.grad(f4, argnums=(0,))(x, y)
        self.assertLess(mx.abs(gx1 - gx2).max(), 1e-5)

        D = 8192
        x = mx.random.uniform(shape=(2, 2, D))
        w = mx.random.uniform(shape=(D,))
        y = mx.random.uniform(shape=(2, 2, D))
        gx1, gw1 = mx.grad(f1, argnums=(0, 1))(x, w, y)
        gx2, gw2 = mx.grad(f2, argnums=(0, 1))(x, w, y)
        self.assertLess(mx.abs(gx1 - gx2).max(), 1e-5)
        self.assertLess(mx.abs(gw1 - gw2).max() / mx.abs(gw1).mean(), 1e-5)
        gx1 = mx.grad(f3, argnums=(0,))(x, y)
        gx2 = mx.grad(f4, argnums=(0,))(x, y)
        self.assertLess(mx.abs(gx1 - gx2).max(), 1e-5)

        def gf(f):
            def inner(x, w, y):
                gx, gw = mx.grad(f, argnums=(0, 1))(x, w, y)
                return (gx + gw).sum()

            return inner

        gx1, gw1 = mx.grad(gf(f1), argnums=(0, 1))(x, w, y)
        gx2, gw2 = mx.grad(gf(f2), argnums=(0, 1))(x, w, y)
        self.assertLess(mx.abs(gx1 - gx2).max(), 1e-5)
        self.assertLess(mx.abs(gw1 - gw2).max() / mx.abs(gw1).mean(), 1e-5)

    def test_layer_norm_dim_check(self):
        with self.assertRaises(ValueError):
            weight = mx.ones((129,))
            x = mx.random.randint(low=0, high=10, shape=(4, 128))
            mx.fast.layer_norm(x, weight, None, 1e-3)

        with self.assertRaises(ValueError):
            bias = mx.ones((129,))
            x = mx.random.randint(low=0, high=10, shape=(4, 128))
            mx.fast.layer_norm(x, None, bias, 1e-3)

    def test_layer_norm(self):
        # Per dtype absolute tolerance
        tolerances = {mx.float32: 1e-5, mx.float16: 5e-3, mx.bfloat16: 5e-2}

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

    def test_slice_into_layer_norm(self):
        dim = 128
        eps = 1e-5
        x = mx.random.uniform(shape=(8, 100, 128))[:, 99:]
        rx_fast = mx.fast.layer_norm(x, weight=None, bias=None, eps=eps)
        rx = layer_norm(x, None, None, eps)
        self.assertLess(mx.abs(rx - rx_fast).max(), 1e-4)

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
        self.assertLess(mx.abs(gx1 - gx2).max(), 5e-5)
        self.assertLess(mx.abs(gw1 - gw2).max() / mx.abs(gw1).mean(), 5e-5)
        self.assertLess(mx.abs(gb1 - gb2).max() / mx.abs(gb1).mean(), 5e-5)

        def gf(f):
            def inner(x, w, b, y):
                gx, gw, gb = mx.grad(f, argnums=(0, 1, 2))(x, w, b, y)
                return ((gx + gw + gb) * y).sum()

            return inner

        gx1, gw1, gb1 = mx.grad(gf(f1), argnums=(0, 1, 2))(x, w, b, y)
        gx2, gw2, gb2 = mx.grad(gf(f2), argnums=(0, 1, 2))(x, w, b, y)
        self.assertLess(mx.abs(gx1 - gx2).max() / mx.abs(gx1).mean(), 5e-5)
        self.assertLess(mx.abs(gw1 - gw2).max() / mx.abs(gw1).mean(), 5e-5)
        self.assertLess(mx.abs(gb1).max(), 1e-9)
        self.assertLess(mx.abs(gb2).max(), 1e-9)

    def test_layer_norm_grad_no_params(self):
        eps = 1e-5
        f1 = lambda x: layer_norm(x, None, None, eps).sum()
        f2 = lambda x: mx.fast.layer_norm(x, None, None, eps).sum()
        x = mx.random.normal(shape=(2, 2, 8))
        mx.eval(x)

        gx1 = mx.grad(f1)(x)
        gx2 = mx.grad(f2)(x)
        self.assertTrue(mx.allclose(gx1, gx2, atol=1e-6))

    def test_layer_norm_grad_params(self):
        eps = 1e-5
        f1 = lambda params, x: (layer_norm(x, params[0], params[1], eps)).sum()
        f2 = lambda params, x: (mx.fast.layer_norm(x, params[0], params[1], eps)).sum()

        w = mx.ones((8,))
        b = mx.zeros((8,))
        x = mx.random.normal(shape=(2, 2, 8))
        mx.eval(x, w, b)

        gw1, gb1 = mx.grad(f1)((w, b), x)
        gw2, gb2 = mx.grad(f2)((w, b), x)
        self.assertLess(mx.abs(gw1 - gw2).max() / mx.abs(gw1).mean(), 1e-5)
        self.assertLess(mx.abs(gb1 - gb2).max() / mx.abs(gb1).mean(), 1e-5)

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

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_custom_kernel_basic(self):
        if mx.metal.is_available():
            source = """
                uint elem = thread_position_in_grid.x;
                out1[elem] = a[elem];
            """
            custom_kernel = mx.fast.metal_kernel
        elif mx.cuda.is_available():
            source = """
                auto elem = cooperative_groups::this_grid().thread_rank();
                out1[elem] = a[elem];
            """
            custom_kernel = mx.fast.cuda_kernel

        mx.random.seed(7)
        a = mx.random.normal(shape=(2, 2))
        kernel = custom_kernel(
            name="basic",
            input_names=["a"],
            output_names=["out1"],
            source=source,
        )
        out = kernel(
            inputs=[a],
            grid=(4, 1, 1),
            threadgroup=(2, 1, 1),
            output_shapes=[(2, 2)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )
        self.assertTrue(mx.allclose(out[0], a))

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_custom_kernel_args(self):
        if mx.metal.is_available():
            source = """
                uint elem = thread_position_in_grid.x;
                T tmp = a[0];
                if (e) {
                    out1[elem] = a[1] + b[2] + c[3] + d + f;
                } else {
                    out1[elem] = 1;
                }
                out2[elem] = a[1] + b[2] + c[1] - d;
            """
            custom_kernel = mx.fast.metal_kernel
        elif mx.cuda.is_available():
            source = """
                auto elem = cooperative_groups::this_grid().thread_rank();
                T tmp = a[0];
                if (e) {
                    out1[elem] = a[1] + b[2] + static_cast<float>(c[3]) + d[0] + f;
                } else {
                    out1[elem] = 1;
                }
                out2[elem] = a[1] + b[2] + static_cast<float>(c[1]) - d[0];
            """
            custom_kernel = mx.fast.cuda_kernel

        mx.random.seed(7)
        a = mx.random.normal(shape=(3, 6))
        c = mx.random.normal(shape=(2, 2)).astype(mx.bfloat16)

        kernel = custom_kernel(
            name="arg_test",
            input_names=["a", "b", "c", "d"],
            output_names=["out1", "out2"],
            source=source,
        )
        out = kernel(
            inputs=[
                a,
                mx.array([3, 4, 5]),
                c,
                7.3,
            ],
            template=[
                ("e", True),
                ("f", 3),
                ("T", mx.float16),
            ],
            grid=(6, 1, 1),
            threadgroup=(2, 1, 1),
            output_shapes=[(3, 2), (3, 2)],
            output_dtypes=[mx.float32, mx.int32],
            stream=mx.gpu,
        )

        self.assertTrue(mx.allclose(out[0], mx.full((3, 2), 14.0484)))
        self.assertTrue(mx.allclose(out[1], mx.full((3, 2), -2, dtype=mx.int32)))

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_custom_kernel_strides(self):
        if mx.metal.is_available():
            source = """
                uint elem = thread_position_in_grid.x;
                uint loc = elem_to_loc(elem, inp_shape, inp_strides, inp_ndim);
                T tmp = inp[loc];
                out[elem] = metal::precise::exp(tmp) * threads_per_simdgroup;
            """
            source_contig = """
                uint elem = thread_position_in_grid.x;
                T tmp = inp[elem];
                out[elem] = metal::precise::exp(tmp) * threads_per_simdgroup;
            """
            custom_kernel = mx.fast.metal_kernel
        elif mx.cuda.is_available():
            source = """
                auto elem = cooperative_groups::this_grid().thread_rank();
                auto loc = elem_to_loc(elem, inp_shape.data(), inp_strides.data(), inp_ndim);
                T tmp = inp[loc];
                out[elem] = exp(tmp) * WARP_SIZE;
            """
            source_contig = """
                auto elem = cooperative_groups::this_grid().thread_rank();
                T tmp = inp[elem];
                out[elem] = exp(tmp) * WARP_SIZE;
            """
            custom_kernel = mx.fast.cuda_kernel

        mx.random.seed(7)
        a = mx.random.normal(shape=(3, 6))

        # non contiguous
        a = mx.tile(a[::2], [4, 1])

        for contig in [True, False]:
            kernel = custom_kernel(
                name="myexp" + str(contig),
                input_names=["inp"],
                output_names=["out"],
                source=source_contig if contig else source,
                ensure_row_contiguous=contig,
            )
            outputs = kernel(
                inputs=[a],
                template=[("T", mx.float32)],
                grid=(a.size, 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[a.shape],
                output_dtypes=[a.dtype],
                stream=mx.gpu,
            )
            self.assertTrue(mx.allclose(mx.exp(a) * 32, outputs[0]))

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_custom_kernel_helper(self):
        if mx.metal.is_available():
            header = """
            template <typename T>
            T do_exp(T x) {
                return metal::precise::exp(x);
            }
            """
            source = """
                uint elem = thread_position_in_grid.x;
                out1[elem] = do_exp(a[elem]);
            """
            custom_kernel = mx.fast.metal_kernel
        elif mx.cuda.is_available():
            header = """
            template <typename T>
            __device__ T do_exp(T x) {
                return exp(x);
            }
            """
            source = """
                auto elem = cooperative_groups::this_grid().thread_rank();
                out1[elem] = do_exp(a[elem]);
            """
            custom_kernel = mx.fast.cuda_kernel

        mx.random.seed(7)
        a = mx.random.normal(shape=(2, 2))
        kernel = custom_kernel(
            name="helper",
            input_names=["a"],
            output_names=["out1"],
            header=header,
            source=source,
        )
        out = kernel(
            inputs=[a],
            grid=(4, 1, 1),
            threadgroup=(2, 1, 1),
            output_shapes=[(2, 2)],
            output_dtypes=[mx.float32],
            stream=mx.gpu,
        )
        self.assertTrue(mx.allclose(out[0], mx.exp(a)))

    @unittest.skipIf(not mx.is_available(mx.gpu), "No GPU available")
    def test_custom_kernel_attributes(self):
        if mx.metal.is_available():
            source = "out[0] = threads_per_threadgroup.x;"
            custom_kernel = mx.fast.metal_kernel
        elif mx.cuda.is_available():
            source = "out[0] = blockDim.x;"
            custom_kernel = mx.fast.cuda_kernel

        a = mx.zeros(shape=(1, 1))
        kernel = custom_kernel(
            name="test_fun",
            input_names=["a"],
            output_names=["out"],
            source=source,
        )
        out = kernel(
            inputs=[a],
            grid=(2, 1, 1),
            threadgroup=(2, 1, 1),
            output_shapes=[(1, 1)],
            output_dtypes=[mx.uint32],
            stream=mx.gpu,
        )[0]
        self.assertEqual(out.item(), 2)

    @unittest.skipIf(not mx.metal.is_available(), "Metal is not available")
    def test_custom_kernel_caching(self):
        def call_kernel(a: mx.array, source):
            kernel = mx.fast.metal_kernel(
                name="my_kernel",
                input_names=["inp"],
                output_names=["out"],
                source=source,
            )
            return kernel(
                inputs=[a],
                grid=(a.size, 1, 1),
                threadgroup=(a.size, 1, 1),
                output_shapes=[a.shape],
                output_dtypes=[a.dtype],
                stream=mx.gpu,
            )[0]

        a = mx.random.normal(shape=(32,))

        source = """
            uint elem = thread_position_in_grid.x;
            out[elem] = 0.0;
        """

        out = call_kernel(a, source)
        self.assertTrue(mx.array_equal(out, mx.zeros_like(out)))

        source = """
            uint elem = thread_position_in_grid.x;
            out[elem] = 1.0;
        """
        out = call_kernel(a, source)
        self.assertTrue(mx.array_equal(out, mx.ones_like(out)))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
