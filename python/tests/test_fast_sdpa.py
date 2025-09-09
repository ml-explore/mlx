import math
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


def mlx_ref_attn(q, k, v, scale=1.0, mask=None, sinks=None):
    q_dtype = q.dtype
    q = q * mx.array(scale, q_dtype)
    n_q_heads = q.shape[-3]
    n_kv_heads = k.shape[-3]
    n_repeats = n_q_heads // n_kv_heads

    B = q.shape[0]
    L = q.shape[2]
    kL = k.shape[2]

    if n_repeats > 1:
        q = mx.reshape(q, [B, n_kv_heads, n_repeats, L, -1])
        k = mx.expand_dims(k, 2)
        v = mx.expand_dims(v, 2)

    scores = q @ mx.swapaxes(k, -1, -2)
    if mask is not None:

        if mask == "causal":
            q_offset = max(0, kL - L)
            q_indices = mx.arange(q_offset, q_offset + L)
            k_indices = mx.arange(kL)
            mask = q_indices[:, None] >= k_indices[None]

        if n_repeats > 1 and mask.ndim >= 3:
            if mask.shape[-3] == 1:
                mask = mx.expand_dims(mask, -3)
            else:
                mask = mx.unflatten(mask, -3, (n_kv_heads, n_repeats))

        if mask.dtype == mx.bool_:
            scores = mx.where(mask, scores, -np.float32(np.inf))
        else:
            scores += mask

    if sinks is not None:
        sinks = mx.expand_dims(sinks, (0, 2, 3))
        if n_repeats > 1:
            sinks = mx.unflatten(sinks, 1, (n_kv_heads, n_repeats))
        score_shape = list(scores.shape)
        score_shape[-1] = 1
        sinks = mx.broadcast_to(sinks, score_shape)
        scores = mx.concatenate([sinks, scores], axis=-1)

    scores = mx.softmax(scores, axis=-1, precise=True)
    if sinks is not None:
        scores = scores[..., 1:]

    out = scores @ v
    if n_repeats > 1:
        out = mx.reshape(out, [B, n_q_heads, L, -1])

    return out


def do_attention(f, q, k, v, scale, mask=None, transpose=False):
    if transpose:
        q_t = mx.transpose(q, (0, 2, 1, 3))
        k_t = mx.transpose(k, (0, 2, 1, 3))
        v_t = mx.transpose(v, (0, 2, 1, 3))
        o_t = f(q_t, k_t, v_t, scale=scale, mask=mask)
        return mx.transpose(o_t, (0, 2, 1, 3))
    else:
        return f(q, k, v, scale=scale, mask=mask)


def prepare_inputs(B, qL, kL, D, qH, kH, mask, transpose, dtype):
    np.random.seed(0)
    np_dtype = getattr(np, dtype)

    shape_q = (B, qL, qH, D) if transpose else (B, qH, qL, D)
    shape_kv = (B, kL, kH, D) if transpose else (B, kH, kL, D)

    scale = 1.0 / math.sqrt(D)

    q_np = np.random.normal(0.0, 0.5, shape_q).astype(np_dtype)
    k_np = np.random.normal(0.0, 0.5, shape_kv).astype(np_dtype)
    v_np = np.random.normal(0.0, scale, shape_kv).astype(np_dtype)

    q_mx = mx.array(q_np)
    k_mx = mx.array(k_np)
    v_mx = mx.array(v_np)

    if mask is not None:
        if mask == "additive":
            mask_np = np.random.normal(0.0, 0.5, (B, qH, qL, kL)).astype(np_dtype)
            mask = mx.array(mask_np)
        elif mask == "bool":
            mask_np = np.random.uniform(0.0, 1.0, (B, qH, qL, kL)) < 0.5
            mask = mx.array(mask_np)

    return q_mx, k_mx, v_mx, scale, mask


# SDPA for MHA (n_heads == n_kv_heads)
def mlx_primitives_sdpa(q, k, v, scale, mask=None):
    p = (q * scale) @ k.transpose(0, 1, 3, 2)
    if mask is not None:
        if mask == "causal":
            q_offset = max(0, k.shape[2] - q.shape[2])
            q_indices = mx.arange(q_offset, q_offset + q.shape[2])
            k_indices = mx.arange(k.shape[2])
            mask = q_indices[:, None] >= k_indices[None]
            p = mx.where(mask, p, mx.finfo(mx.float32).min)
        elif mask.dtype == mx.bool_:
            p = mx.where(mask, p, mx.finfo(mx.float32).min)
        else:
            p += mask
    scores = mx.softmax(p.astype(mx.float32), axis=-1).astype(p.dtype)
    return scores @ v


# SDPA for GQA (n_heads > n_kv_heads, n_kv_heads > 1, n_heads % n_kv_heads == 0)
def mlx_primitives_sdpa_with_gqa(q, k, v, scale, mask=None):
    n_repeats = q.shape[1] // k.shape[1]

    # borrowing kv cache tiling from mlx-examples/llms/mistral/mistral.py
    n_heads = q.shape[1]
    B = q.shape[0]
    L = k.shape[2]

    def repeat(a):
        a = mx.concatenate([mx.expand_dims(a, 2)] * n_repeats, axis=2)
        return a.reshape([B, n_heads, L, -1])

    k, v = map(repeat, (k, v))

    return mlx_primitives_sdpa(q, k, v, scale, mask=mask)


class TestFastSelfAttentionSDPA(mlx_tests.MLXTestCase):
    def test_fast_sdpa(self):
        # Not yet supported:
        # * K pre-transposed in kernel, V pre-transposed in kernel
        np.random.seed(0)
        R = 20
        L = R
        Dk = 64
        H = 3
        scale = float(1.0 / np.sqrt(Dk))
        q_npy = np.random.normal(0.0, 1.0, (1, H, R, Dk)).astype(np.float32)
        k_npy = np.random.normal(0.0, 1.0, (1, H, L, Dk)).astype(np.float32)
        v_npy = np.random.normal(0.0, 1.0, (1, H, L, Dk)).astype(np.float32)

        q_mlx = mx.array(q_npy)
        k_mlx = mx.array(k_npy)
        v_mlx = mx.array(v_npy)

        reference = mlx_primitives_sdpa(q_mlx, k_mlx, v_mlx, scale)

        o_mlx = mx.fast.scaled_dot_product_attention(
            q_mlx, k_mlx, v_mlx, scale=scale, mask=None
        )

        self.assertListEqual(list(reference.shape), list(o_mlx.shape))
        self.assertTrue(mx.allclose(o_mlx, reference, atol=1e-4))

        dtypes = [np.float32]

        Dk = 64

        if self.is_apple_silicon or mx.cuda.is_available():
            dtypes.append(np.half)

        for SEQUENCE_LENGTH in [63, 129, 400]:
            for DTYPE in dtypes:
                B = 2
                H = 24
                n_kv_heads = H
                q_npy = np.random.normal(0.0, 1.0, (B, H, SEQUENCE_LENGTH, Dk)).astype(
                    DTYPE
                )
                k_npy = np.random.normal(
                    0.0, 1.0, (B, n_kv_heads, SEQUENCE_LENGTH, Dk)
                ).astype(DTYPE)
                v_npy = np.random.normal(
                    0.0, 1.0, (B, n_kv_heads, SEQUENCE_LENGTH, Dk)
                ).astype(DTYPE)

                q_mlx = mx.array(q_npy)
                k_mlx = mx.array(k_npy)
                v_mlx = mx.array(v_npy)

                reference = mlx_primitives_sdpa_with_gqa(q_mlx, k_mlx, v_mlx, scale)
                o_mlx = mx.fast.scaled_dot_product_attention(
                    q_mlx,
                    k_mlx,
                    v_mlx,
                    scale=scale,
                )

                self.assertListEqual(list(reference.shape), list(o_mlx.shape))
                rtol = 1e-3
                atol = 1e-2

                if SEQUENCE_LENGTH > 500:
                    rtol = 1e-2

                if DTYPE == np.half:
                    rtol = 1e-2

                self.assertTrue(mx.allclose(o_mlx, reference, rtol=rtol, atol=atol))


class TestFastSDPA(mlx_tests.MLXTestCase):
    def test_fast_sdpa(self):
        # Not yet supported:
        # * K pre-transposed in kernel, V pre-transposed in kernel
        np.random.seed(0)
        L = 43
        R = 1
        Dk = 128
        scale = float(1.0 / np.sqrt(128.0))
        q_npy = np.random.normal(0.0, 1.0, (1, 32, R, Dk)).astype(np.float32)
        k_npy = np.random.normal(0.0, 1.0, (1, 32, L, Dk)).astype(np.float32)
        v_npy = np.random.normal(0.0, 1.0, (1, 32, L, Dk)).astype(np.float32)

        q_mlx = mx.array(q_npy)
        k_mlx = mx.array(k_npy)
        v_mlx = mx.array(v_npy)

        reference = mlx_primitives_sdpa(q_mlx, k_mlx, v_mlx, scale)

        o_mlx = mx.fast.scaled_dot_product_attention(
            q_mlx, k_mlx, v_mlx, scale=scale, mask=None
        )

        self.assertListEqual(list(reference.shape), list(o_mlx.shape))
        self.assertTrue(mx.allclose(o_mlx, reference, atol=1e-4))

        B = 1
        H = 32
        dtypes = [np.float32]
        if self.is_apple_silicon or mx.cuda.is_available():
            dtypes.append(np.half)

        for SEQUENCE_LENGTH in [1, 7, 9, 32, 63, 67, 129, 400, 2000]:
            for DO_GQA in [0, 1]:
                for DTYPE in dtypes:
                    n_kv_heads = 8 if DO_GQA else 32
                    q_npy = np.random.normal(0.0, 1.0, (B, H, R, Dk)).astype(DTYPE)
                    k_npy = np.random.normal(
                        0.0, 1.0, (B, n_kv_heads, SEQUENCE_LENGTH, Dk)
                    ).astype(DTYPE)
                    v_npy = np.random.normal(
                        0.0, 1.0, (B, n_kv_heads, SEQUENCE_LENGTH, Dk)
                    ).astype(DTYPE)

                    q_mlx = mx.array(q_npy)
                    k_mlx = mx.array(k_npy)
                    v_mlx = mx.array(v_npy)

                    reference = mlx_primitives_sdpa_with_gqa(q_mlx, k_mlx, v_mlx, scale)
                    o_mlx = mx.fast.scaled_dot_product_attention(
                        q_mlx, k_mlx, v_mlx, scale=scale
                    )

                    self.assertListEqual(list(reference.shape), list(o_mlx.shape))
                    rtol = 1e-5
                    atol = 1e-1

                    if SEQUENCE_LENGTH > 500:
                        rtol = 1e-2

                    if DTYPE == np.half:
                        rtol = 1e-2

                    self.assertTrue(mx.allclose(o_mlx, reference, rtol=rtol, atol=atol))
        q = mx.random.normal(shape=(1, 32, 1, Dk))
        k = mx.random.normal(shape=(1, 32, 32, Dk))
        v = mx.random.normal(shape=(1, 32, 128, Dk))

        atol = 1e-6
        y = mlx_primitives_sdpa(q, k, v[:, :, :32], scale)
        y_hat = mx.fast.scaled_dot_product_attention(q, k, v[:, :, :32], scale=scale)
        self.assertTrue(mx.allclose(y, y_hat, atol=atol))

        # Test with per-example mask
        q = mx.random.normal(shape=(2, 8, 4, 32))
        k = mx.random.normal(shape=(2, 2, 8, 32))
        v = mx.random.normal(shape=(2, 2, 8, 32))
        mask = 10 * mx.random.normal(shape=(2, 1, 4, 8))
        y = mlx_primitives_sdpa_with_gqa(q, k, v, scale, mask=mask)
        y_hat = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        self.assertTrue(mx.allclose(y, y_hat, atol=atol))

        # Test with boolean causal mask
        indices = mx.arange(8)
        bool_mask = indices[:, None] >= indices[None]
        additive_mask = (~bool_mask).astype(mx.float32) * mx.finfo(mx.float32).min
        x = mx.random.normal(shape=(1, 2, 8, 32))
        y = mlx_primitives_sdpa_with_gqa(x, x, x, scale, mask=additive_mask)
        y_hat = mx.fast.scaled_dot_product_attention(
            x, x, x, scale=scale, mask=bool_mask
        )
        self.assertTrue(mx.allclose(y, y_hat, atol=atol))

    def test_fast_sdpa_vector_kv_transposed_head_seq(self):
        D = 64
        Nq = 4
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))

        lengths = [43, 4096]
        for L in lengths:
            k = 5e-1 * mx.random.normal(shape=(1, L, Nkv, D))
            v = 5e-1 * mx.random.normal(shape=(1, L, Nkv, D))
            k = k.swapaxes(1, 2)
            v = v.swapaxes(1, 2)
            masks = [
                mx.array(True),
                mx.array([True] * (L - 10) + [False] * 10),
                mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
                mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            ]

            for m in masks:
                ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
                out = mx.fast.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=scale,
                    mask=m,
                )
                self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_fast_sdpa_vector(self):
        D = 64
        L = 43
        Nq = 4
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=mx.full((Nq, 2, L), False),
            )

        masks = [
            None,
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            mx.random.uniform(shape=(Nq, 1, L)),
            mx.random.uniform(shape=(L, 1, Nq)).T,
            mx.log(mx.random.uniform(shape=(Nq, 1, L)) > 0.2),
            mx.log(mx.random.uniform(shape=(L, 1, Nq)).T > 0.2),
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        L = 4096
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        masks = [
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            mx.random.uniform(shape=(Nq, 1, L)),
            mx.random.uniform(shape=(L, 1, Nq)).T,
            mx.log(mx.random.uniform(shape=(Nq, 1, L)) > 0.2),
            mx.log(mx.random.uniform(shape=(L, 1, Nq)).T > 0.2),
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_fully_masked(self):
        Lkv = 8
        masks = [mx.array(False), mx.array(-float("inf"))]
        for mask in masks:
            for D in [4, 128]:
                for Lq in [1, 8]:
                    q = mx.random.normal(shape=(1, 4, Lq, D))
                    k = mx.random.normal(shape=(1, 4, Lkv, D))
                    v = mx.random.normal(shape=(1, 4, Lkv, D))

                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, mask=mask, scale=1
                    )
                    self.assertTrue(mx.all(mx.isnan(out)))

    def test_inf_score(self):
        Lkv = 8
        for D in [4, 128]:
            for Lq in [1, 8]:
                q = mx.ones(shape=(1, 4, Lq, D))
                k = mx.ones(shape=(1, 4, Lkv, D))
                v = mx.random.normal(shape=(1, 4, Lkv, D))
                k[..., 0, :] = -float("inf")
                ref = mlx_primitives_sdpa(q, k, v, scale=1, mask=None)
                out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1)
                self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_fast_sdpa_few_query(self):
        D = 64
        L = 43
        Lq = 8
        Nq = 8
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Lq, Nq, D))
        q = q.swapaxes(1, 2)
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        masks = [
            None,
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        L = 4096
        scale = 1.0
        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, Lq, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))

        masks = [
            None,
            mx.array(True),
            mx.array([True] * (L - 10) + [False] * 10),
            mx.random.uniform(shape=(Nq, 1, L)) > 0.2,
            mx.random.uniform(shape=(L, 1, Nq)).T > 0.2,
            "causal",
        ]
        for m in masks:
            ref = mlx_primitives_sdpa(q, k, v, scale, mask=m)
            out = mx.fast.scaled_dot_product_attention(
                q,
                k,
                v,
                scale=scale,
                mask=m,
            )
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    @unittest.skip("Different head and value dims is not enabled")
    def test_fast_sdpa_vector_value_dims(self):
        D = 192
        V = 128
        Nq = 4
        Nkv = 1
        scale = 1.0
        mx.random.seed(0)

        for L in [43, 128, 237, 8192]:
            q = 5e-1 * mx.random.normal(shape=(1, Nq, 1, D))
            k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
            v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, V))
            ref = mlx_primitives_sdpa(q, k, v, scale)
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_vector_batched(self):
        D = 64
        q = mx.random.normal(shape=(2, 1, 3, D))
        k = mx.random.normal(shape=(2, 1, 3, D))
        v = mx.random.normal(shape=(2, 1, 3, D))

        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 4, 3, D))
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 3, 4, D)).swapaxes(1, 2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        k = mx.random.normal(shape=(2, 3, 1, D)).swapaxes(1, 2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 4, 3, D))
        k = mx.random.normal(shape=(2, 3, 2, D)).swapaxes(1, 2)
        v = mx.random.normal(shape=(2, 2, 3, D))
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=None, scale=1.0)
        ref = mlx_ref_attn(q, k, v)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

        q = mx.random.normal(shape=(2, 4, 3, D))
        k = mx.random.normal(shape=(2, 1, 3, D))
        v = mx.random.normal(shape=(2, 1, 3, D))
        mask = 10 * mx.random.normal(shape=(1, 2, 3, 3)).swapaxes(0, 1)
        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0)
        ref = mlx_ref_attn(q, k, v, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))


class TestSDPA(mlx_tests.MLXTestCase):
    @property
    def dtypes(self):
        return ["float32", "float16"] if mx.metal.is_available() else ["float32"]

    def test_sdpa(self):
        if not mx.metal.is_available():
            return

        # fmt: off
        shapes_64 = (
            # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
            (  1,   128,   128,       64,   32,    32),
            (  1,    64,   128,       64,   32,    32),
            (  1,    65,   128,       64,   32,     8),
            (  1,    64,   127,       64,   32,     8),
            (  1,    65,   127,       64,   32,     8),
            (  1,   127,    65,       64,   32,     8),
        )

        shapes_128 = (
            # (  B,   qsl,   ksl, head_dim, n_qh, n_kvh)
            (  1,   128,   128,      128,   32,     8),
            (  1,    64,   128,      128,   32,     8),
            (  1,    65,   127,      128,   32,     8),
            (  1,   127,    65,      128,   32,     8),
        )
        # fmt: on

        shapes = shapes_64 + shapes_128
        masks = [None, "additive", "bool", "causal"]
        transposes = (False, True)

        for dtype in self.dtypes:
            for t in transposes:
                for mask_str in masks:
                    for B, qL, kL, D, qH, kH in shapes:
                        with self.subTest(
                            B=B,
                            qsl=qL,
                            ksl=kL,
                            head_dim=D,
                            n_q_heads=qH,
                            n_kv_heads=kH,
                            mask=mask_str,
                            transpose=t,
                            dtype=dtype,
                        ):

                            np.random.seed(0)
                            q_mx, k_mx, v_mx, scale, mask = prepare_inputs(
                                B, qL, kL, D, qH, kH, mask_str, t, dtype
                            )

                            out_ref = do_attention(
                                mlx_ref_attn, q_mx, k_mx, v_mx, scale, mask, t
                            )

                            out_fst = do_attention(
                                mx.fast.scaled_dot_product_attention,
                                q_mx,
                                k_mx,
                                v_mx,
                                scale,
                                mask,
                                t,
                            )

                            atol = 2e-5 if dtype == "float32" else 3e-4

                            self.assertListEqual(
                                list(out_ref.shape), list(out_fst.shape)
                            )

                            diff = mx.abs(out_fst - out_ref) - atol * mx.abs(out_ref)
                            self.assertLessEqual(mx.max(diff).item(), atol)

    def test_sdpa_broadcast_mask(self):
        mask = mx.array(True)
        D = 64
        Nq = 4
        Nkv = 1
        scale = 1.0
        L = 256

        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, L, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        ref = mlx_primitives_sdpa(q, k, v, scale, mask=mask)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_noncontiguous_inputs(self):
        mask = mx.ones(shape=(4, 1, 7, 7), dtype=mx.bool_)
        mx.random.seed(0)
        q = mx.random.normal(shape=(4, 7, 32, 64)).swapaxes(1, 2)

        k = mx.random.normal(shape=(4, 7, 8, 64)).swapaxes(1, 2)
        v = mx.random.normal(shape=(4, 7, 8, 64)).swapaxes(1, 2)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        ref = mlx_ref_attn(q, k, v, scale=1.0, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_promote_mask(self):
        mask = mx.array(2.0, mx.bfloat16)
        D = 64
        Nq = 4
        Nkv = 1
        scale = 1.0
        L = 256

        mx.random.seed(0)
        q = 5e-1 * mx.random.normal(shape=(1, Nq, L, D))
        k = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        v = 5e-1 * mx.random.normal(shape=(1, Nkv, L, D))
        ref = mlx_primitives_sdpa(q, k, v, scale, mask=mask)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        self.assertTrue(mx.allclose(ref, out, atol=1e-4, rtol=1e-4))

    def test_sdpa_nan_bug(self):
        N = 128
        q_shape = (1, 1, N, 128)
        kv_shape = (1, 1, N, 128)
        q = mx.random.uniform(shape=q_shape)
        k = mx.random.uniform(shape=kv_shape)
        v = mx.random.uniform(shape=kv_shape)

        # Make boolean window causal mask
        linds = rinds = mx.arange(N)
        linds = linds[:, None]
        rinds = rinds[None]
        mask = linds >= rinds
        mask = mask & (linds <= rinds + 111)

        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0)
        expected = mlx_ref_attn(q, k, v, mask=mask, scale=1.0)
        self.assertFalse(mx.isnan(out).any().item())
        self.assertLessEqual(mx.abs(out - expected).max().item(), 1e-4)

        # And an additive one
        mask = mx.log(mask)

        out = mx.fast.scaled_dot_product_attention(q, k, v, mask=mask, scale=1.0)
        expected = mlx_ref_attn(q, k, v, mask=mask, scale=1.0)
        self.assertFalse(mx.isnan(out).any().item())
        self.assertLessEqual(mx.abs(out - expected).max().item(), 1e-4)

    def test_sdpa_attention_sinks(self):
        B = 2
        N_q = N_kv = 8
        T_q = T_kv = 128
        D = 64

        q = mx.random.normal(shape=(B, N_q, T_q, D))
        k = mx.random.normal(shape=(B, N_kv, T_kv, D))
        v = mx.random.normal(shape=(B, N_kv, T_kv, D))
        scale = D**-0.5

        # sinks should promote to correct type
        sinks = mx.random.normal(shape=(N_q,))
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(
                q.astype(mx.float16),
                k.astype(mx.float16),
                v.astype(mx.float16),
                scale=scale,
                sinks=sinks,
            )

        # Wrong shapes
        sinks = mx.random.normal(shape=(N_q + 1,))
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, sinks=sinks)

        sinks = mx.random.normal(shape=())
        with self.assertRaises(ValueError):
            mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, sinks=sinks)

        for T_kv in [128, 4096]:
            for T_q in [1, 128]:
                for N_kv in [2, 8]:
                    q = mx.random.normal(shape=(B, N_q, T_q, D))
                    k = mx.random.normal(shape=(B, N_kv, T_kv, D))
                    v = mx.random.normal(shape=(B, N_kv, T_kv, D))
                    sinks = 10 * mx.random.normal(shape=(N_q,))

                    expected = mlx_ref_attn(q, k, v, scale, sinks=sinks)
                    out = mx.fast.scaled_dot_product_attention(
                        q, k, v, scale=scale, sinks=sinks
                    )
                    self.assertTrue(mx.allclose(out, expected, atol=1e-5))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner(failfast=True)
