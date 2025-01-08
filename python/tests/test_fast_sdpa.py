import math
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


# SDPA for MHA (n_heads == n_kv_heads)
def mlx_primitives_sdpa(q, k, v, scale, mask=None):
    p = (q * scale) @ k.transpose(0, 1, 3, 2)
    if mask is not None:
        if mask.dtype == mx.bool_:
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

        if self.is_apple_silicon:
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
                    q_mlx, k_mlx, v_mlx, scale=scale, memory_efficient_threshold=2
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
        if self.is_apple_silicon:
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


if __name__ == "__main__":
    unittest.main(failfast=True)
