import math
import unittest
from itertools import product

import mlx.core as mx
import mlx_tests
import numpy as np


# SDPA for MHA (n_heads == n_kv_heads)
def mlx_primitives_sdpa(q, k, v, scale, mask=None):
    p = (q * scale) @ k.transpose(0, 1, 3, 2)
    if mask is not None:
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
        q = mx.random.normal(shape=(1, 32, R, Dk))
        k = mx.random.normal(shape=(1, 32, L, Dk))
        v = mx.random.normal(shape=(1, 32, L, Dk))

        reference = mlx_primitives_sdpa(q, k, v, scale)

        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)

        self.assertListEqual(list(reference.shape), list(o.shape))
        self.assertTrue(mx.allclose(o, reference, atol=1e-4))

        B = 1
        H = 32

        dtypes = [mx.float32]
        if self.is_apple_silcon:
            dtypes.append(mx.float16)
        tests = product(
            [1, 7, 9, 32, 63, 67, 129, 2000],  # sequence length
            [False, True],  # gqa
            dtypes,
            [4, 8],  # bits
        )
        for sequence_length, do_gqa, dtype, bits in tests:
            with self.subTest(
                sequence_length=sequence_length, gqa=do_gqa, dtype=dtype, bits=bits
            ):
                n_kv_heads = 8 if do_gqa else 32
                q = mx.random.normal(shape=(B, H, R, Dk), dtype=dtype)
                k = mx.random.normal(
                    shape=(B, n_kv_heads, sequence_length, Dk), dtype=dtype
                )
                v = mx.random.normal(
                    shape=(B, n_kv_heads, sequence_length, Dk), dtype=dtype
                )

                k_q = mx.quantize(k, bits=bits)
                v_q = mx.quantize(v, bits=bits)
                k_d = mx.dequantize(*k_q, bits=bits)
                v_d = mx.dequantize(*v_q, bits=bits)

                reference = mlx_primitives_sdpa_with_gqa(q, k_d, v_d, scale)
                o = mx.fast.scaled_dot_product_attention(q, k_d, v_d, scale=scale)
                o_q = mx.fast.quantized_scaled_dot_product_attention(
                    q, *k_q, *v_q, scale=scale, bits=bits
                )

                self.assertListEqual(list(reference.shape), list(o.shape))
                rtol = 1e-5
                atol = 1e-1

                if sequence_length > 500:
                    rtol = 1e-2

                if dtype == mx.float16:
                    rtol = 1e-2

                self.assertTrue(mx.allclose(o_q, reference, rtol=rtol, atol=atol))
                self.assertTrue(mx.allclose(o, reference, rtol=rtol, atol=atol))

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


if __name__ == "__main__":
    unittest.main(failfast=True)
