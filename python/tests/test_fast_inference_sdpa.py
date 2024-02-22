# Created by Brian Keene, 2024

import math
import unittest
from itertools import permutations

import mlx.core as mx
import numpy as np

import mlx_tests


def relerr(truth, other):
    a = truth.flatten()
    b = other.flatten()
    return np.linalg.norm(a - b) / np.linalg.norm(a)


# SDPA for MHA (n_heads == n_kv_heads)
def mlx_primitives_sdpa(q, k, v, scale):
    p = (q * scale) @ k.transpose(0, 1, 3, 2)
    scores = mx.softmax(p.astype(mx.float32), axis=-1).astype(p.dtype)
    return scores @ v


# SDPA for GQA (n_heads > n_kv_heads, n_kv_heads > 1, n_heads % n_kv_heads == 0)
def mlx_primitives_sdpa_with_gqa(q, k, v, scale):

    n_repeats = q.shape[1] // k.shape[1]

    # borrowing kv cache tiling from mlx-examples/llms/mistral/mistral.py
    n_heads = q.shape[1]
    B = q.shape[0]
    L = k.shape[2]

    def repeat(a):
        a = mx.concatenate([mx.expand_dims(a, 2)] * n_repeats, axis=2)
        return a.reshape([B, n_heads, L, -1])

    k, v = map(repeat, (k, v))

    return mlx_primitives_sdpa(q, k, v, scale)


class TestFastInferenceSDPA(mlx_tests.MLXTestCase):
    @property
    def dtypes(self):
        return ["float32", "float16"] if mx.metal.is_available() else ["float32"]

    def test_fast_inference_sdpa(self):

        # Not yet supported:
        # * K pre-transposed in kernel, V pre-transposed in kernel
        # * Custom metal shader is exercised only for query sequence length == 1
        if not mx.metal.is_available():
            return
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

        o_mlx = mx.fast_inference_sdpa(q_mlx, k_mlx, v_mlx, scale, None)

        mx.eval()

        self.assertListEqual(list(reference.shape), list(o_mlx.shape))
        self.assertTrue(np.allclose(o_mlx, reference, atol=1e-4))

        B = 1
        H = 32
        for SEQUENCE_LENGTH in [1, 7, 9, 32, 63, 67, 129, 400, 2000]:
            for DO_GQA in [0, 1]:
                for DTYPE in [np.float32, np.half]:
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
                    o_mlx = mx.fast_inference_sdpa(q_mlx, k_mlx, v_mlx, scale, None)

                    self.assertListEqual(list(reference.shape), list(o_mlx.shape))
                    rtol = 1e-5
                    atol = 1e-1

                    if SEQUENCE_LENGTH > 500:
                        rtol = 1e-2

                    if DTYPE == np.half:
                        rtol = 1e-2

                    self.assertTrue(np.allclose(o_mlx, reference, rtol=rtol, atol=atol))


if __name__ == "__main__":
    unittest.main(failfast=True)
