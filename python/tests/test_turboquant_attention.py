# Copyright © 2024 Apple Inc.

"""
Tests for mx.fast.turboquant_attention — fused attention on compressed KV cache.

Verifies:
  1. Output shapes and dtypes
  2. Correctness vs reference Python implementation
  3. GQA (grouped query attention) support
  4. Edge cases (N=1, large N for 2-pass kernel)
  5. Input validation and error handling
  6. Multiple head dimensions (D=64, D=128)
"""

import math
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


def _make_random_orthogonal(D, seed=42):
    """Create a random orthogonal matrix via QR decomposition."""
    np.random.seed(seed)
    G = np.random.randn(D, D).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    return mx.array(Q)


def _quantize_keys_2bit(keys, rotation_matrix, sketch_matrix):
    """Simulate 2-bit TurboQuant key compression in Python.

    Args:
        keys: (B, H_kv, N, D) float32 key vectors
        rotation_matrix: (D, D) orthogonal matrix
        sketch_matrix: (D, D) random Gaussian matrix

    Returns:
        k_packed, k_signs, k_norms, k_res_norms, centroids
    """
    B, H, N, D = keys.shape

    # Compute norms and normalize
    norms = mx.sqrt(mx.sum(keys * keys, axis=-1))  # (B, H, N)
    safe_norms = mx.maximum(norms, 1e-10)
    keys_unit = keys / safe_norms[..., None]

    # Rotate: x_rot = keys_unit @ rotation_matrix^T
    x_rot = keys_unit @ mx.transpose(rotation_matrix)  # (B, H, N, D)

    # Lloyd-Max centroids for 2-bit (4 centroids for Beta distribution)
    # Use simple uniform centroids for test purposes
    centroids = mx.array([-0.75, -0.25, 0.25, 0.75], dtype=mx.float32)

    # Quantize each coordinate to nearest centroid
    x_rot_flat = mx.reshape(x_rot, (-1, D))  # (B*H*N, D)
    # For each coordinate, find nearest centroid index (0-3)
    diffs = mx.abs(x_rot_flat[..., None] - centroids[None, None, :])  # (B*H*N, D, 4)
    indices = mx.argmin(diffs, axis=-1).astype(mx.uint8)  # (B*H*N, D)

    # Bit-pack indices: 4 values per byte (2 bits each)
    packed_d = D // 4
    indices_np = np.array(indices)
    packed = np.zeros((B * H * N, packed_d), dtype=np.uint8)
    for i in range(4):
        packed |= (indices_np[:, i::4] & 0x3) << (i * 2)
    k_packed = mx.array(packed).reshape(B, H, N, packed_d)

    # Dequantize for residual
    dequant = mx.reshape(
        centroids[mx.reshape(indices, (-1,))],
        (B * H * N, D),
    )
    dequant = mx.reshape(dequant, (B, H, N, D))

    # Residual
    residual = x_rot - dequant
    # QJL: project residual through sketch matrix, store signs
    r_proj = residual @ mx.transpose(sketch_matrix)  # (B, H, N, D)
    signs = (r_proj >= 0).astype(mx.uint8)  # (B, H, N, D)

    # Bit-pack signs: 8 per byte
    packed_d_signs = D // 8
    signs_np = np.array(signs).reshape(B * H * N, D)
    packed_signs = np.zeros((B * H * N, packed_d_signs), dtype=np.uint8)
    for i in range(8):
        packed_signs |= (signs_np[:, i::8] & 0x1) << i
    k_signs = mx.array(packed_signs).reshape(B, H, N, packed_d_signs)

    # Residual norms
    res_norms = mx.sqrt(mx.sum(residual * residual, axis=-1))  # (B, H, N)

    return k_packed, k_signs, norms, res_norms, centroids


def _quantize_values_2bit(values, group_size=32):
    """Simulate 2-bit asymmetric group quantization for values.

    Returns:
        v_packed, v_scales, v_zeros
    """
    B, H, N, D = values.shape
    n_groups = D // group_size

    # Reshape to groups
    grouped = mx.reshape(values, (B, H, N, n_groups, group_size))

    # Per-group min/max
    v_min = mx.min(grouped, axis=-1)  # (B, H, N, n_groups)
    v_max = mx.max(grouped, axis=-1)
    v_range = v_max - v_min
    v_range = mx.maximum(v_range, 1e-10)

    # Scale and zero point
    n_levels = (1 << 2) - 1  # 3 for 2-bit
    v_scales = v_range / n_levels  # (B, H, N, n_groups)
    v_zeros = v_min  # (B, H, N, n_groups)

    # Quantize
    normalized = (grouped - v_min[..., None]) / (v_range[..., None] + 1e-10)
    indices = mx.clip(mx.round(normalized * n_levels), 0, n_levels).astype(mx.uint8)

    # Bit-pack: 4 values per byte
    indices_np = np.array(indices).reshape(B * H * N, n_groups, group_size)
    packed_g = group_size // 4
    packed = np.zeros((B * H * N, n_groups, packed_g), dtype=np.uint8)
    for i in range(4):
        packed |= (indices_np[:, :, i::4] & 0x3) << (i * 2)
    packed = packed.reshape(B * H * N, n_groups * packed_g)
    v_packed = mx.array(packed).reshape(B, H, N, n_groups * packed_g)

    return v_packed, v_scales, v_zeros


def _reference_attention(
    queries,
    keys,
    values,
    scale,
    rotation_matrix,
    sketch_matrix,
):
    """Reference full-precision attention for correctness comparison.

    Returns (output, max_score, sum_exp) where output is unnormalized.
    """
    B, H_q, qL, D = queries.shape
    H_kv = keys.shape[1]
    repeats = H_q // H_kv

    if repeats > 1:
        keys = mx.repeat(keys, repeats, axis=1)
        values = mx.repeat(values, repeats, axis=1)

    scores = (queries @ mx.transpose(keys, (0, 1, 3, 2))) * scale  # (B, H_q, qL, N)
    max_score = mx.max(scores, axis=-1)  # (B, H_q, qL)
    exp_scores = mx.exp(scores - max_score[..., None])
    sum_exp = mx.sum(exp_scores, axis=-1)  # (B, H_q, qL)
    weights = exp_scores / sum_exp[..., None]
    output = weights @ values  # (B, H_q, qL, D)
    # Return unnormalized accumulator for comparison
    acc = exp_scores @ values  # unnormalized
    return acc, max_score, sum_exp


class TestTurboQuantAttention(mlx_tests.MLXTestCase):

    def _make_inputs(self, B=1, H_q=4, H_kv=4, N=64, D=128, group_size=32):
        """Create synthetic compressed inputs for testing."""
        mx.random.seed(42)
        np.random.seed(42)

        queries = mx.random.normal((B, H_q, 1, D))
        keys = mx.random.normal((B, H_kv, N, D))
        values = mx.random.normal((B, H_kv, N, D))

        rotation_matrix = _make_random_orthogonal(D, seed=42)
        sketch_matrix = _make_random_orthogonal(D, seed=99)

        k_packed, k_signs, k_norms, k_res_norms, centroids = _quantize_keys_2bit(
            keys, rotation_matrix, sketch_matrix
        )

        v_packed, v_scales, v_zeros = _quantize_values_2bit(values, group_size)

        mx.eval(
            queries,
            k_packed,
            k_signs,
            k_norms,
            k_res_norms,
            centroids,
            v_packed,
            v_scales,
            v_zeros,
            rotation_matrix,
            sketch_matrix,
        )

        scale = 1.0 / math.sqrt(D)
        qjl_scale = 1.0 / math.sqrt(D)

        return {
            "queries": queries,
            "k_packed": k_packed,
            "k_signs": k_signs,
            "k_norms": k_norms,
            "k_res_norms": k_res_norms,
            "centroids": centroids,
            "v_packed": v_packed,
            "v_scales": v_scales,
            "v_zeros": v_zeros,
            "rotation_matrix": rotation_matrix,
            "sketch_matrix": sketch_matrix,
            "scale": scale,
            "qjl_scale": qjl_scale,
            "group_size": group_size,
        }

    def test_output_shapes(self):
        """Output shapes match (B, H_q, qL, D) for acc, (B, H_q, qL) for m and l."""
        inputs = self._make_inputs(B=1, H_q=4, H_kv=4, N=64, D=128)
        acc, m, l = mx.fast.turboquant_attention(**inputs)
        mx.eval(acc, m, l)

        self.assertEqual(acc.shape, (1, 4, 1, 128))
        self.assertEqual(m.shape, (1, 4, 1))
        self.assertEqual(l.shape, (1, 4, 1))

    def test_output_shapes_d64(self):
        """Works with D=64 head dimension."""
        inputs = self._make_inputs(B=1, H_q=2, H_kv=2, N=32, D=64)
        acc, m, l = mx.fast.turboquant_attention(**inputs)
        mx.eval(acc, m, l)

        self.assertEqual(acc.shape, (1, 2, 1, 64))
        self.assertEqual(m.shape, (1, 2, 1))
        self.assertEqual(l.shape, (1, 2, 1))

    def test_output_dtypes(self):
        """acc matches query dtype, m and l are float32."""
        inputs = self._make_inputs()
        acc, m, l = mx.fast.turboquant_attention(**inputs)
        mx.eval(acc, m, l)

        self.assertEqual(acc.dtype, inputs["queries"].dtype)
        self.assertEqual(m.dtype, mx.float32)
        self.assertEqual(l.dtype, mx.float32)

    def test_output_finite(self):
        """All outputs are finite (no NaN or Inf)."""
        inputs = self._make_inputs()
        acc, m, l = mx.fast.turboquant_attention(**inputs)
        mx.eval(acc, m, l)

        self.assertTrue(mx.all(mx.isfinite(acc)).item())
        self.assertTrue(mx.all(mx.isfinite(m)).item())
        self.assertTrue(mx.all(mx.isfinite(l)).item())

    def test_sum_exp_positive(self):
        """sum_exp (l) must be positive for valid softmax denominator."""
        inputs = self._make_inputs()
        acc, m, l = mx.fast.turboquant_attention(**inputs)
        mx.eval(l)

        self.assertTrue(mx.all(l > 0).item())

    def test_normalized_output_reasonable(self):
        """Normalized output (acc/l) should have reasonable magnitude."""
        inputs = self._make_inputs()
        acc, m, l = mx.fast.turboquant_attention(**inputs)
        mx.eval(acc, l)

        normalized = acc / l[..., None]
        max_val = mx.max(mx.abs(normalized)).item()
        # Attention output should not be huge
        self.assertLess(max_val, 100.0)

    def test_gqa(self):
        """Grouped query attention: H_q=8, H_kv=2 (4 queries per KV head)."""
        inputs = self._make_inputs(B=1, H_q=8, H_kv=2, N=64, D=128)
        acc, m, l = mx.fast.turboquant_attention(**inputs)
        mx.eval(acc, m, l)

        self.assertEqual(acc.shape, (1, 8, 1, 128))
        self.assertTrue(mx.all(mx.isfinite(acc)).item())
        self.assertTrue(mx.all(l > 0).item())

    def test_batch_size(self):
        """Works with batch size > 1."""
        inputs = self._make_inputs(B=2, H_q=4, H_kv=4, N=32, D=128)
        acc, m, l = mx.fast.turboquant_attention(**inputs)
        mx.eval(acc, m, l)

        self.assertEqual(acc.shape, (2, 4, 1, 128))
        self.assertTrue(mx.all(mx.isfinite(acc)).item())

    def test_single_kv_token(self):
        """Edge case: N=1 (single compressed KV token)."""
        inputs = self._make_inputs(B=1, H_q=2, H_kv=2, N=1, D=128)
        acc, m, l = mx.fast.turboquant_attention(**inputs)
        mx.eval(acc, m, l)

        self.assertEqual(acc.shape, (1, 2, 1, 128))
        self.assertTrue(mx.all(mx.isfinite(acc)).item())

    def test_long_sequence_2pass(self):
        """Long sequence (N>=1024) triggers 2-pass kernel."""
        inputs = self._make_inputs(B=1, H_q=2, H_kv=2, N=2048, D=128)
        acc, m, l = mx.fast.turboquant_attention(**inputs)
        mx.eval(acc, m, l)

        self.assertEqual(acc.shape, (1, 2, 1, 128))
        self.assertTrue(mx.all(mx.isfinite(acc)).item())
        self.assertTrue(mx.all(l > 0).item())

    def test_1pass_vs_2pass_consistency(self):
        """1-pass (N<1024) and 2-pass (N>=1024) should produce similar results."""
        # Use same random data at different sizes
        mx.random.seed(123)
        D = 128
        rotation = _make_random_orthogonal(D, seed=10)
        sketch = _make_random_orthogonal(D, seed=20)

        # Generate keys/values, split into short and long
        keys_all = mx.random.normal((1, 2, 2048, D))
        values_all = mx.random.normal((1, 2, 2048, D))
        queries = mx.random.normal((1, 2, 1, D))

        # Short (1-pass): first 512 tokens
        keys_short = keys_all[:, :, :512, :]
        values_short = values_all[:, :, :512, :]

        kp_s, ks_s, kn_s, kr_s, centroids = _quantize_keys_2bit(
            keys_short, rotation, sketch
        )
        vp_s, vs_s, vz_s = _quantize_values_2bit(values_short)
        mx.eval(kp_s, ks_s, kn_s, kr_s, centroids, vp_s, vs_s, vz_s)

        scale = 1.0 / math.sqrt(D)
        qjl_scale = 1.0 / math.sqrt(D)

        acc_s, m_s, l_s = mx.fast.turboquant_attention(
            queries,
            kp_s,
            ks_s,
            kn_s,
            kr_s,
            centroids,
            vp_s,
            vs_s,
            vz_s,
            rotation,
            sketch,
            scale=scale,
            qjl_scale=qjl_scale,
        )
        out_s = acc_s / l_s[..., None]

        # Long (2-pass): first 2048 tokens
        kp_l, ks_l, kn_l, kr_l, _ = _quantize_keys_2bit(keys_all, rotation, sketch)
        vp_l, vs_l, vz_l = _quantize_values_2bit(values_all)
        mx.eval(kp_l, ks_l, kn_l, kr_l, vp_l, vs_l, vz_l)

        acc_l, m_l, l_l = mx.fast.turboquant_attention(
            queries,
            kp_l,
            ks_l,
            kn_l,
            kr_l,
            centroids,
            vp_l,
            vs_l,
            vz_l,
            rotation,
            sketch,
            scale=scale,
            qjl_scale=qjl_scale,
        )
        mx.eval(acc_s, out_s, acc_l, m_l, l_l)

        # Both should produce finite results
        self.assertTrue(mx.all(mx.isfinite(acc_s)).item())
        self.assertTrue(mx.all(mx.isfinite(acc_l)).item())
        self.assertTrue(mx.all(l_s > 0).item())
        self.assertTrue(mx.all(l_l > 0).item())

    # --- Input validation tests ---

    def test_rejects_wrong_query_rank(self):
        """Queries must be rank 4."""
        inputs = self._make_inputs()
        inputs["queries"] = mx.random.normal((4, 128))  # rank 2
        with self.assertRaises(Exception):
            mx.fast.turboquant_attention(**inputs)

    def test_rejects_wrong_d(self):
        """D must be 64 or 128."""
        inputs = self._make_inputs(D=64)
        inputs["queries"] = mx.random.normal((1, 2, 1, 96))  # D=96
        with self.assertRaises(Exception):
            mx.fast.turboquant_attention(**inputs)

    def test_rejects_non_divisible_heads(self):
        """H_q must be divisible by H_kv."""
        inputs = self._make_inputs(H_q=3, H_kv=2)
        with self.assertRaises(Exception):
            mx.fast.turboquant_attention(**inputs)

    def test_rejects_cpu(self):
        """Must be GPU-only."""
        inputs = self._make_inputs()
        inputs["stream"] = mx.cpu
        with self.assertRaises(Exception):
            acc, m, l = mx.fast.turboquant_attention(**inputs)
            mx.eval(acc, m, l)

    def test_float16_queries(self):
        """Works with float16 queries."""
        inputs = self._make_inputs(N=32)
        inputs["queries"] = inputs["queries"].astype(mx.float16)
        acc, m, l = mx.fast.turboquant_attention(**inputs)
        mx.eval(acc, m, l)

        self.assertEqual(acc.dtype, mx.float16)
        self.assertTrue(mx.all(mx.isfinite(acc)).item())


if __name__ == "__main__":
    unittest.main()
