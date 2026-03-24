# Copyright © 2023 Apple Inc.

"""
Tests for fused SDPA attention output correctness, covering the configurations
that will be exercised by the chunked SDPA + logsumexp path.

These tests run against the CURRENT kernel (no logsumexp output yet) to
establish a correctness baseline.  Every test must PASS before any kernel
changes are made.

Conventions (matching test_fast_sdpa.py):
  - Shapes are (B, n_heads, seq_len, head_dim)  [i.e. heads-first]
  - Reference is computed in float32 with manual matmul + softmax
  - GQA handled by repeating K/V heads in the reference
  - Causal mask: position i attends to j iff i + (kL - qL) >= j
"""

import math
import unittest
from itertools import product

import mlx.core as mx
import mlx_tests


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def ref_attention(q, k, v, scale, causal=False):
    """Float32 reference attention with optional causal mask.

    Supports GQA: if n_kv_heads < n_q_heads the KV tensors are tiled.

    Args:
        q: (B, n_q_heads,  qL, D)
        k: (B, n_kv_heads, kL, D)
        v: (B, n_kv_heads, kL, D)
        scale: scalar
        causal: bool

    Returns:
        out:      (B, n_q_heads, qL, D)  float32
        logsumexp:(B, n_q_heads, qL)     float32  — log(sum(exp(scores)))
                                                    used for chunked-SDPA merge
    """
    # Up-cast to float32 for stable reference numerics
    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)

    B, n_q_heads, qL, D = q.shape
    n_kv_heads = k.shape[1]
    kL = k.shape[2]

    # GQA: tile K and V so shapes match Q
    if n_kv_heads != n_q_heads:
        assert n_q_heads % n_kv_heads == 0
        n_rep = n_q_heads // n_kv_heads
        # (B, n_kv_heads, kL, D) -> (B, n_q_heads, kL, D)
        k = mx.repeat(k, n_rep, axis=1)
        v = mx.repeat(v, n_rep, axis=1)

    # Scaled dot-product scores: (B, n_q_heads, qL, kL)
    scores = (q * scale) @ mx.swapaxes(k, -1, -2)

    if causal:
        # Query position i (0-indexed) can attend to key position j iff
        #   i + (kL - qL) >= j
        offset = kL - qL
        q_idx = mx.arange(qL)[:, None] + offset  # (qL, 1)
        k_idx = mx.arange(kL)[None, :]            # (1, kL)
        mask = q_idx >= k_idx                      # (qL, kL) bool
        scores = mx.where(mask, scores, mx.array(-1e9, mx.float32))

    # logsumexp for numerical stability
    scores_max = mx.max(scores, axis=-1, keepdims=True)           # (B, H, qL, 1)
    exp_scores = mx.exp(scores - scores_max)                       # (B, H, qL, kL)
    sum_exp = mx.sum(exp_scores, axis=-1, keepdims=True)           # (B, H, qL, 1)
    attn_weights = exp_scores / sum_exp                            # (B, H, qL, kL)

    out = attn_weights @ v                                         # (B, H, qL, D)

    # logsumexp = max + log(sum_exp)  — shape (B, H, qL)
    logsumexp = (scores_max + mx.log(sum_exp))[..., 0]

    return out, logsumexp


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestSDPALogsumexpBaseline(mlx_tests.MLXTestCase):
    """Verify mx.fast.scaled_dot_product_attention output against float32 reference.

    These tests establish the correctness baseline that the chunked-SDPA path
    must reproduce after logsumexp output support is added to the kernel.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check(self, q, k, v, scale, causal=False, atol=1e-2):
        """Run fused SDPA and compare to float32 reference."""
        mask = "causal" if causal else None

        ref_out, _ = ref_attention(q, k, v, scale, causal=causal)
        fused_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask
        )
        mx.eval(ref_out, fused_out)

        # Cast reference back to the compute dtype for a fair comparison
        ref_out = ref_out.astype(q.dtype)

        max_diff = mx.max(mx.abs(fused_out - ref_out)).item()
        self.assertLessEqual(
            max_diff,
            atol,
            msg=(
                f"max |fused - ref| = {max_diff:.2e} > atol={atol:.2e}  "
                f"shape q={q.shape} k={k.shape} causal={causal} dtype={q.dtype}"
            ),
        )

    def _make_qkv(self, B, qL, kL, n_q, n_kv, D, dtype, seed=42):
        mx.random.seed(seed)
        scale = 1.0 / math.sqrt(D)
        q = mx.random.uniform(-0.5, 0.5, (B, n_q, qL, D)).astype(dtype)
        k = mx.random.uniform(-0.5, 0.5, (B, n_kv, kL, D)).astype(dtype)
        v = mx.random.uniform(-0.5, 0.5, (B, n_kv, kL, D)).astype(dtype)
        return q, k, v, scale

    def _atol_for(self, dtype):
        if dtype == mx.float32:
            return 1e-4
        return 1e-2  # float16 / bfloat16

    # ------------------------------------------------------------------
    # dtype sweep: float16, bfloat16, float32 x head_dim sweep
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_dtype_and_headdim(self):
        """Standard MHA across all required dtypes and head dimensions.

        Note: float32 + D=256 exceeds the Metal threadgroup memory limit
        (53760 bytes > 32768 bytes) on current hardware — that combination is
        the motivation for chunked SDPA and is skipped here.  All other
        (dtype, D) combinations must pass.
        """
        B, qL, kL = 1, 64, 64
        n_heads = 8

        configs = list(product(
            [mx.float16, mx.bfloat16, mx.float32],
            [64, 80, 128, 256],
        ))

        for dtype, D in configs:
            # float32 + D=256 exceeds Metal threadgroup memory on current kernel
            if dtype == mx.float32 and D == 256:
                continue
            with self.subTest(dtype=dtype, D=D):
                q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D, dtype)
                self._check(q, k, v, scale, causal=False, atol=self._atol_for(dtype))
                self._check(q, k, v, scale, causal=True,  atol=self._atol_for(dtype))

    # ------------------------------------------------------------------
    # Causal attention
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_causal_square(self):
        """Causal self-attention with square qL == kL."""
        B, n_heads, D = 1, 8, 128
        for qL in [32, 64, 128, 256]:
            with self.subTest(qL=qL):
                q, k, v, scale = self._make_qkv(B, qL, qL, n_heads, n_heads, D,
                                                 mx.float16)
                self._check(q, k, v, scale, causal=True)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_causal_decode(self):
        """Causal decode: qL=1 attending to growing KV cache."""
        B, n_heads, D = 1, 8, 128
        for kL in [64, 128, 256, 512]:
            with self.subTest(kL=kL):
                q, k, v, scale = self._make_qkv(B, 1, kL, n_heads, n_heads, D,
                                                 mx.float16)
                self._check(q, k, v, scale, causal=True)

    # ------------------------------------------------------------------
    # Cross-attention (qL != kL)
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_cross_attention(self):
        """Cross-attention where query and key/value lengths differ."""
        B, n_heads, D = 1, 8, 128
        cross_shapes = [
            (16, 128),
            (32, 256),
            (64, 512),
            (128, 64),   # qL > kL
        ]
        for qL, kL in cross_shapes:
            for causal in (False, True):
                with self.subTest(qL=qL, kL=kL, causal=causal):
                    q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads,
                                                     D, mx.float16)
                    self._check(q, k, v, scale, causal=causal)

    # ------------------------------------------------------------------
    # GQA (n_kv_heads != n_q_heads)
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_gqa(self):
        """Grouped-query attention: multiple Q heads share each KV head."""
        B, qL, kL, D = 1, 64, 64, 128
        gqa_configs = [
            (32, 8),   # 4:1 ratio  — typical 122B
            (8,  1),   # 8:1 ratio  — extreme GQA / MQA
            (16, 4),   # 4:1 ratio
            (8,  2),   # 4:1 ratio, 2 KV heads
        ]
        for n_q, n_kv in gqa_configs:
            for causal in (False, True):
                with self.subTest(n_q=n_q, n_kv=n_kv, causal=causal):
                    q, k, v, scale = self._make_qkv(B, qL, kL, n_q, n_kv, D,
                                                     mx.float16)
                    self._check(q, k, v, scale, causal=causal)

    # ------------------------------------------------------------------
    # GQA + head_dim sweep
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_gqa_headdim_sweep(self):
        """GQA across all required head dimensions.

        float32 + D=256 is skipped — exceeds Metal threadgroup memory limit on
        the current (pre-chunked) kernel.  That case is the primary target of
        the chunked SDPA implementation and will be covered by Task 4 tests.
        """
        B, qL, kL = 1, 64, 64
        n_q, n_kv = 8, 2  # 4:1 ratio
        for D in [64, 80, 128, 256]:
            for dtype in [mx.float16, mx.float32]:
                # float32 + D=256 exceeds Metal threadgroup memory on current kernel
                if dtype == mx.float32 and D == 256:
                    continue
                with self.subTest(D=D, dtype=dtype):
                    q, k, v, scale = self._make_qkv(B, qL, kL, n_q, n_kv, D, dtype)
                    self._check(q, k, v, scale, causal=True,
                                atol=self._atol_for(dtype))

    # ------------------------------------------------------------------
    # Long context (8K)
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_long_context_8k(self):
        """Long-context self-attention at 8K tokens with causal mask."""
        B, qL, n_heads, D = 1, 8192, 8, 128
        mx.random.seed(7)
        scale = 1.0 / math.sqrt(D)
        # Use smaller values to reduce accumulation error at long range
        q = (0.1 * mx.random.normal((B, n_heads, qL, D))).astype(mx.float16)
        k = (0.1 * mx.random.normal((B, n_heads, qL, D))).astype(mx.float16)
        v = (0.1 * mx.random.normal((B, n_heads, qL, D))).astype(mx.float16)
        self._check(q, k, v, scale, causal=True, atol=1e-2)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_long_context_8k_decode(self):
        """Decode step against an 8K KV cache."""
        B, kL, n_heads, D = 1, 8192, 8, 128
        mx.random.seed(8)
        scale = 1.0 / math.sqrt(D)
        q = (0.1 * mx.random.normal((B, n_heads, 1, D))).astype(mx.float16)
        k = (0.1 * mx.random.normal((B, n_heads, kL, D))).astype(mx.float16)
        v = (0.1 * mx.random.normal((B, n_heads, kL, D))).astype(mx.float16)
        self._check(q, k, v, scale, causal=True, atol=1e-2)

    # ------------------------------------------------------------------
    # Batched inputs
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_batched(self):
        """Batch size > 1 with various head configurations."""
        D = 128
        for B, n_q, n_kv, qL, kL in [
            (2, 8, 8,  64,  64),
            (4, 8, 2,  32, 128),
            (2, 16, 4, 64,  64),
        ]:
            with self.subTest(B=B, n_q=n_q, n_kv=n_kv, qL=qL, kL=kL):
                q, k, v, scale = self._make_qkv(B, qL, kL, n_q, n_kv, D,
                                                 mx.float16)
                self._check(q, k, v, scale, causal=False)
                self._check(q, k, v, scale, causal=True)

    # ------------------------------------------------------------------
    # Reference logsumexp sanity check (no fused kernel needed)
    # ------------------------------------------------------------------

    def test_ref_logsumexp_identity(self):
        """The reference logsumexp must satisfy the online update identity.

        For a single chunk of K/V tokens the logsumexp of the attention scores
        equals log(sum(softmax(scores) * exp(scores))).  More concretely:

            logsumexp = log(sum_j exp(scale * q·k_j))

        We verify this against mx.logsumexp on the raw (unmasked) scores.
        This is a pure CPU/float32 test — no GPU kernel required.
        """
        mx.random.seed(0)
        B, n_q, qL, n_kv, kL, D = 1, 4, 8, 4, 16, 64
        scale = 1.0 / math.sqrt(D)

        q = mx.random.normal((B, n_q, qL, D))
        k = mx.random.normal((B, n_kv, kL, D))
        v = mx.random.normal((B, n_kv, kL, D))

        _, lse = ref_attention(q, k, v, scale, causal=False)

        # Independently compute logsumexp of the raw scaled scores
        raw_scores = (q * scale) @ mx.swapaxes(k, -1, -2)  # (B, H, qL, kL)
        expected_lse = mx.logsumexp(raw_scores, axis=-1)     # (B, H, qL)

        mx.eval(lse, expected_lse)
        max_diff = mx.max(mx.abs(lse - expected_lse)).item()
        self.assertLessEqual(max_diff, 1e-5,
                             msg=f"ref logsumexp drift: {max_diff:.2e}")

    def test_ref_logsumexp_causal(self):
        """Reference logsumexp must match raw scores with causal masking applied."""
        mx.random.seed(1)
        B, n_q, L, D = 1, 4, 32, 64
        scale = 1.0 / math.sqrt(D)

        q = mx.random.normal((B, n_q, L, D))
        k = mx.random.normal((B, n_q, L, D))
        v = mx.random.normal((B, n_q, L, D))

        _, lse = ref_attention(q, k, v, scale, causal=True)

        # Build the causal mask manually and compute expected logsumexp
        raw = (q * scale) @ mx.swapaxes(k, -1, -2)         # (B, H, L, L)
        q_idx = mx.arange(L)[:, None]
        k_idx = mx.arange(L)[None, :]
        mask = q_idx >= k_idx                               # (L, L) bool
        masked = mx.where(mask, raw, mx.array(-1e9, mx.float32))
        expected_lse = mx.logsumexp(masked, axis=-1)        # (B, H, L)

        mx.eval(lse, expected_lse)
        max_diff = mx.max(mx.abs(lse - expected_lse)).item()
        self.assertLessEqual(max_diff, 1e-4,
                             msg=f"causal ref logsumexp drift: {max_diff:.2e}")

    def test_ref_logsumexp_gqa(self):
        """Reference logsumexp tiles KV heads correctly for GQA."""
        mx.random.seed(2)
        B, n_q, n_kv, qL, kL, D = 1, 8, 2, 16, 32, 64
        scale = 1.0 / math.sqrt(D)

        q = mx.random.normal((B, n_q, qL, D))
        k = mx.random.normal((B, n_kv, kL, D))
        v = mx.random.normal((B, n_kv, kL, D))

        out, lse = ref_attention(q, k, v, scale, causal=False)

        # Shapes
        self.assertEqual(out.shape, (B, n_q, qL, D))
        self.assertEqual(lse.shape, (B, n_q, qL))

        # Cross-check: tile K manually and verify logsumexp
        n_rep = n_q // n_kv
        k_tiled = mx.repeat(k, n_rep, axis=1)              # (B, n_q, kL, D)
        raw = (q * scale) @ mx.swapaxes(k_tiled, -1, -2)   # (B, n_q, qL, kL)
        expected_lse = mx.logsumexp(raw, axis=-1)

        mx.eval(lse, expected_lse)
        max_diff = mx.max(mx.abs(lse - expected_lse)).item()
        self.assertLessEqual(max_diff, 1e-5,
                             msg=f"GQA ref logsumexp drift: {max_diff:.2e}")


if __name__ == "__main__":
    mlx_tests.MLXTestRunner(failfast=False)
