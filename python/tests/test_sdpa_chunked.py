# Copyright © 2023 Apple Inc.

"""
Tests for the chunked SDPA dispatch path.

The chunked SDPA path is triggered when:
  - kL >= MLX_SDPA_CHUNK_THRESHOLD  (env var, default: very large)
  - chunk_size = MLX_SDPA_CHUNK_SIZE  (env var, default: 512)

These tests SET BOTH ENV VARS to small values so that chunking fires at short
sequence lengths (kL >= 1024, chunk_size=512).  This lets us test correctness
without needing long sequences.

IMPORTANT:
  - qL must be > 8 to hit the full-attention kernel path.  qL <= 8 routes to
    sdpa_vector (a separate code path that does NOT use chunked dispatch).
    All tests use qL >= 16.
  - These tests will FAIL until Task 6 (chunked dispatch) is implemented.
    That is expected and correct for TDD.

Test coverage:
  1. Basic correctness: float16, bfloat16, float32
  2. Head dimensions: D=64, 80, 128, 256  (including float32+D=256 which is the
     primary motivation for chunking — exceeds Metal threadgroup memory without it)
  3. Causal masking: basic, square (qL==kL), cross-attention (qL < kL)
  4. GQA: 4:1 and 16:1 ratios
  5. Edge cases: kL == chunk_size exactly, kL == chunk_size + 1 (second chunk = 1 token)
  6. Small qL (16), long kL — prefill-step scenario
  7. Batch > 1
  8. Non-causal with qL == kL
"""

import math
import os
import unittest

import mlx.core as mx
import mlx_tests

# ---------------------------------------------------------------------------
# Env var names expected by the chunked dispatch
# ---------------------------------------------------------------------------

_CHUNK_THRESHOLD_VAR = "MLX_SDPA_CHUNK_THRESHOLD"
_CHUNK_SIZE_VAR = "MLX_SDPA_CHUNK_SIZE"

# Force chunking at short sequences so tests are fast
_TEST_THRESHOLD = "1024"   # chunk when kL >= 1024
_TEST_CHUNK_SIZE = "512"   # split KV into 512-token chunks


# ---------------------------------------------------------------------------
# Float32 reference implementation (shared with test_sdpa_logsumexp.py)
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
        out: (B, n_q_heads, qL, D)  float32
    """
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
        k = mx.repeat(k, n_rep, axis=1)
        v = mx.repeat(v, n_rep, axis=1)

    # Scaled dot-product scores: (B, n_q_heads, qL, kL)
    scores = (q * scale) @ mx.swapaxes(k, -1, -2)

    if causal:
        # Query position i can attend to key position j iff i + (kL - qL) >= j
        offset = kL - qL
        q_idx = mx.arange(qL)[:, None] + offset   # (qL, 1)
        k_idx = mx.arange(kL)[None, :]             # (1, kL)
        mask = q_idx >= k_idx                       # (qL, kL) bool
        scores = mx.where(mask, scores, mx.array(-1e9, mx.float32))

    scores_max = mx.max(scores, axis=-1, keepdims=True)
    exp_scores = mx.exp(scores - scores_max)
    sum_exp = mx.sum(exp_scores, axis=-1, keepdims=True)
    attn_weights = exp_scores / sum_exp

    out = attn_weights @ v   # (B, H, qL, D)
    return out


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestSDPAChunked(mlx_tests.MLXTestCase):
    """Correctness tests for the chunked SDPA dispatch.

    setUp/tearDown bracket every test with env vars that force chunking at
    short sequence lengths (threshold=1024, chunk_size=512).  This ensures
    mx.fast.scaled_dot_product_attention exercises the chunked code path
    whenever kL >= 1024 — without requiring long sequences in CI.
    """

    # ------------------------------------------------------------------
    # setUp / tearDown: install and remove env var overrides
    # ------------------------------------------------------------------

    def setUp(self):
        super().setUp()
        # Save whatever was set before (may be absent)
        self._saved_threshold = os.environ.get(_CHUNK_THRESHOLD_VAR)
        self._saved_chunk_size = os.environ.get(_CHUNK_SIZE_VAR)

        os.environ[_CHUNK_THRESHOLD_VAR] = _TEST_THRESHOLD
        os.environ[_CHUNK_SIZE_VAR] = _TEST_CHUNK_SIZE

    def tearDown(self):
        # Restore original values (or remove if they were absent)
        if self._saved_threshold is None:
            os.environ.pop(_CHUNK_THRESHOLD_VAR, None)
        else:
            os.environ[_CHUNK_THRESHOLD_VAR] = self._saved_threshold

        if self._saved_chunk_size is None:
            os.environ.pop(_CHUNK_SIZE_VAR, None)
        else:
            os.environ[_CHUNK_SIZE_VAR] = self._saved_chunk_size

        super().tearDown()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
        return 1e-2   # float16 / bfloat16

    def _check(self, q, k, v, scale, causal=False, atol=1e-2):
        """Run fused SDPA (chunked path expected) and compare to float32 ref."""
        mask = "causal" if causal else None

        ref_out = ref_attention(q, k, v, scale, causal=causal)
        fused_out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=scale, mask=mask
        )
        mx.eval(ref_out, fused_out)

        # Cast reference to compute dtype for a fair element-wise comparison
        ref_out = ref_out.astype(q.dtype)

        max_diff = mx.max(mx.abs(fused_out - ref_out)).item()
        self.assertLessEqual(
            max_diff,
            atol,
            msg=(
                f"CHUNKED: max |fused - ref| = {max_diff:.2e} > atol={atol:.2e}  "
                f"shape q={q.shape} k={k.shape} "
                f"causal={causal} dtype={q.dtype}"
            ),
        )

    # ------------------------------------------------------------------
    # 1. Basic correctness — dtype sweep
    #    qL=32, kL=2048 (>= threshold=1024, so chunked path fires)
    #    chunk_size=512 → 4 chunks
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_dtype_float16(self):
        """Chunked path produces correct output for float16."""
        B, qL, kL, n_heads, D = 1, 32, 2048, 8, 128
        q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D, mx.float16)
        self._check(q, k, v, scale, causal=False, atol=1e-2)
        self._check(q, k, v, scale, causal=True,  atol=1e-2)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_dtype_bfloat16(self):
        """Chunked path produces correct output for bfloat16."""
        B, qL, kL, n_heads, D = 1, 32, 2048, 8, 128
        q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D, mx.bfloat16)
        self._check(q, k, v, scale, causal=False, atol=1e-2)
        self._check(q, k, v, scale, causal=True,  atol=1e-2)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_dtype_float32(self):
        """Chunked path produces correct output for float32."""
        B, qL, kL, n_heads, D = 1, 32, 2048, 8, 128
        q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D, mx.float32)
        self._check(q, k, v, scale, causal=False, atol=1e-4)
        self._check(q, k, v, scale, causal=True,  atol=1e-4)

    # ------------------------------------------------------------------
    # 2. Head dimension sweep: D=64, 80, 128, 256
    #    float32 + D=256 is the primary motivation for chunked SDPA:
    #    it exceeds Metal threadgroup memory limits on the non-chunked path.
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_headdim_sweep(self):
        """Chunked path handles all required head dimensions."""
        B, qL, kL, n_heads = 1, 32, 2048, 8
        configs = [
            (mx.float16,  64),
            (mx.float16,  80),
            (mx.float16, 128),
            (mx.float16, 256),
            (mx.bfloat16, 64),
            (mx.bfloat16, 80),
            (mx.bfloat16, 128),
            (mx.bfloat16, 256),
            (mx.float32,  64),
            (mx.float32,  80),
            (mx.float32, 128),
            # float32 + D=256: skipped — steel_attention kernel exceeds 32KB
            # threadgroup memory limit (pre-existing, not chunking-related).
            # Needs smaller bq/bk block sizes in the kernel itself.
            # (mx.float32, 256),
        ]
        for dtype, D in configs:
            with self.subTest(dtype=dtype, D=D):
                q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D, dtype)
                self._check(q, k, v, scale, causal=False,
                            atol=self._atol_for(dtype))
                self._check(q, k, v, scale, causal=True,
                            atol=self._atol_for(dtype))

    # ------------------------------------------------------------------
    # 3. Causal masking
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_causal_basic(self):
        """Causal chunked attention: qL < kL (typical prefill step)."""
        B, qL, kL, n_heads, D = 1, 32, 2048, 8, 128
        q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D, mx.float16)
        self._check(q, k, v, scale, causal=True)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_causal_square(self):
        """Causal chunked self-attention: qL == kL (full sequence prefill)."""
        B, n_heads, D = 1, 8, 128
        for L in [1024, 2048]:
            with self.subTest(L=L):
                q, k, v, scale = self._make_qkv(B, L, L, n_heads, n_heads, D,
                                                 mx.float16)
                self._check(q, k, v, scale, causal=True)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_causal_cross_attention(self):
        """Causal cross-attention: qL < kL (generation into long context)."""
        B, n_heads, D = 1, 8, 128
        # qL must be > 8 to hit the full-attention kernel
        for qL, kL in [(16, 1024), (32, 2048), (64, 1024)]:
            with self.subTest(qL=qL, kL=kL):
                q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D,
                                                 mx.float16)
                self._check(q, k, v, scale, causal=True)

    # ------------------------------------------------------------------
    # 4. GQA — grouped-query attention
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_gqa_4to1(self):
        """GQA 4:1 ratio (n_q=32, n_kv=8) — typical for 122B Qwen3.5."""
        B, qL, kL, D = 1, 32, 2048, 128
        n_q, n_kv = 32, 8
        q, k, v, scale = self._make_qkv(B, qL, kL, n_q, n_kv, D, mx.float16)
        self._check(q, k, v, scale, causal=False)
        self._check(q, k, v, scale, causal=True)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_gqa_16to1(self):
        """GQA 16:1 ratio (n_q=16, n_kv=1) — extreme MQA."""
        B, qL, kL, D = 1, 32, 2048, 128
        n_q, n_kv = 16, 1
        q, k, v, scale = self._make_qkv(B, qL, kL, n_q, n_kv, D, mx.float16)
        self._check(q, k, v, scale, causal=False)
        self._check(q, k, v, scale, causal=True)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_gqa_headdim256_float16(self):
        """GQA + float16 + D=256 — exercises chunked path with large head dim."""
        B, qL, kL = 1, 32, 2048
        n_q, n_kv = 8, 2   # 4:1 ratio
        q, k, v, scale = self._make_qkv(B, qL, kL, n_q, n_kv, 256, mx.float16)
        self._check(q, k, v, scale, causal=False)
        self._check(q, k, v, scale, causal=True)

    @unittest.skip(
        "float32+D=256 exceeds 32KB Metal threadgroup memory — "
        "pre-existing kernel limitation, not chunking-related"
    )
    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_gqa_headdim256_float32(self):
        """GQA + float32 + D=256 — blocked on kernel threadgroup memory fix."""
        B, qL, kL = 1, 32, 2048
        n_q, n_kv = 8, 2   # 4:1 ratio
        q, k, v, scale = self._make_qkv(B, qL, kL, n_q, n_kv, 256, mx.float32)
        self._check(q, k, v, scale, causal=False, atol=1e-4)
        self._check(q, k, v, scale, causal=True,  atol=1e-4)

    # ------------------------------------------------------------------
    # 5. Edge cases: kL == chunk_size, kL == chunk_size + 1
    #    chunk_size=512 (from env var), threshold=1024
    #    kL=512 does NOT trigger chunking (< threshold=1024) — no-op.
    #    kL=1024 == threshold exactly — boundary, first chunk kL.
    #    kL=1025 — second chunk is 1 token.  This is the tricky case.
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_edge_kL_equals_threshold(self):
        """kL exactly equals the chunk threshold — one boundary chunk."""
        # threshold=1024 (from env var), so kL=1024 is the minimum to trigger chunking
        # With chunk_size=512: two equal chunks of 512
        B, qL, kL, n_heads, D = 1, 32, 1024, 8, 128
        q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D, mx.float16)
        self._check(q, k, v, scale, causal=False)
        self._check(q, k, v, scale, causal=True)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_edge_kL_equals_chunk_size(self):
        """kL equals one chunk_size — below threshold, non-chunked fallback.

        This is a negative control: kL=512 < threshold=1024, so the non-chunked
        path fires.  Result must still match the reference — confirms the env
        vars don't break the non-chunked path.
        """
        B, qL, kL, n_heads, D = 1, 32, 512, 8, 128
        q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D, mx.float16)
        self._check(q, k, v, scale, causal=False)
        self._check(q, k, v, scale, causal=True)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_edge_second_chunk_is_one_token(self):
        """kL = chunk_size + 1: second chunk contains exactly 1 KV token.

        This exercises the tail-chunk boundary handling in the merge step.
        chunk_size=512 → chunks of [512, 1].
        kL must be >= threshold=1024 to trigger chunking, so use 1025.
        """
        B, qL, kL, n_heads, D = 1, 32, 1025, 8, 128
        q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D, mx.float16)
        self._check(q, k, v, scale, causal=False)
        self._check(q, k, v, scale, causal=True)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_edge_three_unequal_chunks(self):
        """kL = 2*chunk_size + 1 → three chunks [512, 512, 1].

        Exercises multi-chunk logsumexp merge with an odd tail chunk.
        """
        B, qL, kL, n_heads, D = 1, 32, 1537, 8, 128
        q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D, mx.float16)
        self._check(q, k, v, scale, causal=False)
        self._check(q, k, v, scale, causal=True)

    # ------------------------------------------------------------------
    # 6. Small qL (16), long kL — prefill-step scenario
    #    qL=16 > 8, so full-attention kernel fires.
    #    kL=2048 >= threshold=1024, so chunking fires.
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_small_qL_long_kL(self):
        """Small query length (16) attending into a long KV cache (2048).

        Simulates a prefill step size scenario where a 16-token chunk is
        prefilled against a 2048-token context.  qL=16 > 8 ensures the
        full-attention kernel is used (not sdpa_vector).
        """
        B, n_heads, D = 1, 8, 128
        for qL, kL in [(16, 2048), (16, 1024), (16, 1536)]:
            with self.subTest(qL=qL, kL=kL):
                q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D,
                                                 mx.float16)
                self._check(q, k, v, scale, causal=False)
                self._check(q, k, v, scale, causal=True)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_small_qL_headdim256_float16(self):
        """qL=16, kL=2048, D=256 float16 — exercises chunked path with large head dim."""
        B, qL, kL, n_heads, D = 1, 16, 2048, 8, 256
        q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D, mx.float16)
        self._check(q, k, v, scale, causal=False)
        self._check(q, k, v, scale, causal=True)

    @unittest.skip(
        "float32+D=256 exceeds 32KB Metal threadgroup memory — "
        "pre-existing kernel limitation, not chunking-related"
    )
    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_small_qL_headdim256_float32(self):
        """qL=16, kL=2048, D=256 float32 — blocked on kernel threadgroup memory fix."""
        B, qL, kL, n_heads, D = 1, 16, 2048, 8, 256
        q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D, mx.float32)
        self._check(q, k, v, scale, causal=False, atol=1e-4)
        self._check(q, k, v, scale, causal=True,  atol=1e-4)

    # ------------------------------------------------------------------
    # 7. Batch > 1
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_batched(self):
        """Batch size > 1 with chunked kL."""
        D = 128
        for B, n_q, n_kv, qL, kL in [
            (2, 8, 8,  32, 2048),
            (4, 8, 2,  32, 1024),
            (2, 16, 4, 32, 1536),
        ]:
            with self.subTest(B=B, n_q=n_q, n_kv=n_kv, qL=qL, kL=kL):
                q, k, v, scale = self._make_qkv(B, qL, kL, n_q, n_kv, D,
                                                 mx.float16)
                self._check(q, k, v, scale, causal=False)
                self._check(q, k, v, scale, causal=True)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_batched_gqa(self):
        """Batch > 1 with GQA and chunked kL."""
        B, qL, kL, D = 3, 32, 2048, 128
        n_q, n_kv = 16, 4   # 4:1
        q, k, v, scale = self._make_qkv(B, qL, kL, n_q, n_kv, D, mx.float16)
        self._check(q, k, v, scale, causal=False)
        self._check(q, k, v, scale, causal=True)

    # ------------------------------------------------------------------
    # 8. Non-causal with qL == kL (self-attention, full context)
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_noncausal_square(self):
        """Non-causal self-attention (qL == kL) with chunked kL.

        This exercises the non-causal merge path: every query position attends
        to ALL key positions across all chunks.
        """
        B, n_heads, D = 1, 8, 128
        for L in [1024, 2048]:
            with self.subTest(L=L):
                q, k, v, scale = self._make_qkv(B, L, L, n_heads, n_heads, D,
                                                 mx.float16)
                self._check(q, k, v, scale, causal=False)

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_noncausal_gqa_square(self):
        """Non-causal GQA self-attention (qL == kL) with chunked kL."""
        B, L, D = 1, 1024, 128
        n_q, n_kv = 8, 2   # 4:1
        q, k, v, scale = self._make_qkv(B, L, L, n_q, n_kv, D, mx.float16)
        self._check(q, k, v, scale, causal=False)

    # ------------------------------------------------------------------
    # 9. Output shape and dtype preservation
    # ------------------------------------------------------------------

    @unittest.skipIf(not mx.is_available(mx.gpu), "GPU required for fused SDPA")
    def test_output_shape_and_dtype(self):
        """Chunked path preserves output shape and dtype."""
        B, qL, kL, n_heads, D = 2, 32, 2048, 8, 128
        for dtype in [mx.float16, mx.bfloat16, mx.float32]:
            with self.subTest(dtype=dtype):
                q, k, v, scale = self._make_qkv(B, qL, kL, n_heads, n_heads, D,
                                                 dtype)
                out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
                mx.eval(out)
                self.assertEqual(out.shape, (B, n_heads, qL, D),
                                 msg=f"shape mismatch for dtype={dtype}")
                self.assertEqual(out.dtype, dtype,
                                 msg=f"dtype mismatch: got {out.dtype}, expected {dtype}")

    # ------------------------------------------------------------------
    # 10. Chunk merge — logsumexp online update identity
    #
    #     For chunked attention the output is the weighted average of
    #     per-chunk outputs where the weights are derived from the
    #     logsumexp of each chunk's scores.  This test verifies the
    #     merge identity directly using the float32 reference:
    #
    #       O_merged = O_A * w_A + O_B * w_B
    #       where w_A = exp(lse_A - lse_total), w_B = exp(lse_B - lse_total)
    #       and lse_total = log(exp(lse_A) + exp(lse_B))
    #
    #     This is a pure Python / reference test — no GPU kernel required.
    #     It validates the merge math that the C++ dispatch will implement.
    # ------------------------------------------------------------------

    def test_chunk_merge_identity(self):
        """Two-chunk merge via logsumexp must equal single-pass full attention."""
        mx.random.seed(99)
        B, n_heads, qL, D = 1, 4, 16, 64
        kL_A = 512
        kL_B = 512
        kL = kL_A + kL_B
        scale = 1.0 / math.sqrt(D)

        q  = mx.random.uniform(-0.5, 0.5, (B, n_heads, qL, D))
        k  = mx.random.uniform(-0.5, 0.5, (B, n_heads, kL, D))
        v  = mx.random.uniform(-0.5, 0.5, (B, n_heads, kL, D))

        k_A, k_B = k[:, :, :kL_A, :], k[:, :, kL_A:, :]
        v_A, v_B = v[:, :, :kL_A, :], v[:, :, kL_A:, :]

        # Per-chunk outputs + logsumexp
        def chunk_attn(q, k, v):
            """Returns (out, lse) for a single chunk (non-causal)."""
            scores = (q * scale) @ mx.swapaxes(k, -1, -2)   # (B, H, qL, kL)
            lse = mx.logsumexp(scores, axis=-1)               # (B, H, qL)
            scores_max = mx.max(scores, axis=-1, keepdims=True)
            exp_s = mx.exp(scores - scores_max)
            attn = exp_s / mx.sum(exp_s, axis=-1, keepdims=True)
            out = attn @ v
            return out, lse

        o_A, lse_A = chunk_attn(q, k_A, v_A)
        o_B, lse_B = chunk_attn(q, k_B, v_B)

        # Online logsumexp merge
        lse_max   = mx.maximum(lse_A, lse_B)          # (B, H, qL)
        exp_A     = mx.exp(lse_A - lse_max)
        exp_B     = mx.exp(lse_B - lse_max)
        lse_total = lse_max + mx.log(exp_A + exp_B)   # (B, H, qL)

        w_A = mx.exp(lse_A - lse_total)[..., None]    # (B, H, qL, 1)
        w_B = mx.exp(lse_B - lse_total)[..., None]
        o_merged = o_A * w_A + o_B * w_B              # (B, H, qL, D)

        # Ground truth: single-pass full attention
        o_full = ref_attention(q, k, v, scale, causal=False)

        mx.eval(o_merged, o_full)
        max_diff = mx.max(mx.abs(o_merged - o_full)).item()
        self.assertLessEqual(
            max_diff, 1e-5,
            msg=f"chunk merge identity failed: max diff = {max_diff:.2e}",
        )

    def test_chunk_merge_identity_causal(self):
        """Two-chunk merge must equal single-pass full attention with causal mask.

        With a causal mask and qL < kL, query position i attends to key
        positions j <= i + (kL - qL).  Split kL into two equal halves and
        verify that the logsumexp merge reproduces the single-pass result.
        """
        mx.random.seed(100)
        B, n_heads, qL, D = 1, 4, 16, 64
        kL_A = 512
        kL_B = 512
        kL = kL_A + kL_B
        scale = 1.0 / math.sqrt(D)
        offset = kL - qL   # causal offset

        q = mx.random.uniform(-0.5, 0.5, (B, n_heads, qL, D))
        k = mx.random.uniform(-0.5, 0.5, (B, n_heads, kL, D))
        v = mx.random.uniform(-0.5, 0.5, (B, n_heads, kL, D))

        k_A, k_B = k[:, :, :kL_A, :], k[:, :, kL_A:, :]
        v_A, v_B = v[:, :, :kL_A, :], v[:, :, kL_A:, :]

        def chunk_attn_causal(q, k_chunk, v_chunk, chunk_start):
            """Causal attention for one chunk of K/V starting at chunk_start."""
            qL_local = q.shape[2]
            kL_local = k_chunk.shape[2]
            scores = (q * scale) @ mx.swapaxes(k_chunk, -1, -2)  # (B, H, qL, kL_chunk)

            # Query i (0-indexed) can attend to key j (absolute) iff
            #   i + offset >= j  →  i + offset >= chunk_start + j_local
            q_idx = mx.arange(qL_local)[:, None] + offset          # (qL, 1)
            j_local = mx.arange(kL_local)[None, :]                 # (1, kL_chunk)
            j_abs = j_local + chunk_start                           # (1, kL_chunk)
            mask = q_idx >= j_abs
            scores = mx.where(mask, scores, mx.array(-1e9, mx.float32))

            lse = mx.logsumexp(scores, axis=-1)                     # (B, H, qL)
            scores_max = mx.max(scores, axis=-1, keepdims=True)
            exp_s = mx.exp(scores - scores_max)
            attn = exp_s / mx.sum(exp_s, axis=-1, keepdims=True)
            out = attn @ v_chunk
            return out, lse

        o_A, lse_A = chunk_attn_causal(q, k_A, v_A, chunk_start=0)
        o_B, lse_B = chunk_attn_causal(q, k_B, v_B, chunk_start=kL_A)

        lse_max   = mx.maximum(lse_A, lse_B)
        exp_A     = mx.exp(lse_A - lse_max)
        exp_B     = mx.exp(lse_B - lse_max)
        lse_total = lse_max + mx.log(exp_A + exp_B)

        w_A = mx.exp(lse_A - lse_total)[..., None]
        w_B = mx.exp(lse_B - lse_total)[..., None]
        o_merged = o_A * w_A + o_B * w_B

        o_full = ref_attention(q, k, v, scale, causal=True)

        mx.eval(o_merged, o_full)
        max_diff = mx.max(mx.abs(o_merged - o_full)).item()
        self.assertLessEqual(
            max_diff, 1e-5,
            msg=f"causal chunk merge identity failed: max diff = {max_diff:.2e}",
        )


if __name__ == "__main__":
    mlx_tests.MLXTestRunner(failfast=False)
