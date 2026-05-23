# Copyright © 2026 Apple Inc.

"""Tests for mx.load() raw K-quant tensor loading from GGUF files.

All 10 K-quant codecs route through the kquant loader, which loads raw wire
bytes as a uint8 array with codec metadata for dispatch.
"""

import os
import tempfile
import unittest

import mlx.core as mx
import mlx_tests
import numpy as np

try:
    from gguf import GGMLQuantizationType, GGUFWriter

    HAS_GGUF = True
except ImportError:
    HAS_GGUF = False


# Codec geometry. Source of truth: mlx/primitives.cpp kquant_codec_by_name.
KQUANT_CODECS = {
    "q4_0": (lambda: GGMLQuantizationType.Q4_0, 32, 18),
    "q4_1": (lambda: GGMLQuantizationType.Q4_1, 32, 20),
    "q5_0": (lambda: GGMLQuantizationType.Q5_0, 32, 22),
    "q5_1": (lambda: GGMLQuantizationType.Q5_1, 32, 24),
    "q8_0": (lambda: GGMLQuantizationType.Q8_0, 32, 34),
    "q2_k": (lambda: GGMLQuantizationType.Q2_K, 256, 84),
    "q3_k": (lambda: GGMLQuantizationType.Q3_K, 256, 110),
    "q4_k": (lambda: GGMLQuantizationType.Q4_K, 256, 144),
    "q5_k": (lambda: GGMLQuantizationType.Q5_K, 256, 176),
    "q6_k": (lambda: GGMLQuantizationType.Q6_K, 256, 210),
}

# All K-quant codecs route through the raw kquant loader.
KQUANT_RAW_TYPES = (
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
)


@unittest.skipUnless(HAS_GGUF, "gguf package not installed")
class TestGGUFKQuantLoad(mlx_tests.MLXTestCase):
    def _write_gguf(self, path, codec_name, N, K, tensor_name="blk.0.attn_q.weight"):
        type_factory, wpb, bpb = KQUANT_CODECS[codec_name]
        gguf_type = type_factory()
        assert K % wpb == 0
        bytes_per_row = (K // wpb) * bpb
        total_bytes = N * bytes_per_row
        rng = np.random.default_rng(42)
        raw = rng.integers(0, 256, size=total_bytes, dtype=np.uint8)

        writer = GGUFWriter(path, arch="test")
        # raw_shape uses numpy convention: last dim is innermost (in bytes when
        # raw_dtype is a quantized type). Library reverses to GGML order on disk.
        writer.add_tensor(
            tensor_name,
            raw,
            raw_shape=(N, bytes_per_row),
            raw_dtype=gguf_type,
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return raw

    def test_kquant_raw_shape_dtype_bytes(self):
        for codec in KQUANT_RAW_TYPES:
            with self.subTest(codec=codec):
                _, wpb, bpb = KQUANT_CODECS[codec]
                N, K = 4, wpb * 2  # 2 blocks per row
                with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
                    path = f.name
                try:
                    raw = self._write_gguf(path, codec, N, K)
                    arrays, _ = mx.load(path, return_metadata=True)
                    w = arrays["blk.0.attn_q.weight"]
                    self.assertEqual(w.dtype, mx.uint8)
                    expected_shape = [N, (K // wpb) * bpb]
                    self.assertEqual(list(w.shape), expected_shape)
                    np.testing.assert_array_equal(np.array(w).flatten(), raw)
                finally:
                    os.unlink(path)

    def test_kquant_metadata(self):
        codec = "q4_k"
        _, wpb, _ = KQUANT_CODECS[codec]
        N, K = 4, wpb * 2
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            path = f.name
        try:
            self._write_gguf(path, codec, N, K)
            _, metadata = mx.load(path, return_metadata=True)
            self.assertIn("__kquant_types__", metadata)
            kq_types = metadata["__kquant_types__"]
            self.assertIsInstance(kq_types, list)
            self.assertEqual(len(kq_types), 1)
            self.assertEqual(kq_types[0], "blk.0.attn_q.weight:q4_k")
        finally:
            os.unlink(path)

    def test_placeholder_scales(self):
        codec = "q6_k"
        _, wpb, _ = KQUANT_CODECS[codec]
        N, K = 4, wpb * 2
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            path = f.name
        try:
            self._write_gguf(path, codec, N, K)
            arrays, _ = mx.load(path, return_metadata=True)
            self.assertIn("blk.0.attn_q.scales", arrays)
            s = arrays["blk.0.attn_q.scales"]
            self.assertEqual(s.dtype, mx.uint8)
            self.assertEqual(list(s.shape), [1])
            self.assertEqual(s.size, 1)
        finally:
            os.unlink(path)

    def test_q4_0_loads_as_kquant(self):
        """Q4_0 routes through the kquant raw loader (not legacy affine)."""
        codec = "q4_0"
        _, wpb, bpb = KQUANT_CODECS[codec]
        N, K = 4, 32
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            path = f.name
        try:
            raw = self._write_gguf(path, codec, N, K, "blk.0.ffn.weight")
            arrays, metadata = mx.load(path, return_metadata=True)
            w = arrays["blk.0.ffn.weight"]
            self.assertEqual(w.dtype, mx.uint8)
            expected_shape = [N, (K // wpb) * bpb]
            self.assertEqual(list(w.shape), expected_shape)
            np.testing.assert_array_equal(np.array(w).flatten(), raw)
            kq = metadata.get("__kquant_types__", [])
            names = [e.split(":")[0] for e in kq] if isinstance(kq, list) else []
            self.assertIn("blk.0.ffn.weight", names)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
