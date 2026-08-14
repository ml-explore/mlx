# Copyright © 2026 Apple Inc.

import gc
import json
import os
import struct
import tempfile
import unittest

import mlx.core as mx
import mlx_tests


def _write_aligned(tmpdir, tensors, align=64):
    """Pack tensors into a flat 64-byte-aligned bin via a scratch
    safetensors (gives exact serialized bytes per tensor)."""
    st = os.path.join(tmpdir, "src.safetensors")
    mx.save_safetensors(st, tensors)
    with open(st, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        blob = f.read()
    path = os.path.join(tmpdir, "packed.bin")
    index = {}
    with open(path, "wb") as out:
        for name, spec in hdr.items():
            if name == "__metadata__":
                continue
            b0, b1 = spec["data_offsets"]
            out.write(b"\0" * ((-out.tell()) % align))
            index[name] = (out.tell(), spec["shape"])
            out.write(blob[b0:b1])
    return path, index


class TestMmapWeights(mlx_tests.MLXTestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        mx.random.seed(7)
        cls.tensors = {
            "f32": mx.random.normal((123, 77)),
            "bf16": mx.random.normal((300, 400)).astype(mx.bfloat16),
            "f16": mx.random.normal((64, 32)).astype(mx.float16),
            "u32": (mx.random.uniform(shape=(50, 9)) * 1e6).astype(mx.uint32),
        }
        mx.eval(cls.tensors)
        cls.path, cls.index = _write_aligned(cls._tmp.name, cls.tensors)
        cls.mx_dtypes = {
            "f32": mx.float32,
            "bf16": mx.bfloat16,
            "f16": mx.float16,
            "u32": mx.uint32,
        }

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _view(self, name):
        off, shape = self.index[name]
        return mx.mmap_weights(self.path, off, shape, self.mx_dtypes[name])

    def _assert_bit_equal(self, a, b):
        if a.dtype in (mx.bfloat16, mx.float16):
            a, b = a.view(mx.uint16), b.view(mx.uint16)
        elif a.dtype == mx.float32:
            a, b = a.view(mx.uint32), b.view(mx.uint32)
        self.assertTrue(mx.array_equal(a, b).item())

    def test_bit_equality_all_dtypes(self):
        for name, ref in self.tensors.items():
            with self.subTest(dtype=name):
                self._assert_bit_equal(self._view(name), ref)

    def test_cpu_and_gpu_backends(self):
        for dev in (mx.cpu,) + ((mx.gpu,) if mx.metal.is_available() else ()):
            with self.subTest(device=dev):
                with mx.stream(dev):
                    v = self._view("f32")
                    s = (v * 2).sum()
                    mx.eval(s)
                    self.assertAlmostEqual(
                        s.item(), (self.tensors["f32"] * 2).sum().item(), places=3
                    )

    def test_ops_through_view(self):
        v = self._view("f32")
        ref = self.tensors["f32"]
        out_v = mx.softmax(v @ v.T, axis=-1)
        out_r = mx.softmax(ref @ ref.T, axis=-1)
        mx.eval(out_v, out_r)
        self._assert_bit_equal(out_v, out_r)

    def test_donation_window_never_mutates_mapping(self):
        # Drop the view's last reference before eval — if the buffer were
        # donated, the graph would write into the read-only mapping.
        expect = mx.array(self._view("bf16"))
        mx.eval(expect)
        v = self._view("bf16")
        out = mx.abs(-(v + mx.array(1.0, dtype=mx.bfloat16)))
        del v
        mx.eval(out)
        gc.collect()
        self._assert_bit_equal(self._view("bf16"), expect)

    def test_lifecycle_many_cycles(self):
        for _ in range(200):
            v = self._view("f32")
            mx.eval(v.sum())
            del v
        gc.collect()
        self._assert_bit_equal(self._view("f32"), self.tensors["f32"])

    def test_errors(self):
        off, shape = self.index["f32"]
        with self.assertRaises(Exception):
            mx.mmap_weights(self.path + ".missing", 0, [4], mx.float32)
        with self.assertRaises(Exception):  # out of bounds
            mx.mmap_weights(self.path, off, [10**6, 10**6], mx.float32)
        with self.assertRaises(Exception):  # misaligned for element size
            mx.mmap_weights(self.path, off + 1, shape, mx.float32)


if __name__ == "__main__":
    unittest.main()
