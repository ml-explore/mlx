# Copyright © 2023 Apple Inc.

import os
import platform
import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
import mlx_tests
import numpy as np


class TestLoad(mlx_tests.MLXTestCase):
    dtypes = [
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float16",
        "bfloat16",
        "complex64",
    ]

    def _to_np_dtype(self, dt):
        """Helper to map MLX dtypes to NumPy-compatible dtypes."""
        return np.float32 if dt == "bfloat16" else getattr(np, dt)

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        if not os.path.isdir(cls.test_dir):
            os.mkdir(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_save_and_load(self):
        for dt in self.dtypes:
            with self.subTest(dtype=dt):
                for i, shape in enumerate([(1,), (23,), (1024, 1024), (4, 6, 3, 1, 2)]):
                    with self.subTest(shape=shape):
                        s_mlx = os.path.join(self.test_dir, f"mlx_{dt}_{i}.npy")
                        s_npy = os.path.join(self.test_dir, f"npy_{dt}_{i}.npy")
                        save_arr = np.random.uniform(0.0, 32.0, size=shape)

                        if dt == "bfloat16":
                            save_arr_mlx = mx.array(save_arr, mx.bfloat16)
                            save_arr_npy = save_arr.astype(np.float32)
                        else:
                            save_arr_npy = save_arr.astype(getattr(np, dt))
                            save_arr_mlx = mx.array(save_arr_npy)

                        mx.save(s_mlx, save_arr_mlx)
                        np.save(s_npy, save_arr_npy)

                        # Load and verify
                        self.assertTrue(mx.array_equal(mx.load(s_mlx), save_arr_mlx))
                        l_npy_mlx = mx.load(s_npy)
                        if dt == "bfloat16":
                            self.assertTrue(
                                mx.array_equal(
                                    l_npy_mlx.astype(mx.float32), mx.array(save_arr_npy)
                                )
                            )
                        else:
                            self.assertTrue(mx.array_equal(l_npy_mlx, save_arr_mlx))

                        if dt != "bfloat16":
                            self.assertTrue(
                                np.array_equal(np.load(s_mlx), save_arr_npy)
                            )

        # Path-based test
        s_file = os.path.join(self.test_dir, f"mlx_path.npy")
        save_arr = mx.ones((32,))
        mx.save(Path(s_file), save_arr)
        self.assertTrue(mx.array_equal(mx.load(Path(s_file)), save_arr))

    def test_load_npy_dtype(self):
        save_file = os.path.join(self.test_dir, "mlx_path.npy")
        a = np.random.randn(8).astype(np.float64)
        np.save(save_file, a)
        out = mx.load(save_file, stream=mx.cpu)
        self.assertEqual(out.dtype, mx.float64)
        self.assertTrue(np.array_equal(np.array(out), a))

    def test_save_and_load_safetensors(self):
        for dt in self.dtypes:
            with self.subTest(dtype=dt):
                for i, shape in enumerate([(1,), (23,)]):
                    s_file = os.path.join(self.test_dir, f"test_{dt}_{i}.safetensors")
                    arr = (
                        mx.random.normal(shape, dtype=getattr(mx, dt))
                        if "float" in dt
                        else mx.ones(shape, dtype=getattr(mx, dt))
                    )
                    mx.save_safetensors(s_file, {"t": arr})
                    self.assertTrue(mx.array_equal(mx.load(s_file)["t"], arr))

    @unittest.skipIf(platform.system() == "Windows", "GGUF disabled on Windows")
    def test_save_and_load_gguf(self):
        # bfloat16 is currently not supported by the GGUF C++ backend
        supported = ["float16", "float32", "int8", "int16", "int32"]
        for dt in supported:
            s_file = os.path.join(self.test_dir, f"test_{dt}.gguf")
            arr = mx.ones((8, 8), dtype=getattr(mx, dt))
            mx.save_gguf(s_file, {"t": arr})
            self.assertTrue(mx.array_equal(mx.load(s_file)["t"], arr))

    def test_save_and_load_fs(self):
        for dt in self.dtypes:
            with self.subTest(dtype=dt):
                s_file = os.path.join(self.test_dir, f"fs_{dt}.npy")
                save_arr_npy = np.random.uniform(0.0, 32.0, size=(8, 8)).astype(
                    self._to_np_dtype(dt)
                )
                save_arr_mlx = mx.array(save_arr_npy).astype(getattr(mx, dt))
                with open(s_file, "wb") as f:
                    mx.save(f, save_arr_mlx)
                with open(s_file, "rb") as f:
                    self.assertTrue(mx.array_equal(mx.load(f), save_arr_mlx))
                if dt != "bfloat16":
                    self.assertTrue(np.array_equal(np.load(s_file), save_arr_npy))

    def test_savez_and_loadz(self):
        for dt in self.dtypes:
            with self.subTest(dtype=dt):
                s_file = os.path.join(self.test_dir, f"test_{dt}.npz")
                s_npy = {
                    f"a_{i}": np.random.uniform(0, 32, (4,)).astype(
                        self._to_np_dtype(dt)
                    )
                    for i in range(2)
                }
                s_mlx = {
                    k: mx.array(v).astype(getattr(mx, dt)) for k, v in s_npy.items()
                }
                mx.savez(s_file, **s_mlx)
                l_mlx = mx.load(s_file)
                for k in s_mlx:
                    self.assertTrue(mx.array_equal(l_mlx[k], s_mlx[k]))
                if dt != "bfloat16":
                    l_npy = np.load(s_file)
                    for k in s_npy:
                        self.assertTrue(np.array_equal(l_npy[k], s_npy[k]))

    def test_non_contiguous(self):
        # Ensure non-contiguous arrays are handled correctly
        a = mx.arange(4).reshape(2, 2).T
        s_file = os.path.join(self.test_dir, "noncontig.safetensors")
        mx.save_safetensors(s_file, {"a": a})
        self.assertTrue(mx.array_equal(mx.load(s_file)["a"], a))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
