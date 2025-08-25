# Copyright © 2023 Apple Inc.

import os
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
        "complex64",
    ]

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
                        save_file_mlx = os.path.join(self.test_dir, f"mlx_{dt}_{i}.npy")
                        save_file_npy = os.path.join(self.test_dir, f"npy_{dt}_{i}.npy")

                        save_arr = np.random.uniform(0.0, 32.0, size=shape)
                        save_arr_npy = save_arr.astype(getattr(np, dt))
                        save_arr_mlx = mx.array(save_arr_npy)

                        mx.save(save_file_mlx, save_arr_mlx)
                        np.save(save_file_npy, save_arr_npy)

                        # Load array saved by mlx as mlx array
                        load_arr_mlx_mlx = mx.load(save_file_mlx)
                        self.assertTrue(mx.array_equal(load_arr_mlx_mlx, save_arr_mlx))

                        # Load array saved by numpy as mlx array
                        load_arr_npy_mlx = mx.load(save_file_npy)
                        self.assertTrue(mx.array_equal(load_arr_npy_mlx, save_arr_mlx))

                        # Load array saved by mlx as numpy array
                        load_arr_mlx_npy = np.load(save_file_mlx)
                        self.assertTrue(np.array_equal(load_arr_mlx_npy, save_arr_npy))

        save_file = os.path.join(self.test_dir, f"mlx_path.npy")
        save_arr = mx.ones((32,))
        mx.save(Path(save_file), save_arr)

        # Load array saved by mlx as mlx array
        load_arr = mx.load(Path(save_file))
        self.assertTrue(mx.array_equal(load_arr, save_arr))

    def test_save_and_load_safetensors(self):
        test_file = os.path.join(self.test_dir, "test.safetensors")
        with self.assertRaises(Exception):
            mx.save_safetensors(test_file, {"a": mx.ones((4, 4))}, {"testing": 0})

        for obj in [str, Path]:
            mx.save_safetensors(
                obj(test_file),
                {"test": mx.ones((2, 2))},
                {"testing": "test", "format": "mlx"},
            )
            res = mx.load(obj(test_file), return_metadata=True)
            self.assertEqual(len(res), 2)
            self.assertEqual(res[1], {"testing": "test", "format": "mlx"})

        for dt in self.dtypes + ["bfloat16"]:
            with self.subTest(dtype=dt):
                for i, shape in enumerate([(1,), (23,), (1024, 1024), (4, 6, 3, 1, 2)]):
                    with self.subTest(shape=shape):
                        save_file_mlx = os.path.join(
                            self.test_dir, f"mlx_{dt}_{i}_fs.safetensors"
                        )
                        save_dict = {
                            "test": (
                                mx.random.normal(shape=shape, dtype=getattr(mx, dt))
                                if dt in ["float32", "float16", "bfloat16"]
                                else mx.ones(shape, dtype=getattr(mx, dt))
                            )
                        }

                        with open(save_file_mlx, "wb") as f:
                            mx.save_safetensors(f, save_dict)
                        with open(save_file_mlx, "rb") as f:
                            load_dict = mx.load(f)

                        self.assertTrue("test" in load_dict)
                        self.assertTrue(
                            mx.array_equal(load_dict["test"], save_dict["test"])
                        )

    def test_save_and_load_gguf(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        # TODO: Add support for other dtypes (self.dtypes + ["bfloat16"])
        supported_dtypes = ["float16", "float32", "int8", "int16", "int32"]
        for dt in supported_dtypes:
            with self.subTest(dtype=dt):
                for i, shape in enumerate([(1,), (23,), (1024, 1024), (4, 6, 3, 1, 2)]):
                    with self.subTest(shape=shape):
                        save_file_mlx = os.path.join(
                            self.test_dir, f"mlx_{dt}_{i}_fs.gguf"
                        )
                        save_dict = {
                            "test": (
                                mx.random.normal(shape=shape, dtype=getattr(mx, dt))
                                if dt in ["float32", "float16", "bfloat16"]
                                else mx.ones(shape, dtype=getattr(mx, dt))
                            )
                        }

                        mx.save_gguf(save_file_mlx, save_dict)
                        load_dict = mx.load(save_file_mlx)

                        self.assertTrue("test" in load_dict)
                        self.assertTrue(
                            mx.array_equal(load_dict["test"], save_dict["test"])
                        )

        save_file_mlx = os.path.join(self.test_dir, f"mlx_path_test_fs.gguf")
        save_dict = {"test": mx.ones(shape)}
        mx.save_gguf(Path(save_file_mlx), save_dict)
        load_dict = mx.load(Path(save_file_mlx))
        self.assertTrue("test" in load_dict)
        self.assertTrue(mx.array_equal(load_dict["test"], save_dict["test"]))

    def test_load_f8_e4m3(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        expected = [
            0,
            mx.nan,
            mx.nan,
            -0.875,
            0.4375,
            -0.005859,
            -1.25,
            -1.25,
            -1.5,
            -0.0039,
        ]
        expected = mx.array(expected, dtype=mx.bfloat16)
        contents = b'H\x00\x00\x00\x00\x00\x00\x00{"tensor":{"dtype":"F8_E4M3","shape":[10],"data_offsets":[0,10]}}       \x00\x7f\xff\xb6.\x83\xba\xba\xbc\x82'
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            f.write(contents)
            f.seek(0)
            out = mx.load(f)["tensor"]
        self.assertTrue(mx.allclose(out[0], expected[0], equal_nan=True))

    def test_save_and_load_gguf_metadata_basic(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        save_file_mlx = os.path.join(self.test_dir, f"mlx_gguf_with_metadata.gguf")
        save_dict = {"test": mx.ones((4, 4), dtype=mx.int32)}
        metadata = {}

        # Empty works
        mx.save_gguf(save_file_mlx, save_dict, metadata)

        # Loads without the metadata
        load_dict = mx.load(save_file_mlx)
        self.assertTrue("test" in load_dict)
        self.assertTrue(mx.array_equal(load_dict["test"], save_dict["test"]))

        # Loads empty metadata
        load_dict, meta_load_dict = mx.load(save_file_mlx, return_metadata=True)
        self.assertTrue("test" in load_dict)
        self.assertTrue(mx.array_equal(load_dict["test"], save_dict["test"]))
        self.assertEqual(len(meta_load_dict), 0)

        # Loads string metadata
        metadata = {"meta": "data"}
        mx.save_gguf(save_file_mlx, save_dict, metadata)
        load_dict, meta_load_dict = mx.load(save_file_mlx, return_metadata=True)
        self.assertTrue("test" in load_dict)
        self.assertTrue(mx.array_equal(load_dict["test"], save_dict["test"]))
        self.assertEqual(len(meta_load_dict), 1)
        self.assertTrue("meta" in meta_load_dict)
        self.assertEqual(meta_load_dict["meta"], "data")

    def test_save_and_load_gguf_metadata_arrays(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        save_file_mlx = os.path.join(self.test_dir, f"mlx_gguf_with_metadata.gguf")
        save_dict = {"test": mx.ones((4, 4), dtype=mx.int32)}

        # Test scalars and one dimensional arrays
        for t in [
            mx.uint8,
            mx.int8,
            mx.uint16,
            mx.int16,
            mx.uint32,
            mx.int32,
            mx.uint64,
            mx.int64,
            mx.float32,
        ]:
            for shape in [(), (2,)]:
                arr = mx.random.uniform(shape=shape).astype(t)
                metadata = {"meta": arr}
                mx.save_gguf(save_file_mlx, save_dict, metadata)
                _, meta_load_dict = mx.load(save_file_mlx, return_metadata=True)
                self.assertEqual(len(meta_load_dict), 1)
                self.assertTrue("meta" in meta_load_dict)
                self.assertTrue(mx.array_equal(meta_load_dict["meta"], arr))
                self.assertEqual(meta_load_dict["meta"].dtype, arr.dtype)

        for t in [mx.float16, mx.bfloat16, mx.complex64]:
            with self.assertRaises(ValueError):
                arr = mx.array(1, t)
                metadata = {"meta": arr}
                mx.save_gguf(save_file_mlx, save_dict, metadata)

    def test_save_and_load_gguf_metadata_mixed(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        save_file_mlx = os.path.join(self.test_dir, f"mlx_gguf_with_metadata.gguf")
        save_dict = {"test": mx.ones((4, 4), dtype=mx.int32)}

        # Test string and array
        arr = mx.array(1.5)
        metadata = {"meta1": arr, "meta2": "data"}
        mx.save_gguf(save_file_mlx, save_dict, metadata)
        _, meta_load_dict = mx.load(save_file_mlx, return_metadata=True)
        self.assertEqual(len(meta_load_dict), 2)
        self.assertTrue("meta1" in meta_load_dict)
        self.assertTrue(mx.array_equal(meta_load_dict["meta1"], arr))
        self.assertEqual(meta_load_dict["meta1"].dtype, arr.dtype)
        self.assertTrue("meta2" in meta_load_dict)
        self.assertEqual(meta_load_dict["meta2"], "data")

        # Test list of strings
        metadata = {"meta": ["data1", "data2", "data345"]}
        mx.save_gguf(save_file_mlx, save_dict, metadata)
        _, meta_load_dict = mx.load(save_file_mlx, return_metadata=True)
        self.assertEqual(len(meta_load_dict), 1)
        self.assertEqual(meta_load_dict["meta"], metadata["meta"])

        # Test a combination of stuff
        metadata = {
            "meta1": ["data1", "data2", "data345"],
            "meta2": mx.array([1, 2, 3, 4]),
            "meta3": "data",
            "meta4": mx.array(1.5),
        }
        mx.save_gguf(save_file_mlx, save_dict, metadata)
        _, meta_load_dict = mx.load(save_file_mlx, return_metadata=True)
        self.assertEqual(len(meta_load_dict), 4)
        for k, v in metadata.items():
            if isinstance(v, mx.array):
                self.assertTrue(mx.array_equal(meta_load_dict[k], v))
            else:
                self.assertEqual(meta_load_dict[k], v)

    def test_save_and_load_fs(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        for dt in self.dtypes:
            with self.subTest(dtype=dt):
                for i, shape in enumerate([(1,), (23,), (1024, 1024), (4, 6, 3, 1, 2)]):
                    with self.subTest(shape=shape):
                        save_file_mlx = os.path.join(
                            self.test_dir, f"mlx_{dt}_{i}_fs.npy"
                        )
                        save_file_npy = os.path.join(
                            self.test_dir, f"npy_{dt}_{i}_fs.npy"
                        )

                        save_arr = np.random.uniform(0.0, 32.0, size=shape)
                        save_arr_npy = save_arr.astype(getattr(np, dt))
                        save_arr_mlx = mx.array(save_arr_npy)

                        with open(save_file_mlx, "wb") as f:
                            mx.save(f, save_arr_mlx)

                        np.save(save_file_npy, save_arr_npy)

                        # Load array saved by mlx as mlx array
                        with open(save_file_mlx, "rb") as f:
                            load_arr_mlx_mlx = mx.load(f)
                        self.assertTrue(mx.array_equal(load_arr_mlx_mlx, save_arr_mlx))

                        # Load array saved by numpy as mlx array
                        with open(save_file_npy, "rb") as f:
                            load_arr_npy_mlx = mx.load(f)
                        self.assertTrue(mx.array_equal(load_arr_npy_mlx, save_arr_mlx))

                        # Load array saved by mlx as numpy array
                        load_arr_mlx_npy = np.load(save_file_mlx)
                        self.assertTrue(np.array_equal(load_arr_mlx_npy, save_arr_npy))

    def test_savez_and_loadz(self):
        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

        for dt in self.dtypes:
            with self.subTest(dtype=dt):
                shapes = [(6,), (6, 6), (4, 1, 3, 1, 2)]
                save_file_mlx_uncomp = os.path.join(
                    self.test_dir, f"mlx_{dt}_uncomp.npz"
                )
                save_file_npy_uncomp = os.path.join(
                    self.test_dir, f"npy_{dt}_uncomp.npz"
                )
                save_file_mlx_comp = os.path.join(self.test_dir, f"mlx_{dt}_comp.npz")
                save_file_npy_comp = os.path.join(self.test_dir, f"npy_{dt}_comp.npz")

                # Make dictionary of multiple
                save_arrs_npy = {
                    f"save_arr_{i}": np.random.uniform(
                        0.0, 32.0, size=shapes[i]
                    ).astype(getattr(np, dt))
                    for i in range(len(shapes))
                }
                save_arrs_mlx = {k: mx.array(v) for k, v in save_arrs_npy.items()}

                # Save as npz files
                np.savez(save_file_npy_uncomp, **save_arrs_npy)
                mx.savez(save_file_mlx_uncomp, **save_arrs_mlx)
                np.savez_compressed(save_file_npy_comp, **save_arrs_npy)
                mx.savez_compressed(save_file_mlx_comp, **save_arrs_mlx)

                for save_file_npy, save_file_mlx in (
                    (save_file_npy_uncomp, save_file_mlx_uncomp),
                    (save_file_npy_comp, save_file_mlx_comp),
                ):
                    # Load array saved by mlx as mlx array
                    load_arr_mlx_mlx = mx.load(save_file_mlx)
                    for k, v in load_arr_mlx_mlx.items():
                        self.assertTrue(mx.array_equal(save_arrs_mlx[k], v))

                    # Load arrays saved by numpy as mlx arrays
                    load_arr_npy_mlx = mx.load(save_file_npy)
                    for k, v in load_arr_npy_mlx.items():
                        self.assertTrue(mx.array_equal(save_arrs_mlx[k], v))

                    # Load array saved by mlx as numpy array
                    load_arr_mlx_npy = np.load(save_file_mlx)
                    for k, v in load_arr_mlx_npy.items():
                        self.assertTrue(np.array_equal(save_arrs_npy[k], v))

    def test_non_contiguous(self):
        a = mx.broadcast_to(mx.array([1, 2]), [4, 2])

        save_file = os.path.join(self.test_dir, "a.npy")
        mx.save(save_file, a)
        aload = mx.load(save_file)
        self.assertTrue(mx.array_equal(a, aload))

        save_file = os.path.join(self.test_dir, "a.safetensors")
        mx.save_safetensors(save_file, {"a": a})
        aload = mx.load(save_file)["a"]
        self.assertTrue(mx.array_equal(a, aload))

        save_file = os.path.join(self.test_dir, "a.gguf")
        mx.save_gguf(save_file, {"a": a})
        aload = mx.load(save_file)["a"]
        self.assertTrue(mx.array_equal(a, aload))

        # safetensors and gguf only work with row contiguous
        # make sure col contiguous is handled properly
        save_file = os.path.join(self.test_dir, "a.safetensors")
        a = mx.arange(4).reshape(2, 2).T
        mx.save_safetensors(save_file, {"a": a})
        aload = mx.load(save_file)["a"]
        self.assertTrue(mx.array_equal(a, aload))

        save_file = os.path.join(self.test_dir, "a.gguf")
        mx.save_gguf(save_file, {"a": a})
        aload = mx.load(save_file)["a"]
        self.assertTrue(mx.array_equal(a, aload))

    def test_load_donation(self):
        x = mx.random.normal((1024,))
        mx.eval(x)
        save_file = os.path.join(self.test_dir, "donation.npy")
        mx.save(save_file, x)
        mx.synchronize()

        mx.reset_peak_memory()
        scale = mx.array(2.0)
        y = mx.load(save_file)
        mx.eval(y)
        mx.synchronize()
        load_only = mx.get_peak_memory()
        y = mx.load(save_file) * scale
        mx.eval(y)
        mx.synchronize()
        load_with_binary = mx.get_peak_memory()

        self.assertEqual(load_only, load_with_binary)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
