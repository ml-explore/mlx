# Copyright © 2023 Apple Inc.

import os
import tempfile
import unittest

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

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_save_and_load(self):

        if not os.path.isdir(self.test_dir):
            os.mkdir(self.test_dir)

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


if __name__ == "__main__":
    unittest.main()
