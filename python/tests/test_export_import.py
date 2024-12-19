# Copyright © 2024 Apple Inc.

import os
import tempfile
import unittest

import mlx.core as mx
import mlx_tests


class TestExportImport(mlx_tests.MLXTestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir_fid = tempfile.TemporaryDirectory()
        cls.test_dir = cls.test_dir_fid.name
        if not os.path.isdir(cls.test_dir):
            os.mkdir(cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        cls.test_dir_fid.cleanup()

    def test_basic_import_export(self):
        path = os.path.join(self.test_dir, "fn.mlxfn")

        # Function with no inputs
        def fun():
            return mx.zeros((3, 3))

        mx.export_function(path, fun)
        imported = mx.import_function(path)

        expected = fun()
        (out,) = imported()
        self.assertTrue(mx.array_equal(out, expected))

        # Simple function with inputs
        def fun(x):
            return mx.abs(mx.sin(x))

        inputs = mx.array([1.0, 2.0, 3.0, 4.0, 5.0])

        mx.export_function(path, fun, inputs)
        imported = mx.import_function(path)

        expected = fun(inputs)
        (out,) = imported(inputs)
        self.assertTrue(mx.allclose(out, expected))

        # Inputs in a list or tuple
        def fun(x):
            return mx.abs(mx.sin(x[0]))

        mx.export_function(path, fun, [inputs])
        imported = mx.import_function(path)

        expected = fun([inputs])
        (out,) = imported([inputs])
        self.assertTrue(mx.allclose(out, expected))

        mx.export_function(path, fun, (inputs,))
        imported = mx.import_function(path)
        (out,) = imported((inputs,))
        self.assertTrue(mx.allclose(out, expected))

        # Outputs in a list
        def fun(x):
            return [mx.abs(mx.sin(x))]

        mx.export_function(path, fun, inputs)
        imported = mx.import_function(path)
        (out,) = imported(inputs)
        self.assertTrue(mx.allclose(out, expected))

        # Outputs in a tuple
        def fun(x):
            return (mx.abs(mx.sin(x)),)

        mx.export_function(path, fun, inputs)
        imported = mx.import_function(path)
        (out,) = imported(inputs)
        self.assertTrue(mx.allclose(out, expected))

        # Check throws on invalid inputs / outputs
        def fun(x):
            return mx.abs(x)

        with self.assertRaises(ValueError):
            mx.export_function(path, fun, "hi")

        with self.assertRaises(ValueError):
            mx.export_function(path, fun, mx.array(1.0), "hi")

        def fun(x):
            return mx.abs(x["hi"])

        with self.assertRaises(ValueError):
            mx.export_function(path, fun, {"hi": mx.array(1.0)})

        def fun(x):
            return mx.abs(x[0][0])

        with self.assertRaises(ValueError):
            mx.export_function(path, fun, [[mx.array(1.0)]])

        def fun():
            return (mx.zeros((3, 3)), 1)

        with self.assertRaises(ValueError):
            mx.export_function(path, fun)

        def fun():
            return (mx.zeros((3, 3)), [mx.zeros((3, 3))])

        with self.assertRaises(ValueError):
            mx.export_function(path, fun)

        def fun(x, y):
            return x + y

        mx.export_function(path, fun, mx.array(1.0), mx.array(1.0))
        imported = mx.import_function(path)

        with self.assertRaises(ValueError):
            imported(mx.array(1.0), 1.0)

        with self.assertRaises(ValueError):
            imported(mx.array(1.0), mx.array(1.0), mx.array(1.0))

        with self.assertRaises(ValueError):
            imported(mx.array(1.0), [mx.array(1.0)])

    def test_export_random_sample(self):
        path = os.path.join(self.test_dir, "fn.mlxfn")

        mx.random.seed(5)

        def fun():
            return mx.random.uniform(shape=(3,))

        mx.export_function(path, fun)
        imported = mx.import_function(path)

        (out,) = imported()

        mx.random.seed(5)
        expected = fun()

        self.assertTrue(mx.array_equal(out, expected))


if __name__ == "__main__":
    unittest.main()
