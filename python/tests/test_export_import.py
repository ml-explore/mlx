# Copyright © 2024 Apple Inc.

import gc
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

    def test_basic_export_import(self):
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
            x = mx.abs(mx.sin(x))
            return x

        mx.export_function(path, fun, [inputs])
        imported = mx.import_function(path)

        expected = fun(inputs)
        (out,) = imported([inputs])
        self.assertTrue(mx.allclose(out, expected))

        (out,) = imported(inputs)
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

    def test_export_with_kwargs(self):
        path = os.path.join(self.test_dir, "fn.mlxfn")

        def fun(x, z=None):
            out = x
            if z is not None:
                out += z
            return out

        x = mx.array([1, 2, 3])
        y = mx.array([1, 1, 0])
        z = mx.array([2, 2, 2])

        mx.export_function(path, fun, (x,), {"z": z})
        imported_fun = mx.import_function(path)

        with self.assertRaises(ValueError):
            imported_fun(x, z)

        with self.assertRaises(ValueError):
            imported_fun(x, y=z)

        with self.assertRaises(ValueError):
            imported_fun((x,), {"y": z})

        out = imported_fun(x, z=z)[0]
        self.assertTrue(mx.array_equal(out, mx.array([3, 4, 5])))

        out = imported_fun((x,), {"z": z})[0]
        self.assertTrue(mx.array_equal(out, mx.array([3, 4, 5])))

        mx.export_function(path, fun, x, z=z)
        imported_fun = mx.import_function(path)
        out = imported_fun(x, z=z)[0]
        self.assertTrue(mx.array_equal(out, mx.array([3, 4, 5])))

        out = imported_fun((x,), {"z": z})[0]
        self.assertTrue(mx.array_equal(out, mx.array([3, 4, 5])))

        # Only specify kwargs
        mx.export_function(path, fun, x=x, z=z)
        imported_fun = mx.import_function(path)
        with self.assertRaises(ValueError):
            out = imported_fun(x, z=z)[0]

        out = imported_fun(x=x, z=z)[0]
        self.assertTrue(mx.array_equal(out, mx.array([3, 4, 5])))

        out = imported_fun({"x": x, "z": z})[0]
        self.assertTrue(mx.array_equal(out, mx.array([3, 4, 5])))

    def test_export_variable_inputs(self):
        path = os.path.join(self.test_dir, "fn.mlxfn")

        def fun(x, y, z=None):
            out = x + y
            if z is not None:
                out += z
            return out

        with mx.exporter(path, fun) as exporter:
            exporter(mx.array([1, 2, 3]), mx.array([1, 1, 1]))
            exporter(mx.array([1, 2, 3]), mx.array([1, 1, 1]), z=mx.array([2]))

        with self.assertRaises(RuntimeError):
            exporter(mx.array([1, 2, 3, 4]), mx.array([1, 1, 1, 1]))

        imported_fun = mx.import_function(path)
        out = imported_fun(mx.array([1, 2, 3]), mx.array([1, 1, 1]))[0]
        self.assertTrue(mx.array_equal(out, mx.array([2, 3, 4])))

        out = imported_fun(mx.array([1, 2, 3]), mx.array([1, 1, 1]), z=mx.array([2]))[0]
        self.assertTrue(mx.array_equal(out, mx.array([4, 5, 6])))

        with self.assertRaises(ValueError):
            imported_fun(mx.array([1, 2, 3, 4]), mx.array([1, 1, 1, 1]))

        # A function with a large constant
        constant = mx.zeros((16, 2048))
        mx.eval(constant)

        def fun(*args):
            return constant + sum(args)

        with mx.exporter(path, fun) as exporter:
            for i in range(5):
                exporter(*[mx.array(1)] * i)

        # Check the exported file size < constant size + small amount
        constants_size = constant.nbytes + 8192
        self.assertTrue(os.path.getsize(path) < constants_size)

    def test_leaks(self):
        path = os.path.join(self.test_dir, "fn.mlxfn")
        if mx.metal.is_available():
            mem_pre = mx.metal.get_active_memory()
        else:
            mem_pre = 0

        def outer():
            d = {}

            def f(x):
                return d["x"]

            d["f"] = mx.exporter(path, f)
            d["x"] = mx.array([0] * 1000)

        for _ in range(5):
            outer()
            gc.collect()

        if mx.metal.is_available():
            mem_post = mx.metal.get_active_memory()
        else:
            mem_post = 0

        self.assertEqual(mem_pre, mem_post)


if __name__ == "__main__":
    unittest.main()
