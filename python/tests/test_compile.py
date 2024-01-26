# Copyright © 2023-2024 Apple Inc.

import io
import unittest

import mlx.core as mx
import mlx_tests


class TestCompile(mlx_tests.MLXTestCase):
    def test_simple_compile(self):
        def fun(x, y):
            return x + y

        compiled_fn = mx.compile(fun)
        compiled_fn = mx.compile(fun)
        x = mx.array(1.0)
        y = mx.array(1.0)
        out = compiled_fn(x, y)
        self.assertEqual(out.item(), 2.0)

        # Try again
        out = compiled_fn(x, y)
        self.assertEqual(out.item(), 2.0)

        # Change sizes
        x = mx.array([1.0, 2.0])
        out = compiled_fn(x, y)
        self.assertTrue(mx.array_equal(out, mx.array([2.0, 3.0])))

        y = mx.array([1.0, 2.0])
        out = compiled_fn(x, y)
        self.assertTrue(mx.array_equal(out, mx.array([2.0, 4.0])))

        # Change types
        x = mx.array([1, 2], mx.int32)
        y = mx.array([1, 2], mx.int32)
        out = compiled_fn(x, y)
        self.assertEqual(out.dtype, mx.int32)
        self.assertTrue(mx.array_equal(out, mx.array([2, 4])))

    def test_compile_grad(self):
        def loss_fn(x):
            return mx.exp(x).sum()

        grad_fn = mx.grad(loss_fn)

        x = mx.array([0.5, -0.5, 1.2])
        dfdx = grad_fn(x)
        compile_grad_fn = mx.compile(grad_fn)
        c_dfdx = grad_fn(x)

        self.assertTrue(mx.allclose(c_dfdx, dfdx))

        # Run it again without calling compile
        c_dfdx = compile_grad_fn(x)
        self.assertTrue(mx.allclose(c_dfdx, dfdx))

        # Run it again with calling compile
        c_dfdx = mx.compile(grad_fn)(x)
        self.assertTrue(mx.allclose(c_dfdx, dfdx))

        # Value and grad
        def loss_fn(x):
            return mx.exp(x).sum(), mx.sin(x)

        val_and_grad_fn = mx.value_and_grad(loss_fn)
        (loss, val), dfdx = val_and_grad_fn(x)
        (c_loss, c_val), c_dfdx = mx.compile(val_and_grad_fn)(x)

        self.assertTrue(mx.allclose(c_dfdx, dfdx))
        self.assertTrue(mx.allclose(c_loss, loss))
        self.assertTrue(mx.allclose(c_val, val))

    def test_compile_inputs_with_primitives(self):
        x = mx.array([1, 2, 3])
        y = mx.array([1, 2, 3])
        for _ in range(5):
            x = x + y
            y = y + 1

        def fun(x, y):
            return x * y

        out = fun(x, y)

        x = mx.array([1, 2, 3])
        y = mx.array([1, 2, 3])
        for _ in range(5):
            x = x + y
            y = y + 1

        c_out = mx.compile(fun)(x, y)
        self.assertTrue(mx.array_equal(out, c_out))

        # Try again
        c_out = mx.compile(fun)(x, y)
        self.assertTrue(mx.array_equal(out, c_out))

    def test_compile_with_closure(self):
        x = mx.array(1)

        def closure(y):
            return x + y

        compiled = mx.compile(closure)
        out = compiled(mx.array(1))
        self.assertEqual(out.item(), 2)

        # Try again
        out = compiled(mx.array(1))
        self.assertEqual(out.item(), 2)

        # Change the shape of the enclosed variable
        x = mx.array([1, 2])
        out = compiled(mx.array(1))

        # We still get the original input (closures are not updated)
        self.assertEqual(out.item(), 2)

        # Try with a tree of enclosed variables
        x = {"a": mx.array(1), "b": mx.array(2)}

        def closure(y):
            return x["a"] + y + x["b"]

        compiled = mx.compile(closure)
        out = compiled(mx.array(1))
        self.assertEqual(out.item(), 4)

        # Change the shape of one input
        x["a"] = mx.array([4, 5])
        out = compiled(mx.array(1))
        self.assertEqual(out.item(), 4)

        x["b"] = mx.array([-6, -8])
        out = compiled(mx.array(1))
        self.assertEqual(out.item(), 4)

        # Enclosed variable is not evaluated yet
        x = mx.array(1)
        x = x + x

        def closure(y):
            return x + y

        compiled = mx.compile(closure)
        out = compiled(mx.array(2))
        self.assertEqual(out.item(), 4)

        # And again
        out = compiled(mx.array(2))
        self.assertEqual(out.item(), 4)

    def test_function_creates_array(self):
        def fun(x):
            return x + mx.array(1)

        cfun = mx.compile(fun)
        out = cfun(mx.array(3))
        self.assertEqual(out.item(), 4)

        # And again
        out = cfun(mx.array(3))
        self.assertEqual(out.item(), 4)

    def test_enable_disable(self):
        def fun(x):
            y = x + 1
            z = x + 1
            return y + z

        def count_prims(outputs):
            buf = io.StringIO()
            mx.export_to_dot(buf, outputs)
            buf.seek(0)
            return len([l for l in buf.read().split() if "label" in l])

        x = mx.array(1.0)
        cfun = mx.compile(fun)
        n_compiled = count_prims(cfun(x))

        # Check disabled
        mx.disable_compile()
        n_uncompiled = count_prims(cfun(x))
        self.assertTrue(n_compiled < n_uncompiled)

        # Check renabled
        mx.enable_compile()
        n_enable_compiled = count_prims(cfun(x))
        self.assertEqual(n_compiled, n_enable_compiled)


if __name__ == "__main__":
    unittest.main()
