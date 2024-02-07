# Copyright © 2023-2024 Apple Inc.

import io
import unittest
from functools import partial

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

    def test_compile_two_input_grad(self):
        def loss(w, x):
            y = x * w
            return (y * mx.exp(y)).sum()

        x = mx.array([1.0, 0.5, 2.0, -0.5])
        w = mx.array([-1.0, 0.3, 1.0, -0.9])

        expected_grad = mx.grad(loss)(w, x)
        compiled_grad = mx.compile(mx.grad(loss))(w, x)
        self.assertTrue(mx.allclose(expected_grad, compiled_grad))

    def test_vmap_compiled(self):
        def simple_unary(x):
            return -mx.exp(x)

        x = mx.array([[1.0, 2.0], [2.0, 3.0]])

        expected_out = mx.vmap(simple_unary)(x)
        out = mx.vmap(mx.compile(simple_unary))(x)
        self.assertTrue(mx.allclose(expected_out, out))

        def simple_binary(x, y):
            return mx.abs(mx.exp(x + y) + y)

        x = mx.array([[1.0, -3.0], [0.5, -0.5]])
        y = mx.array([[2.0, -1.0], [0.25, -0.25]])

        expected_out = mx.vmap(simple_binary)(x, y)
        out = mx.vmap(mx.compile(simple_binary))(x, y)
        self.assertTrue(mx.allclose(expected_out, out))

        expected_out = mx.vmap(simple_binary, in_axes=(0, 1))(x, y)
        out = mx.vmap(mx.compile(simple_binary), in_axes=(0, 1))(x, y)
        self.assertTrue(mx.allclose(expected_out, out))

        y = mx.array([0.25, -0.25])
        expected_out = mx.vmap(simple_binary, in_axes=(0, None))(x, y)
        out = mx.vmap(mx.compile(simple_binary), in_axes=(0, None))(x, y)
        self.assertTrue(mx.allclose(expected_out, out))

        def simple_unary_outer(x):
            x = mx.abs(x)

            @mx.compile
            def simple_unary_inner(z):
                return -mx.exp(x)

            return simple_unary_inner(x)

        expected_out = -mx.exp(mx.abs(x))
        out = mx.vmap(simple_unary_outer)(x)
        self.assertTrue(mx.allclose(expected_out, out))

    def test_vjp_vjp_compiled(self):
        def simple_unary(x):
            return -mx.exp(x)

        x = mx.array([[1.0, 2.0], [2.0, 3.0]])
        y = mx.array([[1.0, 1.0], [1.0, 1.0]])

        expected_out, expected_vjp_out = mx.vjp(simple_unary, (x,), (y,))
        out, vjp_out = mx.vjp(mx.compile(simple_unary), (x,), (y,))
        self.assertTrue(mx.allclose(expected_vjp_out[0], vjp_out[0]))
        self.assertTrue(mx.allclose(expected_out[0], out[0]))

        expected_out, expected_jvp_out = mx.jvp(simple_unary, (x,), (y,))
        out, jvp_out = mx.jvp(mx.compile(simple_unary), (x,), (y,))
        self.assertTrue(mx.allclose(expected_jvp_out[0], jvp_out[0]))
        self.assertTrue(mx.allclose(expected_out[0], out[0]))

        def simple_binary(x, y):
            return mx.abs(mx.exp(x + y) + y)

        x = mx.array([[1.0, -3.0], [0.5, -0.5]])
        y = mx.array([[2.0, -1.0], [0.25, -0.25]])
        cotans = mx.ones_like(x)

        expected_out, expected_vjp_out = mx.vjp(simple_binary, (x, y), (cotans,))
        out, vjp_out = mx.vjp(mx.compile(simple_binary), (x, y), (cotans,))
        self.assertTrue(mx.allclose(expected_out[0], out[0]))
        self.assertTrue(mx.allclose(expected_vjp_out[0], vjp_out[0]))
        self.assertTrue(mx.allclose(expected_vjp_out[1], vjp_out[1]))

        tans = (mx.ones_like(x), mx.ones_like(y))
        expected_out, expected_jvp_out = mx.jvp(simple_binary, (x, y), tans)
        out, jvp_out = mx.jvp(mx.compile(simple_binary), (x, y), tans)
        self.assertTrue(mx.allclose(expected_jvp_out[0], jvp_out[0]))
        self.assertTrue(mx.allclose(expected_out[0], out[0]))

    def test_transform_over_eval_compiled(self):
        def outer(x):
            y = mx.exp(mx.abs(x))
            mx.eval(y)
            return y.sum()

        x = mx.array([2.0, -1.0, 0.5])
        dfdx = mx.grad(outer)(x)

        @mx.compile
        def simple_unary(x):
            return mx.exp(mx.abs(x))

        def outer(x):
            y = simple_unary(x)
            mx.eval(y)
            return y.sum()

        cdfdx = mx.grad(outer)(x)
        self.assertTrue(mx.allclose(dfdx, cdfdx))

    def test_compile_capture(self):
        # Test update captured state outside compiled function
        state = {"y": mx.array(2)}

        @partial(mx.compile, inputs=state)
        def test_state(x):
            x = x + state["y"]
            return x

        test_state(mx.array(1))
        # Check the state is unchanged
        self.assertEqual(state["y"], 2)

        # Check the udpated state is used
        state["y"] = mx.array(3)
        out = test_state(mx.array(1))
        self.assertEqual(out.item(), 4)

        # Capture list
        state = [mx.array(2)]

        @partial(mx.compile, inputs=state)
        def test_state(x):
            x = x + state[0]
            return x

        out = test_state(mx.array(1))
        self.assertEqual(out.item(), 3)
        state[0] = mx.array(3)
        out = test_state(mx.array(1))
        self.assertEqual(out.item(), 4)

        # Capture tuple of list
        state = ([mx.array(2)],)

        @partial(mx.compile, inputs=state)
        def test_state(x):
            x = x + state[0][0]
            return x

        out = test_state(mx.array(1))
        self.assertEqual(out.item(), 3)
        state[0][0] = mx.array(3)
        out = test_state(mx.array(1))
        self.assertEqual(out.item(), 4)

        # Test state updated inside compiled function
        state = {}

        @partial(mx.compile, outputs=state)
        def test_state(x):
            state["y"] = x + 3
            return mx.abs(x)

        test_state(mx.array(-1))
        self.assertEqual(state["y"].item(), 2)

        # Test state changed inside compiled function
        # triggers recompile
        state = {}

        @partial(mx.compile, inputs=state, outputs=state)
        def test_state(x):
            y = state.get("y", mx.array(0))
            state["y"] = x + y
            return x + 2 * y

        test_state(mx.array(1))
        self.assertEqual(state["y"].item(), 1)
        test_state(mx.array(1))
        self.assertEqual(state["y"].item(), 2)

    def test_compile_rng(self):
        @partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
        def fun():
            return mx.random.uniform(shape=(10, 10))

        self.assertFalse(mx.allclose(fun(), fun(), 1e-2, 1e-2))


if __name__ == "__main__":
    unittest.main()
