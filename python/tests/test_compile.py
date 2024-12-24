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

    def test_compile_kwargs(self):
        @mx.compile
        def fun(x, y, z):
            return x + y + z

        x = mx.array(1)
        y = mx.array(2)
        z = mx.array(3)
        out = fun(x, y=y, z=z)
        self.assertEqual(out.item(), 6)

    def test_shapeless_compile(self):
        y = 1

        @partial(mx.compile, shapeless=True)
        def fun(x):
            return x + y

        x = mx.array([1, 2])
        self.assertTrue(mx.array_equal(fun(x), mx.array([2, 3])))

        # The function is not recompiled, so the change
        # to y should not be reflected in the output
        y = 2
        x = mx.array([1, 2, 3])
        self.assertTrue(mx.array_equal(fun(x), mx.array([2, 3, 4])))

        # Type change recompiles
        x = mx.array([1.0, 2.0, 3.0])
        self.assertTrue(mx.array_equal(fun(x), mx.array([3.0, 4.0, 5.0])))

        # Dim change recompiles
        x = mx.array([[1, 2, 3]])
        self.assertTrue(mx.array_equal(fun(x), mx.array([[3, 4, 5]])))

    def test_shapeless_compile_with_broadcasts(self):
        x = mx.ones((2, 2))
        y = mx.array([2, 2])

        def fun(x, y):
            return x * y

        cfun = mx.compile(fun, shapeless=True)
        self.assertTrue(mx.array_equal(cfun(x, y), fun(x, y)))
        self.assertTrue(mx.array_equal(cfun(y, x), fun(y, x)))
        y = mx.array([[3]])
        self.assertTrue(mx.array_equal(cfun(x, y), fun(x, y)))
        self.assertTrue(mx.array_equal(cfun(y, x), fun(y, x)))

    def test_shapeless_compile_with_reduction(self):
        # Test shapeless compile with a reduction
        z = 1

        @partial(mx.compile, shapeless=True)
        def fun(x, y):
            return x + y.sum(0, keepdims=True) + z

        x = mx.ones((2, 2), mx.int32)
        y = mx.ones((2, 2), mx.int32)
        self.assertTrue(mx.array_equal(fun(x, y), mx.full(shape=(2, 2), vals=4)))
        x = mx.ones((3, 3), mx.int32)
        y = mx.ones((3, 3), mx.int32)
        z = 2
        self.assertTrue(mx.array_equal(fun(x, y), mx.full(shape=(3, 3), vals=5)))

        x1 = mx.array([[1, 2], [3, 4], [5, 6]])
        x2 = mx.array([[1, 2]])

        def fun(x):
            return x * x.sum(-1, keepdims=True)

        cfun = mx.compile(fun, shapeless=True)
        mx.eval(cfun(x1))
        self.assertTrue(mx.array_equal(fun(x2), cfun(x2)))

        def fun(x):
            return x * x.sum(-1, keepdims=False)

        cfun = mx.compile(fun, shapeless=True)
        self.assertTrue(mx.array_equal(fun(x2), cfun(x2)))

    def test_shapeless_compile_unflatten(self):
        x = mx.zeros((1, 1, 4 * 32))

        def fun(x):
            return mx.unflatten(x, -1, (4, -1))

        self.assertEqual(mx.compile(fun, shapeless=True)(x).shape, (1, 1, 4, 32))

    def test_shapeless_compile_gather(self):
        x = mx.zeros((1, 1, 32))

        def fun(x):
            return x[:, -1, :]

        self.assertEqual(mx.compile(fun, shapeless=True)(x).shape, (1, 32))

    def test_compile_with_constant(self):
        # Test float
        @partial(mx.compile)
        def fun(x, y):
            return x + y

        z = fun(mx.array(1.0), 1.0)
        self.assertEqual(z.item(), 2.0)

        z = fun(mx.array(1.0), 2.0)
        self.assertEqual(z.item(), 3.0)

        z = fun(mx.array(1.0), y=1.0)
        self.assertEqual(z.item(), 2.0)

        z = fun(mx.array(1.0), y=3.0)
        self.assertEqual(z.item(), 4.0)

        # Test tuple
        @partial(mx.compile)
        def fun(x, y=(1, 2)):
            return x + y[0] + y[1]

        z = fun(mx.array(1))
        self.assertEqual(z.item(), 4)

        z = fun(mx.array(1), (2, 2))
        self.assertEqual(z.item(), 5)

        z = fun(mx.array(1), (2, 1))
        self.assertEqual(z.item(), 4)

        # Test bool
        @partial(mx.compile)
        def fun(x, y):
            if y:
                return x + 1
            else:
                return x + 2

        z = fun(mx.array(1), True)
        self.assertEqual(z.item(), 2)

        z = fun(mx.array(1), False)
        self.assertEqual(z.item(), 3)

        # Test string
        @partial(mx.compile)
        def fun(x, y):
            if y == "one":
                return x + 1
            else:
                return x + 2

        z = fun(mx.array(1), "one")
        self.assertEqual(z.item(), 2)

        z = fun(mx.array(1), "two")
        self.assertEqual(z.item(), 3)

        # Test nested constant
        @partial(mx.compile)
        def fun(x, y):
            if y[0][0] == 1:
                return x + 1
            else:
                return x + 2

        z = fun(mx.array(1), [[1]])
        self.assertEqual(z.item(), 2)

        z = fun(mx.array(1), [[0]])
        self.assertEqual(z.item(), 3)

        @partial(mx.compile)
        def fun(x, a, b):
            for ai in a:
                for bi in b:
                    x = bi * x + ai
            return x

        z = fun(mx.array(1), [1, 1], [2])
        self.assertEqual(z.item(), 7)

        z = fun(mx.array(1), [1], [1, 2])
        self.assertEqual(z.item(), 5)

        counter = [0]

        @partial(mx.compile)
        def fun(x, y):
            counter[0] += 1
            return x + y

        z = fun(mx.array(1), 1)
        self.assertEqual(z.item(), 2)

        z = fun(1, mx.array(1))
        self.assertEqual(z.item(), 2)

        self.assertEqual(counter[0], 2)

        y = 1.0

        @mx.compile
        def fun(x, constant):
            return x + y

        constant1 = "abc"
        out = fun(mx.array(0.0), constant1)
        self.assertEqual(out, mx.array(1.0))

        # new object, same value, no recompilation
        y = 2.0
        constant2 = "abc".encode("utf-8").decode("utf-8")
        out = fun(mx.array(0.0), constant2)
        self.assertEqual(out, mx.array(1.0))

        # same object, new value, recompilation
        constant2 = "xyz"
        out = fun(mx.array(0.0), constant2)
        self.assertEqual(out, mx.array(2.0))

    def test_compile_inf(self):
        @mx.compile
        def fun(x):
            return mx.isinf(x + 2)

        out = fun(mx.array([0.0]))
        self.assertEqual(out.item(), False)

    def test_unsupported_input_types(self):
        class MyClass:
            value = 1

        @mx.compile
        def fun(x, y):
            return x + y.value

        with self.assertRaises(ValueError):
            out = fun(mx.array(0.0), MyClass())

        with self.assertRaises(ValueError):
            out = fun(mx.array(0.0), y=MyClass())

    def test_compile_create_list(self):
        @mx.compile
        def fun():
            return [0.1 * mx.zeros((2,)), 0.1 * mx.zeros((2,))]

        out = fun()
        mx.eval(out)

    def test_compile_vjp(self):
        def fun(w):
            w1 = w + w
            w2 = w + w
            return w @ w1 + w2 @ w2

        def step(w):
            out, grad = mx.vjp(fun, (w,), (mx.array([[1.0, 1.0], [1.0, 1.0]]),))
            return out[0], grad[0]

        w = mx.zeros((2, 2))
        mx.eval(w)

        expected = step(w)
        out = mx.compile(step)(w)
        self.assertTrue(mx.allclose(expected[0], out[0]))
        self.assertTrue(mx.allclose(expected[1], out[1]))

        def fun(w1, w2, x):
            x = x @ w1
            y = x @ w2
            x = x + y * y
            return (x * x).sum()

        w1 = mx.zeros((4, 4))
        w2 = mx.zeros((4, 4))
        x = mx.zeros((4, 4))

        def step(w1, w2, x):
            loss, gradient = mx.value_and_grad(fun)(w1, w2, x)
            w1 = w1 + gradient
            return loss, w1

        mx.eval(x, w1, w2)
        expected = step(w1, w2, x)
        out = mx.compile(step)(w1, w2, x)

        self.assertTrue(mx.allclose(expected[0], out[0]))
        self.assertTrue(mx.allclose(expected[1], out[1]))

    def test_shapeless_mean(self):
        def mean(x):
            return mx.mean(x, keepdims=True)

        cmean = mx.compile(mean, shapeless=True)

        x = mx.ones(2)
        out = cmean(x)
        self.assertTrue(mx.allclose(out, mean(x)))

        x = mx.ones(4)
        out = cmean(x)
        self.assertTrue(mx.allclose(out, mean(x)))

        x = mx.ones(7)
        out = cmean(x)
        self.assertTrue(mx.allclose(out, mean(x)))

    def test_compile_broadcast_only(self):
        def fn(a):
            a = mx.broadcast_to(a, (1,))
            return a + a

        out = mx.compile(fn)(mx.array(2.0))
        # Make sure repr can be called
        self.assertTrue(repr(out) is not None)
        self.assertTrue(mx.array_equal(out, mx.array([4.0])))

    def test_compile_with_long_name(self):
        def fn(a, b):
            for _ in range(10):
                a = a - 1.0
                b = b - 1.0
            return a + b

        out = mx.compile(fn)(mx.array(10.0), mx.array(20.0))
        self.assertEqual(out.item(), 10.0)

    def test_compile_multi_output(self):
        def fn(x):
            ys = [x]
            for i in range(5):
                ys.append(ys[-1] + x)
            return ys, mx.sum(ys[-1])

        x = mx.ones(1, dtype=mx.int32)
        y1 = mx.compile(fn)(x)[1]
        y2 = fn(x)[1]
        self.assertEqual(y1.item(), y2.item())
        self.assertEqual(y1.item(), 6)

    def test_inf_constant(self):
        def fn(x):
            return mx.where(mx.isinf(x), 0, 1)

        x = mx.array([0, float("inf"), 1], dtype=mx.bfloat16)
        self.assertTrue(mx.array_equal(mx.compile(fn)(x), fn(x)))

    def test_max_into_equal(self):
        x = mx.random.uniform(shape=(1, 2, 2))
        mx.eval(x)

        def fn():
            maxes = mx.max(x, axis=(1, 2), keepdims=True)
            return x == maxes

        out = mx.compile(fn)()
        expected = fn()
        self.assertTrue(mx.array_equal(expected, out))

    def test_dtypes(self):
        x = mx.array([0, 1, 2, 3])
        dtypes = [mx.bool_, mx.int8, mx.uint8, mx.int16, mx.uint16]
        for dtype in dtypes:
            x = x.astype(dtype)
            mx.eval(x)

            def fn(x):
                return x * 1 + 0

            out = mx.compile(fn)(x)
            expected = fn(x)
            self.assertTrue(mx.array_equal(expected, out))

    def test_compile_without_captured_inputs(self):
        x = mx.array([1, 2, 3]) + 2

        def fn(a):
            y = x + 1
            return a + y

        with self.assertRaises(ValueError):
            y = mx.compile(fn)(x)

        x = mx.array([1.0, 2.0]) + mx.array([1.0, 2.0])
        y = None

        def fn(x):
            nonlocal y
            if y is None:
                y = mx.array([1.0, 2.0])

            y = y + x
            return y

        fn(x)
        with self.assertRaises(ValueError):
            y = mx.compile(fn)(x)

    def test_compile_dynamic_dims(self):
        a = mx.random.uniform(shape=(2,) * 10)
        b = mx.random.uniform(shape=(2,) * 10)
        a = a.T
        mx.eval(a, b)

        def fn(a, b):
            return mx.abs(a + b)

        out = mx.compile(fn)(a, b)
        expected = fn(a, b)
        self.assertTrue(mx.allclose(out, expected))

    def test_compile_many_inputs(self):
        inputs = [mx.ones((2, 2, 2, 2)) for _ in range(20)]
        inputs[0] = inputs[0].T

        @mx.compile
        def fun(*inputs):
            x = inputs[0]
            for y in inputs[1:10]:
                x = x + y
            a = inputs[10]
            for b in inputs[11:]:
                a = a + b
            return x + a

        out = fun(*inputs)
        self.assertTrue(mx.allclose(out, mx.full((2, 2), 20)))

    def test_shapeless_compile_matmul(self):
        a = mx.array([0.0, 1.0, 2.0])
        b = mx.array([0.0, 1.0, 2.0])

        fun = mx.compile(lambda a, b: a @ b, shapeless=True)
        self.assertTrue(mx.allclose(fun(a, b), a @ b))

    def test_shapeless_compile_slice_update(self):
        def fun(x):
            x[2] = mx.array([3.0])
            return x

        cfun = mx.compile(fun, shapeless=True)

        a = mx.array([0.0, 1.0, 2.0, 3.0])
        self.assertTrue(mx.allclose(cfun(a), fun(a)))

        a = mx.array([0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertTrue(mx.allclose(cfun(a), fun(a)))

    def test_compile_shapeless_with_broadcast(self):
        a = mx.array(0.0)
        b = mx.ones((2, 2))

        def fun(a):
            return mx.broadcast_to(a, b.shape)

        cfun = mx.compile(fun, shapeless=True)
        # Works on the first shape
        cfun(a)

        # Fails on a different shape
        with self.assertRaises(ValueError):
            cfun(mx.array(0.0).reshape(1, 1, 1))

        def fun(a, b):
            return mx.broadcast_arrays(a, b)

        cfun = mx.compile(fun, shapeless=True)
        a, b = cfun(a, b)
        self.assertEqual(a.shape, (2, 2))
        self.assertEqual(b.shape, (2, 2))

        # Batched matmul
        a = mx.zeros((2, 1, 4, 2))
        b = mx.zeros((3, 2, 5))

        def fun(a, b):
            return a @ b

        cfun = mx.compile(fun, shapeless=True)
        out = cfun(a, b)
        self.assertEqual(out.shape, (2, 3, 4, 5))


if __name__ == "__main__":
    unittest.main()
