# Copyright © 2023 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests


class TestAutograd(mlx_tests.MLXTestCase):
    def test_jvp(self):
        fun = lambda x: 2 * x
        out, dout = mx.jvp(fun, [mx.array(1.0)], [mx.array(2.0)])
        self.assertEqual(out[0].item(), 2.0)
        self.assertEqual(dout[0].item(), 4.0)

        fun = lambda x, y: x * y
        _, out = mx.jvp(
            fun, [mx.array(4.0), mx.array(2.0)], [mx.array(3.0), mx.array(2.0)]
        )
        self.assertEqual(out[0].item(), 4.0 * 2.0 + 2.0 * 3.0)

        fun = lambda x, y, z: (x * y, y * z)
        _, out = mx.jvp(
            fun,
            [mx.array(2.0), mx.array(4.0), mx.array(6.0)],
            [mx.array(1.0), mx.array(3.0), mx.array(1.0)],
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].item(), 4.0 * 1.0 + 2.0 * 3.0)
        self.assertEqual(out[1].item(), 4.0 * 1.0 + 6.0 * 3.0)

    def test_vjp(self):
        fun = lambda x: 2 * x
        out, dout = mx.vjp(fun, [mx.array(1.0)], [mx.array(2.0)])
        self.assertEqual(out[0].item(), 2.0)
        self.assertEqual(dout[0].item(), 4.0)

        fun = lambda x, y: x * y
        _, dout = mx.vjp(fun, [mx.array(4.0), mx.array(2.0)], [mx.array(3.0)])
        self.assertEqual(dout[0].item(), 6.0)
        self.assertEqual(dout[1].item(), 12.0)

        fun = lambda x, y, z: (x * y, y * z)
        _, out = mx.vjp(
            fun,
            [mx.array(2.0), mx.array(4.0), mx.array(6.0)],
            [mx.array(1.0), mx.array(3.0)],
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0].item(), 4.0 * 1.0)
        self.assertEqual(out[1].item(), 2.0 * 1.0 + 6.0 * 3.0)
        self.assertEqual(out[2].item(), 4.0 * 3.0)

    def test_grad(self):
        fun = lambda x: x * x

        value, dfdx = mx.value_and_grad(fun)(mx.array(0.5))
        self.assertEqual(value.item(), 0.25)
        self.assertEqual(dfdx.item(), 1.0)

        dfdx = mx.grad(fun)(mx.array(0.5))
        self.assertEqual(dfdx.item(), 1.0)

        df2dx2 = mx.grad(mx.grad(fun))(mx.array(0.5))
        self.assertEqual(df2dx2.item(), 2.0)
        df3dx3 = mx.grad(mx.grad(mx.grad(fun)))(mx.array(0.5))
        self.assertEqual(df3dx3.item(), 0.0)

        fun = lambda x, y: x * y
        x = mx.array(2.0)
        y = mx.array(3.0)
        dfdx = mx.grad(fun, argnums=0)(x, y)
        self.assertEqual(dfdx.item(), 3.0)
        dfdx = mx.grad(fun, argnums=1)(x, y)
        self.assertEqual(dfdx.item(), 2.0)

        # Pass non array args to functions works
        fun = lambda x, y: x
        value, dfdx = mx.value_and_grad(fun)(mx.array(2.0), "hello")
        self.assertEqual(value.item(), 2.0)
        self.assertEqual(dfdx.item(), 1.0)

        dfdx = mx.grad(fun)(mx.array(2.0), "hello")
        self.assertEqual(dfdx.item(), 1.0)

        # Raises when function does not return array
        fun = lambda x: "hello"
        with self.assertRaises(ValueError):
            mx.grad(fun)(mx.array(2.0))

        # Raises for invalid argument number or argument type
        fun = lambda x: x
        with self.assertRaises(ValueError):
            mx.grad(fun, argnums=2)(mx.array(2.0))
        with self.assertRaises(ValueError):
            mx.grad(fun, argnums=-2)(mx.array(2.0))
        with self.assertRaises(ValueError):
            mx.grad(fun)("hello")

        # Raises when output is not a scalar array
        fun = lambda x: mx.sum(x, keepdims=True)
        with self.assertRaises(ValueError):
            mx.grad(fun)(mx.ones((2, 2)))

    def test_grad_trees(self):
        fun = lambda x, y: x * y
        value, dfdx = mx.value_and_grad(fun, (0, 1))(mx.array(0.5), mx.array(2.0))
        self.assertEqual(value.item(), 1.0)
        self.assertTrue(isinstance(dfdx, tuple))
        self.assertEqual(dfdx[0].item(), 2.0)
        self.assertEqual(dfdx[1].item(), 0.5)

        fun = lambda x, y: x * y
        value, dfdx = mx.value_and_grad(fun, 1)(mx.array(0.5), mx.array(2.0))
        self.assertEqual(value.item(), 1.0)
        self.assertEqual(dfdx.item(), 0.5)

        fun = lambda p: p["x"] * p["y"]
        value, dfdx = mx.value_and_grad(fun)({"x": mx.array(0.5), "y": mx.array(2.0)})
        self.assertEqual(value.item(), 1.0)
        self.assertEqual(dfdx["x"].item(), 2.0)
        self.assertEqual(dfdx["y"].item(), 0.5)

        fun = lambda p: p["x"] * p["y"]
        with self.assertRaises(ValueError):
            mx.value_and_grad(fun)({"x": 0.5, "y": mx.array(2.0)})
        with self.assertRaises(ValueError):
            mx.value_and_grad(fun, (0, 1))({"x": mx.array(0.5), "y": mx.array(2.0)})

        fun = lambda p, b: mx.square(p[0]["foo"][2]) * b
        value, dfdx = mx.value_and_grad(fun)(
            [{"foo": [[], [], mx.array(2.0)]}], mx.array(0.5)
        )
        self.assertEqual(value.item(), 2.0)
        self.assertEqual(dfdx[0]["foo"][2].item(), 2.0)

        fun = lambda x: x
        with self.assertRaises(TypeError):
            mx.value_and_grad(fun, (None, None))
        with self.assertRaises(ValueError):
            mx.value_and_grad(fun, tuple())

    def test_auxiliary_values(self):
        def fun(x, y):
            l = (x * y).sum()
            extra = {"loss": l, "foo": y.square() + x.square(), "bar": [1, 2, 3, y, x]}
            return l, extra

        fun_value_grad = mx.value_and_grad(fun)
        fun_grad = mx.grad(fun)

        (loss, a), b = fun_value_grad(mx.ones((2, 2)), mx.ones((2, 2)))
        self.assertEqual(a["loss"].item(), 4)
        self.assertTrue(mx.array_equal(b, mx.ones((2, 2))))
        self.assertTrue(mx.array_equal(a["foo"], 2 * mx.ones((2, 2))))
        self.assertEqual(a["bar"][:3], [1, 2, 3])
        self.assertTrue(mx.array_equal(a["bar"][3], mx.ones((2, 2))))
        self.assertTrue(mx.array_equal(a["bar"][4], mx.ones((2, 2))))

        with self.assertRaises(ValueError):
            _ = fun_grad(mx.ones((2, 2)), mx.ones((2, 2)))

    def test_grad_kwargs(self):
        fun = lambda x, y: x * y
        a, b = mx.array(0.5), mx.array(2.0)
        dfdx = mx.grad(fun)
        self.assertEqual(dfdx(a, b).item(), 2.0)
        self.assertEqual(dfdx(a, y=b).item(), 2.0)
        with self.assertRaises(ValueError):
            dfdx(x=a, y=b).item()

        dfdy = mx.grad(fun, argnums=[], argnames=["y"])
        with self.assertRaises(ValueError):
            dfdy(a, b)
        grads = dfdy(a, y=b)
        self.assertTrue(isinstance(grads, tuple))
        self.assertTrue(grads[0] is None)
        self.assertTrue(isinstance(grads[1], dict))
        self.assertEqual(grads[1]["y"].item(), 0.5)
        grads = dfdy(x=a, y=b)
        self.assertEqual(grads[1]["y"].item(), 0.5)
        self.assertEqual(len(grads[1]), 1)

        dfdxy = mx.grad(fun, argnums=[0], argnames=["y"])
        with self.assertRaises(ValueError):
            dfdxy(a, b)
        with self.assertRaises(ValueError):
            dfdxy(x=a, y=b)
        grads = dfdxy(a, y=b)
        self.assertTrue(isinstance(grads, tuple))
        self.assertEqual(grads[0].item(), 2.0)
        self.assertTrue(isinstance(grads[1], dict))
        self.assertEqual(grads[1]["y"].item(), 0.5)

        fun = lambda x, y, z: x * y * z
        dfdxyz = mx.grad(fun, argnums=[0, 1], argnames=["z"])
        c = mx.array(4.0)
        grads = dfdxyz(a, b, z=c)
        self.assertTrue(isinstance(grads, tuple))
        self.assertTrue(isinstance(grads[0], tuple))
        self.assertEqual(grads[0][0].item(), 8.0)
        self.assertEqual(grads[0][1].item(), 2.0)
        self.assertTrue(isinstance(grads[1], dict))
        self.assertEqual(grads[1]["z"].item(), 1.0)

        fun = lambda x, y: x * y
        dfdy = mx.grad(fun, argnames=["y"])
        grads = dfdy(a, y=b)
        self.assertTrue(isinstance(grads, tuple))
        self.assertTrue(grads[0] is None)
        self.assertTrue(isinstance(grads[1], dict))
        self.assertEqual(grads[1]["y"].item(), 0.5)

    def test_captured(self):
        a = mx.array(5.0)
        f = lambda x: a + x
        g = lambda x: a + a
        h = lambda x: x + x

        dfdx = mx.grad(f)
        self.assertEqual(dfdx(a).item(), 1.0)

        dgdx = mx.grad(g)
        self.assertEqual(dgdx(a).item(), 0.0)

        dhdx = mx.grad(h)
        self.assertEqual(dhdx(a).item(), 2.0)

        d2fdx2 = mx.grad(dfdx)
        self.assertEqual(d2fdx2(a).item(), 0.0)

        d2gdx2 = mx.grad(dgdx)
        self.assertEqual(d2gdx2(a).item(), 0.0)

        d2hdx2 = mx.grad(dhdx)
        self.assertEqual(d2hdx2(a).item(), 0.0)

    def test_stop_gradient(self):
        shape_in = (4, 4)
        w_in = mx.ones(shape_in)
        x_in = mx.ones(shape_in)
        cotan = mx.ones(shape_in)

        def h(w, x):
            x1 = 2 * x
            y = mx.stop_gradient(x1)
            y1 = 3 * y
            return w @ y1

        vals, vjps = mx.vjp(h, [w_in, x_in], [cotan])
        mx.eval(vjps)

        self.assertTrue(mx.allclose(vjps[0], 24.0 * mx.ones(shape_in)))
        self.assertTrue(mx.allclose(vjps[1], mx.zeros(shape_in)))

        g = lambda x: h(w_in, x)
        vals, vjps = mx.vjp(g, [x_in], [cotan])
        mx.eval(vjps)

        self.assertTrue(mx.allclose(vjps[0], mx.zeros(shape_in)))


if __name__ == "__main__":
    unittest.main()
