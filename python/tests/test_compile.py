# Copyright © 2023-2024 Apple Inc.

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


if __name__ == "__main__":
    unittest.main()
