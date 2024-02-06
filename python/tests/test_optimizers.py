# Copyright © 2023 Apple Inc.

import inspect
import unittest

import mlx.core as mx
import mlx.optimizers as opt
import mlx.utils
import mlx_tests
from mlx.utils import tree_flatten, tree_map


def get_all_optimizers():
    classes = dict()
    for name, obj in inspect.getmembers(opt):
        if inspect.isclass(obj):
            if obj.__name__ not in ["Optimizer"]:
                classes[name] = obj
    return classes


def tree_equal(fn, *args):
    return all(v for _, v in tree_flatten(tree_map(fn, *args)))


optimizers_dict = get_all_optimizers()


class TestOptimizers(mlx_tests.MLXTestCase):
    def test_optimizer_state(self):
        optim = opt.SGD(0.1)
        optim.state["hello"] = "world"
        self.assertEqual(optim.state["hello"], "world")

        optim.state = {0: 1}
        self.assertEqual(optim.state, {0: 1})

    def test_optimizers(self):
        params = {
            "first": [mx.zeros((10,)), mx.zeros((1,))],
            "second": mx.zeros((1,)),
        }
        grads = tree_map(lambda x: mx.ones_like(x), params)

        for optim_class in optimizers_dict.values():
            optim = optim_class(0.1)
            update = optim.apply_gradients(grads, params)
            mx.eval(update)
            equal_shape = tree_map(lambda x, y: x.shape == y.shape, params, update)
            all_equal = all(v for _, v in mlx.utils.tree_flatten(equal_shape))
            self.assertTrue(all_equal)

    def test_sgd(self):
        params = {
            "first": [mx.zeros((10,)), mx.zeros((1,))],
            "second": mx.zeros((1,)),
        }
        grads = tree_map(lambda x: mx.ones_like(x), params)

        # Explicit init
        optim = opt.SGD(learning_rate=1e-2, momentum=0.9)
        optim.init(params)
        self.assertTrue(
            tree_equal(
                lambda p, s: mx.array_equal(s["v"], mx.zeros_like(p)),
                params,
                optim.state,
            )
        )

        # Implicit init
        optim = opt.SGD(learning_rate=1e-2, momentum=0.9)
        optim.apply_gradients(grads, params)
        self.assertTrue(
            tree_equal(lambda g, s: mx.array_equal(s["v"], g), grads, optim.state)
        )

    def test_rmsprop(self):
        params = {
            "first": [mx.zeros((10,)), mx.zeros((1,))],
            "second": mx.zeros((1,)),
        }
        grads = tree_map(lambda x: mx.ones_like(x), params)

        # Explicit init
        optim = opt.RMSprop(learning_rate=1e-2)
        optim.init(params)
        self.assertTrue(
            tree_equal(
                lambda p, s: mx.array_equal(s["v"], mx.zeros_like(p)),
                params,
                optim.state,
            )
        )

        # Implicit init
        alpha = 0.99
        optim = opt.RMSprop(learning_rate=1e-2, alpha=alpha)
        optim.apply_gradients(grads, params)
        self.assertTrue(
            tree_equal(
                lambda g, s: mx.allclose(s["v"], (1 - alpha) * g), grads, optim.state
            )
        )

    def test_adagrad(self):
        params = {
            "first": [mx.zeros((10,)), mx.zeros((1,))],
            "second": mx.zeros((1,)),
        }
        grads = tree_map(lambda x: mx.ones_like(x), params)

        # Explicit init
        optim = opt.Adagrad(learning_rate=1e-2)
        optim.init(params)
        self.assertTrue(
            tree_equal(
                lambda p, s: mx.array_equal(s["v"], mx.zeros_like(p)),
                params,
                optim.state,
            )
        )

    def test_adadelta(self):
        params = {
            "first": [mx.zeros((10,)), mx.zeros((1,))],
            "second": mx.zeros((1,)),
        }
        grads = tree_map(lambda x: mx.ones_like(x), params)

        # Explicit init
        optim = opt.AdaDelta(learning_rate=1e-2)
        optim.init(params)
        self.assertTrue(
            tree_equal(
                lambda p, s: mx.array_equal(s["v"], mx.zeros_like(p)),
                params,
                optim.state,
            )
        )
        self.assertTrue(
            tree_equal(
                lambda p, s: mx.array_equal(s["u"], mx.zeros_like(p)),
                params,
                optim.state,
            )
        )

    def test_adam(self):
        params = {
            "first": [mx.zeros((10,)), mx.zeros((1,))],
            "second": mx.zeros((1,)),
        }
        grads = tree_map(lambda x: mx.ones_like(x), params)

        # Explicit init
        for optimizer in [opt.Adam, opt.AdamW, opt.Adamax]:
            optim = optimizer(learning_rate=1e-2)
            optim.init(params)
            self.assertTrue(
                tree_equal(
                    lambda p, s: mx.array_equal(s["v"], mx.zeros_like(p)),
                    params,
                    optim.state,
                )
            )
            self.assertTrue(
                tree_equal(
                    lambda p, s: mx.array_equal(s["m"], mx.zeros_like(p)),
                    params,
                    optim.state,
                )
            )

    def test_lion(self):
        params = {
            "first": [mx.zeros((10,)), mx.zeros((1,))],
            "second": mx.zeros((1,)),
        }
        grads = tree_map(lambda x: mx.ones_like(x), params)

        # Explicit init
        optim = opt.Lion(learning_rate=1e-2)
        optim.init(params)
        self.assertTrue(
            tree_equal(
                lambda p, s: mx.array_equal(s["m"], mx.zeros_like(p)),
                params,
                optim.state,
            )
        )

    def test_adafactor(self):
        x = mx.zeros((5, 5))
        grad = mx.ones_like(x)
        optimizer = opt.Adafactor()
        optimizer.init(x)
        for _ in range(2):
            xp = optimizer.apply_single(grad, x, optimizer.state)
            self.assertEqual(xp.dtype, x.dtype)
            self.assertEqual(xp.shape, x.shape)

        x = mx.zeros((5, 5), mx.float16)
        grad = mx.ones_like(x)
        optimizer = opt.Adafactor()
        optimizer.init(x)
        for _ in range(2):
            xp = optimizer.apply_single(grad, x, optimizer.state)
            self.assertEqual(xp.dtype, x.dtype)
            self.assertEqual(xp.shape, x.shape)
        self.assertEqual(optimizer.state["step"], 2)


if __name__ == "__main__":
    unittest.main()
