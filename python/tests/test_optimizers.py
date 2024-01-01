# Copyright © 2023 Apple Inc.

import inspect
import math
import unittest
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import mlx.utils
import mlx_tests
from mlx.utils import tree_flatten, tree_map


def get_all_optimizers():
    classes = dict()
    for name, obj in inspect.getmembers(opt):
        if (
            inspect.isclass(obj)
            and issubclass(obj, opt.Optimizer)
            and obj != opt.Optimizer
        ):
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

    def test_types_conserved(self):
        params = {"w": mx.ones((5, 5), mx.float16)}
        grads = tree_map(lambda x: mx.ones_like(x), params)
        for optim_class in optimizers_dict.values():
            optim = optim_class(0.1)
            update = optim.apply_gradients(grads, params)
            self.assertEqual(update["w"].dtype, mx.float16)

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

    def test_compiled_optimizer(self):
        model = nn.Linear(10, 10)
        x = mx.random.uniform(shape=(2, 10))
        optim = opt.SGD(learning_rate=1e-2, momentum=0.9)

        orig_params = model.parameters()

        def loss(model, x):
            return model(x).sum()

        # Uncompiled version
        def step(x):
            _, grad = nn.value_and_grad(model, loss)(model, x)
            optim.update(model, grad)

        step(x)
        uncompiled_params = model.parameters()

        # Pure version
        def loss(params, x):
            model.update(params)
            return model(x).sum()

        model.update(orig_params)
        optim = opt.SGD(learning_rate=1e-2, momentum=0.9)

        @mx.compile
        def step(params, opt_state, x):
            grad = mx.grad(loss)(params, x)
            optim.state = opt_state
            params = optim.apply_gradients(grad, params)
            return params, optim.state

        optim.init(model.parameters())
        pure_params, _ = step(model.parameters(), optim.state, x)
        self.assertTrue(mx.allclose(pure_params["weight"], uncompiled_params["weight"]))
        self.assertTrue(mx.allclose(pure_params["bias"], uncompiled_params["bias"]))

        # Impure version
        def loss(model, x):
            return model(x).sum()

        model.update(orig_params)
        optim = opt.SGD(learning_rate=1e-2, momentum=0.9)
        state = [model.state, optim.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(x):
            _, grad = nn.value_and_grad(model, loss)(model, x)
            optim.update(model, grad)

        step(x)
        impure_params = model.parameters()
        self.assertTrue(
            mx.allclose(impure_params["weight"], uncompiled_params["weight"])
        )
        self.assertTrue(mx.allclose(impure_params["bias"], uncompiled_params["bias"]))

    def test_update_lr_compiled(self):
        params = {"w": mx.ones((5, 5))}
        grads = tree_map(lambda x: mx.ones_like(x), params)
        optim = opt.SGD(-1.0)

        @partial(mx.compile, inputs=optim.state)
        def update(grads):
            return optim.apply_gradients(grads, params)

        result = update(grads)
        self.assertTrue(mx.allclose(result["w"], mx.full((5, 5), 2.0)))
        optim.learning_rate = -2.0
        result = update(grads)
        self.assertTrue(mx.allclose(result["w"], mx.full((5, 5), 3.0)))


class TestLRSchedulers(unittest.TestCase):
    def test_step_lr_with_optimizers(self):
        for optim_class in optimizers_dict.values():
            optimizer = optim_class(learning_rate=0.1)
            scheduler = opt.StepLR(optimizer, step_size=1, gamma=0.9)

            for epoch in range(10):
                scheduler.step()
                expected_lr = 0.1 * (0.9**epoch)
                self.assertAlmostEqual(optimizer.learning_rate, expected_lr, delta=1e-7)

    def test_exponential_lr_with_optimizers(self):
        for optim_class in optimizers_dict.values():
            optimizer = optim_class(learning_rate=0.1)
            scheduler = opt.ExponentialLR(optimizer, gamma=0.9)

            for epoch in range(10):
                scheduler.step()
                expected_lr = 0.1 * (0.9**epoch)
                self.assertAlmostEqual(optimizer.learning_rate, expected_lr, delta=1e-7)

    def test_multi_step_lr_with_optimizers(self):
        for optim_class in optimizers_dict.values():
            optimizer = optim_class(learning_rate=0.1)
            scheduler = opt.MultiStepLR(optimizer, milestones=[2, 4], gamma=0.1)

            for epoch in range(6):
                scheduler.step()
                expected_lr = 0.1 * (
                    0.1 ** sum(epoch >= milestone for milestone in [2, 4])
                )
                self.assertAlmostEqual(optimizer.learning_rate, expected_lr, delta=1e-7)

    def test_lambda_lr_with_optimizers(self):
        lr_lambda = lambda epoch: 0.95**epoch
        for optim_class in optimizers_dict.values():
            optimizer = optim_class(learning_rate=0.1)
            scheduler = opt.LambdaLR(optimizer, lr_lambda)

            for epoch in range(5):
                scheduler.step()
                expected_lr = 0.1 * (0.95**epoch)
                self.assertAlmostEqual(optimizer.learning_rate, expected_lr, delta=1e-7)

    def test_polynomial_lr_with_optimizers(self):
        for optim_class in optimizers_dict.values():
            optimizer = optim_class(learning_rate=0.1)
            scheduler = opt.PolynomialLR(
                optimizer, max_decay_steps=5, end_lr=0.01, power=2
            )

            for epoch in range(7):
                scheduler.step()
                decay_steps = min(epoch, 5)
                decay_factor = (1 - decay_steps / 5) ** 2
                expected_lr = (0.1 - 0.01) * decay_factor + 0.01
                self.assertAlmostEqual(optimizer.learning_rate, expected_lr, delta=1e-7)

    def test_cosine_annealing_lr_with_optimizers(self):
        for optim_class in optimizers_dict.values():
            optimizer = optim_class(learning_rate=0.1)
            scheduler = opt.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.01)

            for epoch in range(12):
                scheduler.step()
                expected_lr = (
                    0.01 + (0.1 - 0.01) * (1 + math.cos(math.pi * epoch / 10)) / 2
                )
                self.assertAlmostEqual(optimizer.learning_rate, expected_lr, delta=1e-7)


if __name__ == "__main__":
    unittest.main()
