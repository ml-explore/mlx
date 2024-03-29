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
        for _ in range(2):
            xp = optimizer.apply_gradients(grad, x)
            self.assertEqual(xp.dtype, x.dtype)
            self.assertEqual(xp.shape, x.shape)

        x = mx.zeros((5, 5), mx.float16)
        grad = mx.ones_like(x)
        optimizer = opt.Adafactor()
        for _ in range(2):
            xp = optimizer.apply_gradients(grad, x)
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


class TestSchedulers(unittest.TestCase):
    def test_decay_lr(self):
        for optim_class in optimizers_dict.values():
            lr_schedule = opt.step_decay(1e-1, 0.9, 1)
            optimizer = optim_class(learning_rate=lr_schedule)

            params = {"w": mx.ones((5, 5))}
            grads = tree_map(lambda x: mx.ones_like(x), params)

            for it in range(10):
                optimizer.apply_gradients(grads, params)
                expected_lr = 0.1 * (0.9**it)
                self.assertAlmostEqual(optimizer.learning_rate, expected_lr, delta=1e-7)

    def test_step_decay(self):
        lr_schedule = opt.step_decay(1e-1, 0.9, 1000)
        lr = lr_schedule(2500)
        expected_lr = 0.1 * (0.9**2)
        self.assertAlmostEqual(lr, expected_lr, delta=1e-7)

    def test_exponential_decay(self):
        lr_schedule = opt.exponential_decay(1e-1, 0.99)
        lr = lr_schedule(10)
        expected_lr = 0.1 * (0.99**10)
        self.assertAlmostEqual(lr, expected_lr, delta=1e-7)

    def test_cosine_decay(self):
        lr_schedule = opt.cosine_decay(0.1, 10)
        lr = lr_schedule(4)
        expected_lr = 0.1 * 0.5 * (1.0 + math.cos(math.pi * 4 / 10))
        self.assertAlmostEqual(lr, expected_lr, delta=1e-7)

        lr_schedule = opt.cosine_decay(0.1, 10, 0.05)
        lr = lr_schedule(20)
        expected_lr = 0.05
        self.assertEqual(lr, expected_lr)

    def test_schedule_joiner(self):
        boundaries = [2, 3, 4]
        schedules = [lambda _: 3, lambda _: 4, lambda _: 5]
        with self.assertRaises(ValueError):
            opt.schedulers.join_schedules(schedules, boundaries)
        boundaries = [2, 4]
        schedule = opt.schedulers.join_schedules(schedules, boundaries)
        self.assertEqual(schedule(0).item(), 3)
        self.assertEqual(schedule(1).item(), 3)
        self.assertEqual(schedule(2).item(), 4)
        self.assertEqual(schedule(3).item(), 4)
        self.assertEqual(schedule(5).item(), 5)
        self.assertEqual(schedule(7).item(), 5)

    def test_linear_warmup_with_cosine_decay(self):
        warmup_schedule = opt.schedulers.linear_schedule(0.0, 1e-5, 100)
        cosine_schedule = opt.schedulers.cosine_decay(1e-5, 100)
        cos_with_warmup = opt.schedulers.join_schedules(
            [warmup_schedule, cosine_schedule], [101]
        )
        self.assertEqual(cos_with_warmup(0), 0.0)
        self.assertAlmostEqual(cos_with_warmup(101), 1e-5, delta=1e-1)
        optimizer = opt.Adam(learning_rate=cos_with_warmup)
        for _ in range(100):
            optimizer.update({}, {})
        self.assertAlmostEqual(optimizer.learning_rate.item(), 1e-5, delta=1e-1)
        for _ in range(100):
            optimizer.update({}, {})
        expected_lr = 1e-5 * 0.5 * (1.0 + math.cos(math.pi * 200 / 10))
        self.assertAlmostEqual(optimizer.learning_rate.item(), expected_lr, delta=1e-1)

    def test_compile_with_schedule(self):
        lr_schedule = opt.exponential_decay(1e-1, 0.9)
        optimizer = opt.SGD(learning_rate=lr_schedule)

        @partial(mx.compile, inputs=optimizer.state, outputs=optimizer.state)
        def update():
            optimizer.update({}, {})

        for step in range(5):
            update()
            self.assertAlmostEqual(lr_schedule(step), optimizer.learning_rate.item())


if __name__ == "__main__":
    unittest.main()
