# Copyright © 2023 Apple Inc.

import inspect
import math
import unittest

import mlx.core as mx
import mlx.optimizers as opt
import mlx.utils
import mlx_tests


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


optimizers_dict = get_all_optimizers()


class TestOptimizers(mlx_tests.MLXTestCase):
    def test_optimizers(self):
        params = {
            "first": [mx.zeros((10,)), mx.zeros((1,))],
            "second": mx.zeros((1,)),
        }
        grads = mlx.utils.tree_map(lambda x: mx.ones_like(x), params)

        for optim_class in optimizers_dict.values():
            optim = optim_class(0.1)
            update = optim.apply_gradients(grads, params)
            mx.eval(update)
            equal_shape = mlx.utils.tree_map(
                lambda x, y: x.shape == y.shape, params, update
            )
            all_equal = all(v for _, v in mlx.utils.tree_flatten(equal_shape))
            self.assertTrue(all_equal)


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
