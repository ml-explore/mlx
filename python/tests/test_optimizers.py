# Copyright © 2023 Apple Inc.

import inspect
import unittest

import mlx.core as mx
import mlx.optimizers as opt
import mlx.utils
import mlx_tests


def get_all_optimizers():
    classes = dict()
    for name, obj in inspect.getmembers(opt):
        if inspect.isclass(obj):
            if obj.__name__ not in ["OptimizerState", "Optimizer"]:
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

    def test_adafactor(self):
        x = mx.zeros((5, 5))
        grad = mx.ones_like(x)
        optimizer = opt.Adafactor()
        for _ in range(2):
            xp = optimizer.apply_single(grad, x, optimizer.state)
            self.assertEqual(xp.dtype, x.dtype)
            self.assertEqual(xp.shape, x.shape)

        x = mx.zeros((5, 5), mx.float16)
        grad = mx.ones_like(x)
        optimizer = opt.Adafactor()
        for _ in range(2):
            xp = optimizer.apply_single(grad, x, optimizer.state)
            self.assertEqual(xp.dtype, x.dtype)
            self.assertEqual(xp.shape, x.shape)
        self.assertEqual(optimizer.state["step"], 2)


if __name__ == "__main__":
    unittest.main()
