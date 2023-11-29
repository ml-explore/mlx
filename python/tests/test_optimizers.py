import unittest

import mlx.core as mx
import mlx.optimizers as opt
import mlx.utils

import mlx_tests


class TestOptimizers(mlx_tests.MLXTestCase):
    def test_optimizers(self):
        params = {
            "first": [mx.zeros((10,)), mx.zeros((1,))],
            "second": mx.zeros((1,)),
        }
        grads = mlx.utils.tree_map(lambda x: mx.ones_like(x), params)

        for optim in [opt.SGD(0.1), opt.Adam(0.1)]:
            update = optim.apply_gradients(grads, params)
            mx.eval(update)
            equal_shape = mlx.utils.tree_map(
                lambda x, y: x.shape == y.shape, params, update
            )
            all_equal = all(v for _, v in mlx.utils.tree_flatten(equal_shape))
            self.assertTrue(all_equal)


if __name__ == "__main__":
    unittest.main()
