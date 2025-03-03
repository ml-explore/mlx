# Copyright © 2023 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import mlx_tests


class TestTreeUtils(mlx_tests.MLXTestCase):
    def test_tree_map(self):
        tree = {"a": 0, "b": 1, "c": 2}
        tree = mlx.utils.tree_map(lambda x: x + 1, tree)

        expected_tree = {"a": 1, "b": 2, "c": 3}
        self.assertEqual(tree, expected_tree)

    def test_tree_flatten(self):
        tree = [{"a": 1, "b": 2}, "c"]
        vals = (1, 2, "c")
        flat_tree = mlx.utils.tree_flatten(tree)
        self.assertEqual(list(zip(*flat_tree))[1], vals)
        self.assertEqual(mlx.utils.tree_unflatten(flat_tree), tree)

    def test_merge(self):
        t1 = {"a": 0}
        t2 = {"b": 1}
        t = mlx.utils.tree_merge(t1, t2)
        self.assertEqual({"a": 0, "b": 1}, t)
        with self.assertRaises(ValueError):
            mlx.utils.tree_merge(t1, t1)
        with self.assertRaises(ValueError):
            mlx.utils.tree_merge(t, t1)

        mod1 = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        mod2 = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        mod = nn.Sequential(mod1, mod2)

        params1 = {"layers": [mod1.parameters()]}
        params2 = {"layers": [None, mod2.parameters()]}
        params = mlx.utils.tree_merge(params1, params2)
        for (k1, v1), (k2, v2) in zip(
            mlx.utils.tree_flatten(params), mlx.utils.tree_flatten(mod.parameters())
        ):
            self.assertEqual(k1, k2)
            self.assertTrue(mx.array_equal(v1, v2))


if __name__ == "__main__":
    unittest.main()
