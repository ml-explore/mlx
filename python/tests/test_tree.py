# Copyright © 2023 Apple Inc.

import unittest

import mlx.core as mx
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


if __name__ == "__main__":
    unittest.main()
