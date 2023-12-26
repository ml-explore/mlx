# Copyright © 2023 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests


class TestVmap(mlx_tests.MLXTestCase):
    def test_basics(self):
        # Can't vmap over scalars
        with self.assertRaises(ValueError):
            mx.vmap(mx.exp)(mx.array(1.0))

        # Invalid input
        with self.assertRaises(ValueError):
            mx.vmap(mx.exp)("hello")

        # Invalid axes
        with self.assertRaises(ValueError):
            mx.vmap(mx.exp, in_axes="hello")(mx.array([0, 1]))

        with self.assertRaises(ValueError):
            mx.vmap(mx.exp, in_axes=2)(mx.array([0, 1]))

        with self.assertRaises(ValueError):
            mx.vmap(mx.exp, out_axes="hello")(mx.array([0, 1]))

        with self.assertRaises(ValueError):
            mx.vmap(mx.exp, out_axes=2)(mx.array([0, 1]))

    def test_unary(self):
        ops = [
            "abs",
            "cos",
            "erf",
            "erfinv",
            "exp",
            "log",
            "log1p",
            "log2",
            "log10",
            "logical_not",
            "negative",
            "reciprocal",
            "rsqrt",
            "sigmoid",
            "sign",
            "sin",
            "sqrt",
            "square",
        ]
        ops = ["erfinv"]
        for opname in ops:
            with self.subTest(op=opname):
                op = getattr(mx, opname)
                x = mx.arange(5)
                y = mx.vmap(op)(x)
                self.assertTrue(mx.array_equal(y, op(x), equal_nan=True))

                x = mx.arange(8).reshape(2, 4)
                y = mx.vmap(op)(x)
                self.assertTrue(mx.array_equal(y, op(x), equal_nan=True))

                y = mx.vmap(op, in_axes=1, out_axes=1)(x)
                self.assertTrue(mx.array_equal(y, op(x), equal_nan=True))

    def test_binary(self):
        ops = [
            "add",
            "divide",
            "equal",
            "greater",
            "greater_equal",
            "less",
            "less_equal",
            "logaddexp",
            "maximum",
            "minimum",
            "multiply",
            "power",
            "subtract",
        ]
        for opname in ops:
            with self.subTest(op=opname):
                op = getattr(mx, opname)
                x = mx.random.uniform(shape=(5,))
                y = mx.random.uniform(shape=(5,))
                out = mx.vmap(op)(x, y)
                self.assertTrue(mx.array_equal(out, op(x, y)))

                x = mx.random.uniform(shape=(2, 4))
                y = mx.random.uniform(shape=(2, 4))
                out = mx.vmap(op)(x, y)
                self.assertTrue(mx.array_equal(out, op(x, y)))

                out = mx.vmap(op, in_axes=(0, 0), out_axes=0)(x, y)
                self.assertTrue(mx.array_equal(out, op(x, y)))

                y = mx.random.uniform(shape=(4, 2))
                out = mx.vmap(op, in_axes=(0, 1), out_axes=0)(x, y)
                self.assertTrue(mx.array_equal(out, op(x, y.T)))

                out = mx.vmap(op, in_axes=(0, 1), out_axes=1)(x, y)
                self.assertTrue(mx.array_equal(out, op(x, y.T).T))

    def test_tree(self):
        def my_fun(tree):
            return (tree["a"] + tree["b"][0]) * tree["b"][1]

        tree = {
            "a": mx.random.uniform(shape=(2, 4)),
            "b": (
                mx.random.uniform(shape=(2, 4)),
                mx.random.uniform(shape=(2, 4)),
            ),
        }
        out = mx.vmap(my_fun)(tree)
        expected = my_fun(tree)
        self.assertTrue(mx.array_equal(out, my_fun(tree)))

        with self.assertRaises(ValueError):
            mx.vmap(my_fun, in_axes={"a": 0, "b": 0}, out_axes=0)(tree)

        with self.assertRaises(ValueError):
            mx.vmap(my_fun, in_axes={"a": 0, "b": ((0, 0), 0)}, out_axes=0)(tree)

        out = mx.vmap(my_fun, in_axes=({"a": 0, "b": 0},), out_axes=0)(tree)
        self.assertTrue(mx.array_equal(out, my_fun(tree)))

        out = mx.vmap(my_fun, in_axes=({"a": 0, "b": (0, 0)},), out_axes=0)(tree)
        self.assertTrue(mx.array_equal(out, my_fun(tree)))

        tree = {
            "a": mx.random.uniform(shape=(2, 4)),
            "b": (
                mx.random.uniform(shape=(4, 2)),
                mx.random.uniform(shape=(4, 2)),
            ),
        }
        out = mx.vmap(my_fun, in_axes=({"a": 0, "b": (1, 1)},), out_axes=0)(tree)
        expected = (tree["a"] + tree["b"][0].T) * tree["b"][1].T
        self.assertTrue(mx.array_equal(out, expected))

        def my_fun(x, y):
            return {"a": x + y, "b": x * y}

        x = mx.random.uniform(shape=(2, 4))
        y = mx.random.uniform(shape=(2, 4))
        out = mx.vmap(my_fun, in_axes=0, out_axes=0)(x, y)
        expected = my_fun(x, y)
        self.assertTrue(mx.array_equal(out["a"], expected["a"]))
        self.assertTrue(mx.array_equal(out["b"], expected["b"]))

        with self.assertRaises(ValueError):
            mx.vmap(my_fun, in_axes=0, out_axes=(0, 1))(x, y)

        with self.assertRaises(ValueError):
            mx.vmap(my_fun, in_axes=0, out_axes={"a": 0, "c": 1})(x, y)

        out = mx.vmap(my_fun, in_axes=0, out_axes={"a": 1, "b": 0})(x, y)
        expected = my_fun(x, y)
        self.assertTrue(mx.array_equal(out["a"].T, expected["a"]))
        self.assertTrue(mx.array_equal(out["b"], expected["b"]))

    def test_vmap_indexing(self):
        x = mx.arange(16).reshape(2, 2, 2, 2)
        inds = mx.array([[0, 1, 0], [1, 1, 0]])

        out = mx.vmap(lambda x, y: x[y], in_axes=(0, 0))(x, inds)
        expected = mx.array(
            [
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[0, 1], [2, 3]]],
                [[[12, 13], [14, 15]], [[12, 13], [14, 15]], [[8, 9], [10, 11]]],
            ]
        )
        self.assertTrue(mx.array_equal(out, expected))

        out = mx.vmap(lambda x, y: x[y], in_axes=(0, None))(x, inds)
        expected = mx.array(
            [
                [
                    [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[0, 1], [2, 3]]],
                    [[[4, 5], [6, 7]], [[4, 5], [6, 7]], [[0, 1], [2, 3]]],
                ],
                [
                    [[[8, 9], [10, 11]], [[12, 13], [14, 15]], [[8, 9], [10, 11]]],
                    [[[12, 13], [14, 15]], [[12, 13], [14, 15]], [[8, 9], [10, 11]]],
                ],
            ]
        )
        self.assertTrue(mx.array_equal(out, expected))

        out = mx.vmap(lambda x, y: x[y], in_axes=(None, 0))(x, inds)
        expected = mx.array(
            [
                [
                    [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                    [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
                    [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                ],
                [
                    [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
                    [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
                    [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
                ],
            ]
        )
        self.assertTrue(mx.array_equal(out, expected))

        inds2 = mx.array([[0, 1, 0], [0, 1, 0]])
        out = mx.vmap(lambda x, y, z: x[y, z], in_axes=(None, 0, 0))(x, inds, inds2)
        expected = mx.array(
            [
                [[[0, 1], [2, 3]], [[12, 13], [14, 15]], [[0, 1], [2, 3]]],
                [[[8, 9], [10, 11]], [[12, 13], [14, 15]], [[0, 1], [2, 3]]],
            ]
        )
        self.assertTrue(mx.array_equal(out, expected))


if __name__ == "__main__":
    unittest.main()
