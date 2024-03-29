# Copyright © 2023-2024 Apple Inc.

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
            "logical_or",
            "logical_and",
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

    def test_vmap_reduce(self):
        a = mx.ones((5, 5), mx.int32)
        out = mx.vmap(lambda x: x.sum())(a)
        self.assertTrue(mx.array_equal(out, mx.full((5,), 5)))

        out = mx.vmap(lambda x: x.sum(keepdims=True))(a)
        self.assertTrue(mx.array_equal(out, mx.full((5, 1), 5)))

        out = mx.vmap(lambda x: x.sum(axis=0))(a)
        self.assertTrue(mx.array_equal(out, mx.full((5,), 5)))

        a = mx.ones((5, 3, 2), mx.int32)
        out = mx.vmap(lambda x: x.sum(axis=(0, 1)))(a)
        self.assertTrue(mx.array_equal(out, mx.full((5,), 6)))

        a = mx.ones((5, 3, 2), mx.int32)
        out = mx.vmap(lambda x: x.sum(axis=(0, 1)), in_axes=(1,))(a)
        self.assertTrue(mx.array_equal(out, mx.full((3,), 10)))

        a = mx.ones((5, 3, 2), mx.int32)
        out = mx.vmap(lambda x: x.sum(axis=(0, 1)), in_axes=(2,))(a)
        self.assertTrue(mx.array_equal(out, mx.full((2,), 15)))

    def test_vmap_argreduce(self):
        a = mx.array([[1, 2, 3], [2, 3, 1]])
        out = mx.vmap(lambda x: mx.argmin(x))(a)
        expected = mx.array([0, 2])
        self.assertTrue(mx.array_equal(out, expected))

        out = mx.vmap(lambda x: mx.argmax(x))(a)
        expected = mx.array([2, 1])
        self.assertTrue(mx.array_equal(out, expected))

    def test_vmap_mean(self):
        a = mx.arange(8).reshape(2, 4)
        out = mx.vmap(mx.mean)(a)
        expected = mx.mean(a, axis=1)
        self.assertTrue(mx.allclose(out, expected))

        a = mx.arange(16).reshape(2, 2, 4)
        out = mx.vmap(mx.vmap(mx.mean))(a)
        expected = mx.mean(a, axis=2)
        self.assertTrue(mx.allclose(out, expected))

    def test_mismatch_input_sizes(self):
        a = mx.ones((10, 1))
        b = mx.ones((1, 1, 1, 5))

        with self.assertRaises(ValueError):
            out = mx.vmap(lambda x, y: x + y)(a, b)

        b = mx.ones((10, 5))
        with self.assertRaises(ValueError):
            out = mx.vmap(lambda x, y: x + y, in_axes=(0, 1))(a, b)

    def test_vmap_matmul(self):
        a = mx.random.uniform(shape=(2, 3, 4))
        b = mx.random.uniform(shape=(4, 3))

        # matmul
        out = mx.vmap(mx.matmul, in_axes=(0, None))(a, b)
        self.assertTrue(mx.allclose(out, a @ b))

        # addmm
        c = mx.random.uniform(shape=(3,))
        out = mx.vmap(mx.addmm, in_axes=(None, 0, None))(c, a, b)
        self.assertTrue(mx.allclose(out, mx.addmm(c, a, b)))

        b = mx.random.uniform(shape=(4, 2))

        # matmul
        out = mx.vmap(mx.matmul, in_axes=(1, None), out_axes=(1,))(a, b)
        expected = mx.moveaxis(mx.moveaxis(a, 1, 0) @ b, 0, 1)
        self.assertTrue(mx.allclose(out, expected))

        # addmm
        c = mx.random.uniform(shape=(2,))
        out = mx.vmap(mx.addmm, in_axes=(None, 1, None))(c, a, b)
        self.assertTrue(mx.allclose(out, mx.addmm(c, mx.moveaxis(a, 1, 0), b)))

        a = mx.random.uniform(shape=(2, 3, 4))
        b = mx.random.uniform(shape=(4, 2, 3))

        # matmul
        out = mx.vmap(mx.matmul, in_axes=(0, 1))(a, b)
        expected = a @ mx.moveaxis(b, 1, 0)
        self.assertTrue(mx.allclose(out, expected))

        # addmm
        c = mx.random.uniform(shape=(3, 3, 2))
        out = mx.vmap(mx.addmm, in_axes=(2, 0, 1))(c, a, b)
        expected = mx.addmm(mx.moveaxis(c, 2, 0), a, mx.moveaxis(b, 1, 0))
        self.assertTrue(mx.allclose(out, expected))

    def test_vmap_svd(self):
        a = mx.random.uniform(shape=(3, 4, 2))

        cpu_svd = lambda x: mx.linalg.svd(x, stream=mx.cpu)

        # Vmap over the first axis (this is already supported natively by the primitive).
        Us, Ss, Vts = mx.vmap(cpu_svd, in_axes=(0,))(a)
        self.assertEqual(Us.shape, (a.shape[0], a.shape[1], a.shape[1]))
        self.assertEqual(Ss.shape, (a.shape[0], a.shape[2]))
        self.assertEqual(Vts.shape, (a.shape[0], a.shape[2], a.shape[2]))

        for i in range(a.shape[0]):
            M = a[i]
            U, S, Vt = Us[i], Ss[i], Vts[i]
            self.assertTrue(
                mx.allclose(U[:, : len(S)] @ mx.diag(S) @ Vt, M, rtol=1e-5, atol=1e-7)
            )

        # Vmap over the second axis.
        Us, Ss, Vts = mx.vmap(cpu_svd, in_axes=(1,))(a)
        self.assertEqual(Us.shape, (a.shape[1], a.shape[0], a.shape[0]))
        self.assertEqual(Ss.shape, (a.shape[1], a.shape[2]))
        self.assertEqual(Vts.shape, (a.shape[1], a.shape[2], a.shape[2]))

        for i in range(a.shape[1]):
            M = a[:, i, :]
            U, S, Vt = Us[i], Ss[i], Vts[i]
            self.assertTrue(
                mx.allclose(U[:, : len(S)] @ mx.diag(S) @ Vt, M, rtol=1e-5, atol=1e-7)
            )

    def test_vmap_inverse(self):
        a = mx.random.uniform(shape=(3, 4, 4))

        cpu_inv = lambda x: mx.linalg.inv(x, stream=mx.cpu)

        # Vmap over the first axis (this is already supported natively by the primitive).
        invs = mx.vmap(cpu_inv, in_axes=(0,))(a)

        for i in range(a.shape[0]):
            self.assertTrue(
                mx.allclose(a[i] @ invs[i], mx.eye(a.shape[1]), rtol=0, atol=1e-5)
            )

        a = mx.random.uniform(shape=(4, 3, 4))

        # Without vmapping, each input matrix is not square.
        with self.assertRaises(ValueError):
            mx.eval(cpu_inv(a))

        # Vmap over the second axis.
        invs = mx.vmap(cpu_inv, in_axes=(1,))(a)

        for i in range(a.shape[1]):
            self.assertTrue(
                mx.allclose(a[:, i, :] @ invs[i], mx.eye(a.shape[0]), rtol=0, atol=1e-5)
            )


if __name__ == "__main__":
    unittest.main()
