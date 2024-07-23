# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestEinsum(mlx_tests.MLXTestCase):

    def test_simple_path(self):
        a = mx.zeros((5, 5))
        path = mx.einsum_path("ii", a)
        self.assertEqual(path[0], [(0,)])

        path = mx.einsum_path("ij->i", a)
        self.assertEqual(path[0], [(0,)])

        path = mx.einsum_path("ii->i", a)
        self.assertEqual(path[0], [(0,)])

        a = mx.zeros((5, 8))
        b = mx.zeros((8, 3))
        path = mx.einsum_path("ij,jk", a, b)
        self.assertEqual(path[0], [(0, 1)])
        path = mx.einsum_path("ij,jk -> ijk", a, b)
        self.assertEqual(path[0], [(0, 1)])

        a = mx.zeros((5, 8))
        b = mx.zeros((8, 3))
        c = mx.zeros((3, 7))
        path = mx.einsum_path("ij,jk,kl", a, b, c)

        self.assertEqual(path[0], [(0, 1), (0, 1)])

        a = mx.zeros((5, 8))
        b = mx.zeros((8, 10))
        c = mx.zeros((10, 7))
        path = mx.einsum_path("ij,jk,kl", a, b, c)
        self.assertEqual(path[0], [(1, 2), (0, 1)])

    def test_longer_paths(self):
        chars = "abcdefghijklmopqABC"
        sizes = [2, 3, 4, 5, 4, 3, 2, 6, 5, 4, 3, 2, 5, 7, 4, 3, 2, 3, 4]
        dim_dict = {c: s for c, s in zip(chars, sizes)}
        cases = [
            "eb,cb,fb->cef",
            "dd,fb,be,cdb->cef",
            "dd,fb,be,cdb->cef",
            "bca,cdb,dbf,afc->",
            "dcc,fce,ea,dbf->ab",
            "dcc,fce,ea,dbf->ab",
        ]

        for case in cases:
            subscripts = case[: case.find("->")].split(",")
            inputs = []
            for s in subscripts:
                shape = [dim_dict[c] for c in s]
                inputs.append(np.ones(shape))
            np_path = np.einsum_path(case, *inputs)

            inputs = [mx.array(i) for i in inputs]
            mx_path = mx.einsum_path(case, *inputs)
            self.assertEqual(np_path[0][1:], mx_path[0])

    def test_simple_einsum(self):
        a = mx.arange(4 * 4).reshape(4, 4)
        a_mx = mx.einsum("ii->i", a)
        a_np = np.einsum("ii->i", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 2 * 2).reshape(2, 2, 2)
        a_mx = mx.einsum("iii->i", a)
        a_np = np.einsum("iii->i", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 2 * 3 * 3).reshape(2, 2, 3, 3)
        a_mx = mx.einsum("iijj->ij", a)
        a_np = np.einsum("iijj->ij", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 2 * 3 * 3).reshape(2, 3, 2, 3)
        a_mx = mx.einsum("ijij->ij", a)
        a_np = np.einsum("ijij->ij", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        # Test some simple reductions
        a = mx.arange(2 * 2).reshape(2, 2)
        a_mx = mx.einsum("ii", a)
        a_np = np.einsum("ii", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 4).reshape(2, 4)
        a_mx = mx.einsum("ij->", a)
        a_np = np.einsum("ij->", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 4).reshape(2, 4)
        a_mx = mx.einsum("ij->i", a)
        a_np = np.einsum("ij->i", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 4).reshape(2, 4)
        a_mx = mx.einsum("ij->j", a)
        a_np = np.einsum("ij->j", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 2 * 2).reshape(2, 2, 2)
        a_mx = mx.einsum("iii->", a)
        a_np = np.einsum("iii->", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 2 * 3 * 3).reshape(2, 3, 2, 3)
        a_mx = mx.einsum("ijij->j", a)
        a_np = np.einsum("ijij->j", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        # Test some simple transposes
        a = mx.arange(2 * 4).reshape(2, 4)
        a_mx = mx.einsum("ij", a)
        a_np = np.einsum("ij", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 4).reshape(2, 4)
        a_mx = mx.einsum("ij->ji", a)
        a_np = np.einsum("ij->ji", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 3 * 4).reshape(2, 3, 4)
        a_mx = mx.einsum("ijk->jki", a)
        a_np = np.einsum("ijk->jki", a)
        self.assertTrue(np.array_equal(a_mx, a_np))

    def test_two_input_einsum(self):

        # Matmul
        a = mx.full((2, 8), 1.0)
        b = mx.full((8, 2), 1.0)
        a_mx = mx.einsum("ik,kj", a, b)
        a_np = np.einsum("ik,kj", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

        # Matmul + transpose
        a = mx.full((2, 8), 1.0)
        b = mx.full((8, 3), 1.0)
        a_mx = mx.einsum("ik,kj->ji", a, b)
        a_np = np.einsum("ik,kj->ji", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

        # Inner product
        a = mx.full((4,), 1.0)
        b = mx.full((4,), 1.0)
        a_mx = mx.einsum("i,i", a, b)
        a_np = np.einsum("i,i", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

        # Outer product
        a = mx.full((4,), 0.5)
        b = mx.full((6,), 2.0)
        a_mx = mx.einsum("i,j->ij", a, b)
        a_np = np.einsum("i,j->ij", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

        # Elementwise multiply
        a = mx.full((2, 8), 1.0)
        b = mx.full((2, 8), 1.0)
        a_mx = mx.einsum("ij,ij->ij", a, b)
        a_np = np.einsum("ij,ij->ij", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

        # Medley
        a = mx.full((2, 8, 3, 5), 1.0)
        b = mx.full((3, 7, 5, 2), 1.0)
        a_mx = mx.einsum("abcd,fgda->bfca", a, b)
        a_np = np.einsum("abcd,fgda->bfca", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

    def test_sum_first(self):
        a = mx.full((5, 8), 1.0)
        b = mx.full((8, 2), 1.0)
        a_mx = mx.einsum("ab,bc->c", a, b)
        a_np = np.einsum("ab,bc->c", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

    def test_broadcasting(self):
        a = mx.full((5, 1), 1.0)
        b = mx.full((8, 2), 1.0)
        a_mx = mx.einsum("ab,bc->c", a, b)
        return
        a_np = np.einsum("ab,bc->c", a, b)
        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.random.uniform(shape=(5, 1, 3, 1))
        b = mx.random.uniform(shape=(1, 7, 1, 2))
        a_mx = mx.einsum("abcd,cdab->abcd", a, b)
        a_np = np.einsum("abcd,cdab->abcd", a, b)
        self.assertTrue(np.allclose(a_mx, a_np))

    def test_attention(self):
        q = mx.random.uniform(shape=(2, 3, 4, 5))
        k = mx.random.uniform(shape=(2, 3, 4, 5))
        v = mx.random.uniform(shape=(2, 3, 4, 5))

        s = mx.einsum("itjk,iujk->ijtu", q, k)
        out_mx = mx.einsum("ijtu,iujk->itjk", s, v)

        s = np.einsum("itjk,iujk->ijtu", q, k)
        out_np = np.einsum("ijtu,iujk->itjk", s, v)

        self.assertTrue(np.allclose(out_mx, out_np))

    def test_multi_input_einsum(self):
        a = mx.ones((3, 4, 5))
        out_mx = mx.einsum("ijk,lmk,ijf->lf", a, a, a)
        out_np = np.einsum("ijk,lmk,ijf->lf", a, a, a)
        self.assertTrue(np.allclose(out_mx, out_np))


if __name__ == "__main__":
    unittest.main()
