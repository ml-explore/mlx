# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests
import numpy as np


class TestEinsum(mlx_tests.MLXTestCase):

    def test_simple_path(self):
        a = mx.zeros((5, 5))
        path = mx.einsum_path("ii", a)
        self.assertEqual(path[0], [[0]])

        path = mx.einsum_path("ij->i", a)
        self.assertEqual(path[0], [[0]])

        path = mx.einsum_path("ii->i", a)
        self.assertEqual(path[0], [[0]])

        a = mx.zeros((5, 8))
        b = mx.zeros((8, 3))
        path = mx.einsum_path("ij,jk", a, b)
        self.assertEqual(path[0], [[0, 1]])
        path = mx.einsum_path("ij,jk -> ijk", a, b)
        self.assertEqual(path[0], [[0, 1]])

        a = mx.zeros((5, 8))
        b = mx.zeros((8, 3))
        c = mx.zeros((3, 7))
        path = mx.einsum_path("ij,jk,kl", a, b, c)

        self.assertEqual(path[0], [[0, 1], [0, 1]])

        a = mx.zeros((5, 8))
        b = mx.zeros((8, 10))
        c = mx.zeros((10, 7))
        path = mx.einsum_path("ij,jk,kl", a, b, c)
        self.assertEqual(path[0], [[1, 2], [0, 1]])

    def test_long_greedy_path(self):
        pass

    def test_einsum(self):
        #        a = mx.arange(4 * 4).reshape(4, 4)
        #        a_mx = mx.einsum("ii->i", a)
        #        a_np = np.einsum("ii->i", a)
        #        self.assertTrue(np.array_equal(a_mx, a_np))
        #
        #        a = mx.arange(2 * 2 * 2).reshape(2, 2, 2)
        #        a_mx = mx.einsum("iii->i", a)
        #        a_np = np.einsum("iii->i", a)
        #        self.assertTrue(np.array_equal(a_mx, a_np))
        #
        #        a = mx.arange(2 * 2 * 3 * 3).reshape(2, 2, 3, 3)
        #        a_mx = mx.einsum("iijj->ij", a)
        #        a_np = np.einsum("iijj->ij", a)
        #        self.assertTrue(np.array_equal(a_mx, a_np))
        #
        #        a = mx.arange(2 * 2 * 3 * 3).reshape(2, 3, 2, 3)
        #        a_mx = mx.einsum("ijij->ij", a)
        #        a_np = np.einsum("ijij->ij", a)
        #        self.assertTrue(np.array_equal(a_mx, a_np))
        #
        #        a = mx.arange(2 * 2 * 2).reshape(2, 2, 2)
        #        a_mx = mx.einsum("iii->", a)
        #        a_np = np.einsum("iii->", a)
        #        self.assertTrue(np.array_equal(a_mx, a_np))

        a = mx.arange(2 * 2 * 3 * 3).reshape(2, 3, 2, 3)
        a_mx = mx.einsum("ijij->j", a)
        a_np = np.einsum("ijij->j", a)
        self.assertTrue(np.array_equal(a_mx, a_np))


#        self.assertCmpNumpy(["jki", mx.full((2, 3, 4), 3.0)], mx.einsum, np.einsum)
#        self.assertCmpNumpy(
#            [
#                "ij,jk->ik",
#                mx.full(
#                    (
#                        2,
#                        2,
#                    ),
#                    2.0,
#                ),
#                mx.full(
#                    (
#                        2,
#                        2,
#                    ),
#                    3.0,
#                ),
#            ],
#            mx.einsum,
#            np.einsum,
#        )
#        self.assertCmpNumpy(
#            ["i,j->ij", mx.full((10,), 15.0), mx.full((10,), 20.0)],
#            mx.einsum,
#            np.einsum,
#        )
#        self.assertCmpNumpy(
#            ["i,i->", mx.full((10,), 15.0), mx.full((10,), 20.0)], mx.einsum, np.einsum
#        )
#        self.assertCmpNumpy(
#            [
#                "ijkl,mlopq->ikmop",
#                mx.full((4, 5, 9, 4), 20.0),
#                mx.full((14, 4, 16, 7, 5), 10.0),
#            ],
#            mx.einsum,
#            np.einsum,
#        )

if __name__ == "__main__":
    unittest.main()
