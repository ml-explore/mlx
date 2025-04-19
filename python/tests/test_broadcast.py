# Copyright © 2023 Apple Inc.

import mlx.utils
import mlx_tests


class TestBroadcast(mlx_tests.MLXTestCase):
    def test_broadcast_shapes(self):
        # Basic broadcasting
        self.assertEqual(mlx.utils.broadcast_shapes((1, 2, 3), (3,)), (1, 2, 3))
        self.assertEqual(mlx.utils.broadcast_shapes((4, 1, 6), (5, 6)), (4, 5, 6))
        self.assertEqual(mlx.utils.broadcast_shapes((5, 1, 4), (1, 3, 4)), (5, 3, 4))

        # Multiple arguments
        self.assertEqual(mlx.utils.broadcast_shapes((1, 1), (1, 8), (7, 1)), (7, 8))
        self.assertEqual(
            mlx.utils.broadcast_shapes((6, 1, 5), (1, 7, 1), (6, 7, 5)), (6, 7, 5)
        )

        # Same shapes
        self.assertEqual(mlx.utils.broadcast_shapes((3, 4, 5), (3, 4, 5)), (3, 4, 5))

        # Single argument
        self.assertEqual(mlx.utils.broadcast_shapes((2, 3)), (2, 3))

        # Empty shapes
        self.assertEqual(mlx.utils.broadcast_shapes((), ()), ())
        self.assertEqual(mlx.utils.broadcast_shapes((), (1,)), (1,))
        self.assertEqual(mlx.utils.broadcast_shapes((1,), ()), (1,))

        # Broadcasting with zeroes
        self.assertEqual(mlx.utils.broadcast_shapes((0,), (0,)), (0,))
        self.assertEqual(mlx.utils.broadcast_shapes((1, 0, 5), (3, 1, 5)), (3, 0, 5))
        self.assertEqual(mlx.utils.broadcast_shapes((5, 0), (0, 5, 0)), (0, 5, 0))

        # Error cases
        with self.assertRaises(ValueError):
            mlx.utils.broadcast_shapes((3, 4), (4, 3))

        with self.assertRaises(ValueError):
            mlx.utils.broadcast_shapes((2, 3, 4), (2, 5, 4))

        with self.assertRaises(ValueError):
            mlx.utils.broadcast_shapes()
