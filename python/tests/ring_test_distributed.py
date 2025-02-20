# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests


class TestRingDistributed(mlx_tests.MLXTestCase):
    @classmethod
    def setUpClass(cls):
        world = mx.distributed.init(strict=True, backend="ring")

    def test_groups(self):
        world = mx.distributed.init()
        self.assertEqual(world.size(), 8)
        self.assertTrue(0 <= world.rank() < 8)

        world2 = mx.distributed.init()
        self.assertEqual(world.size(), world2.size())
        self.assertEqual(world.rank(), world2.rank())

        with self.assertRaises(RuntimeError):
            sub = world.split(world.rank() % 2)

    def test_all_reduce(self):
        world = mx.distributed.init()
        dtypes = [
            (mx.int8, 0),
            (mx.uint8, 0),
            (mx.int16, 0),
            (mx.uint16, 0),
            (mx.int32, 0),
            (mx.uint32, 0),
            (mx.float32, 1e-6),
            (mx.float16, 5e-3),
            (mx.bfloat16, 1e-1),
            (mx.complex64, 1e-6),
        ]
        sizes = [
            (7,),
            (10,),
            (1024,),
            (1024, 1024),
        ]
        key = mx.random.key(0)
        for dt, rtol in dtypes:
            for sh in sizes:
                x = (
                    mx.random.uniform(shape=(world.size(),) + sh, key=key) * 10
                ).astype(dt)
                y = mx.distributed.all_sum(x[world.rank()])
                z = sum(
                    x[i] for i in range(world.size())
                )  # to ensure that we don't sum to int32
                maxrelerror = ((y - z).abs() / z.abs()).max()
                self.assertLessEqual(maxrelerror, rtol)

    def test_send_recv(self):
        world = mx.distributed.init()
        dtypes = [
            mx.int8,
            mx.uint8,
            mx.int16,
            mx.uint16,
            mx.int32,
            mx.uint32,
            mx.float32,
            mx.float16,
            mx.bfloat16,
            mx.complex64,
        ]
        sizes = [
            (7,),
            (10,),
            (1024,),
            (1024, 1024),
        ]
        key = mx.random.key(0)
        right = (world.rank() + 1) % world.size()
        left = (world.rank() + world.size() - 1) % world.size()
        for dt in dtypes:
            for sh in sizes:
                x = (
                    mx.random.uniform(shape=(world.size(),) + sh, key=key) * 10
                ).astype(dt)
                if world.rank() % 2 == 0:
                    y = mx.distributed.send(x[world.rank()], right)
                    z = mx.distributed.recv_like(y, left)
                    mx.eval(y, z)
                else:
                    z = mx.distributed.recv_like(x[world.rank()], left)
                    y = mx.distributed.send(x[world.rank()], right)
                    mx.eval(z, y)
                self.assertTrue(mx.all(y == x[world.rank()]))
                self.assertTrue(mx.all(z == x[left]))


if __name__ == "__main__":
    unittest.main()
