# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx_distributed_tests


class TestMPIDistributed(mlx_distributed_tests.MLXDistributedCommonTestCase):
    @classmethod
    def setUpClass(cls):
        world = mx.distributed.init(strict=True, backend="mpi")

    def test_groups(self):
        world = mx.distributed.init()
        self.assertEqual(world.size(), 8)
        self.assertTrue(0 <= world.rank() < 8)

        world2 = mx.distributed.init()
        self.assertEqual(world.size(), world2.size())
        self.assertEqual(world.rank(), world2.rank())

        sub = world.split(world.rank() % 2)
        self.assertEqual(sub.size(), 4)
        self.assertEqual(sub.rank(), world.rank() // 2)

        sub = world.split(world.rank() // 2)
        self.assertEqual(sub.size(), 2)

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
        group = world.split(world.rank() % 2)

        for dt, rtol in dtypes:
            for sh in sizes:
                for g in [world, group]:
                    x = (
                        mx.random.uniform(shape=(g.size(),) + sh, key=key) * 10
                    ).astype(dt)

                    # All sum
                    y = mx.distributed.all_sum(x[g.rank()], group=g)
                    z = x.sum(0)
                    maxrelerror = (y - z).abs()
                    if rtol > 0:
                        maxrelerror /= z.abs()
                    maxrelerror = maxrelerror.max()
                    self.assertLessEqual(maxrelerror, rtol)

                    # All max
                    y = mx.distributed.all_max(x[g.rank()], group=g)
                    z = x.max(0)
                    self.assertTrue(mx.all(y == z))

                    # All min
                    y = mx.distributed.all_min(x[g.rank()], group=g)
                    z = x.min(0)
                    self.assertTrue(mx.all(y == z))

    def test_all_gather(self):
        world = mx.distributed.init()
        dtypes = [
            mx.int8,
            mx.uint8,
            mx.int16,
            mx.uint16,
            mx.int32,
            mx.uint32,
            mx.float32,
            mx.complex64,
        ]
        for dt in dtypes:
            x = mx.ones((2, 2, 4), dtype=dt)
            y = mx.distributed.all_gather(x)
            self.assertEqual(y.shape, (world.size() * 2, 2, 4))
            self.assertTrue(mx.all(y == 1))

        sub = world.split(world.rank() % 2)
        for dt in dtypes:
            x = mx.ones((2, 2, 4), dtype=dt)
            y = mx.distributed.all_gather(x, group=sub)
            self.assertEqual(y.shape, (sub.size() * 2, 2, 4))
            self.assertTrue(mx.all(y == 1))

    def test_mixed(self):
        # Make the following groups:
        # - world: 0 1 2 3 4 5 6 7
        # - sub_1: 0 1 0 1 0 1 0 1
        # - sub_2: 0 0 1 1 2 2 3 3
        #
        # The corresponding colors to make them are
        # - world: N/A
        # - sub_1: 0 0 1 1 2 2 3 3
        # - sub_2: 0 1 0 1 0 1 0 1

        world = mx.distributed.init()
        sub_1 = world.split(world.rank() // 2)
        sub_2 = world.split(world.rank() % 2)

        x = mx.ones((1, 8)) * world.rank()
        y = mx.distributed.all_sum(x, group=sub_1)
        z = mx.distributed.all_gather(y, group=sub_2)
        z_target = mx.arange(8).reshape(4, 2).sum(-1, keepdims=True)

        self.assertTrue(mx.all(z == z_target))

    def test_send_recv(self):
        world = mx.distributed.init()
        pairs = world.split(world.rank() // 2)
        neighbor = (pairs.rank() + 1) % 2
        send = pairs.rank() == 0

        x = mx.ones(10)
        for i in range(10):
            if send:
                mx.eval(mx.distributed.send(2 * x, neighbor, group=pairs))
            else:
                x = mx.distributed.recv_like(x, neighbor, group=pairs)
                mx.eval(x)
            send = not send

        self.assertTrue(mx.all(x == (1024 if pairs.rank() == 0 else 512)))

        # Check recv and computation in same eval:
        y = mx.ones((5, 5)) + mx.array(2.0)
        if send:
            x = mx.distributed.send(2 * x, neighbor, group=pairs)
        else:
            x = mx.distributed.recv_like(x, neighbor, group=pairs)
        mx.eval(y, x)


if __name__ == "__main__":
    unittest.main()
