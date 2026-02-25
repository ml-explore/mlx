# Copyright © 2024 Apple Inc.

import mlx.core as mx
import mlx_distributed_tests
import mlx_tests


class TestNCCLDistributed(mlx_distributed_tests.MLXDistributedCommonTestCase):
    @classmethod
    def setUpClass(cls):
        _ = mx.distributed.init(strict=True, backend="nccl")
        cls.atol = 1e-4
        cls.rtol = 1e-4

    def test_sum_scatter(self):

        world = mx.distributed.init()

        dtypes = [
            (mx.float32, 1e-6),
            (mx.float16, 5e-3),
            (mx.bfloat16, 1e-1),
        ]
        sizes = [
            (8,),
            (64,),
            (1024,),
            (1024, 1024),
        ]
        key = mx.random.key(world.rank())

        for dt, rtol in dtypes:
            for sh in sizes:
                x = (mx.random.uniform(shape=sh, key=key) * 10).astype(dt)  # shape=sh

                # Sum scatter
                y = mx.distributed.sum_scatter(x)  # shape=sh/world.size()
                z = mx.distributed.all_sum(x)  # shape=sh
                chunk = sh[0] // world.size()
                start = world.rank() * chunk
                stop = start + chunk
                z_ref = z[start:stop]

                maxrelerror = (y - z_ref).abs()
                if rtol > 0:
                    maxrelerror /= z_ref.abs()
                maxrelerror = maxrelerror.max()
                self.assertLessEqual(maxrelerror, rtol)

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

    def test_split(self):
        world = mx.distributed.init()

        sub = world.split(world.rank() % 2)
        self.assertEqual(sub.size(), world.size() // 2)
        self.assertEqual(sub.rank(), world.rank() // 2)

        x = mx.ones((2, 4), dtype=mx.float32)
        y = mx.distributed.all_gather(x, group=sub)
        self.assertEqual(y.shape, (sub.size() * 2, 4))
        self.assertTrue(mx.all(y == 1))

        pairs = world.split(world.rank() // 2)
        self.assertEqual(pairs.size(), 2)

        x = mx.ones((8,)) * world.rank()
        y = mx.distributed.all_sum(x, group=pairs)
        pair_base = (world.rank() // 2) * 2
        expected = pair_base + (pair_base + 1)
        self.assertTrue(mx.all(y == expected))


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
