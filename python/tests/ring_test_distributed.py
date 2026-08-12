# Copyright © 2024 Apple Inc.

import mlx.core as mx
import mlx_distributed_tests
import mlx_tests


class TestRingDistributed(mlx_distributed_tests.MLXDistributedCommonTestCase):
    @classmethod
    def setUpClass(cls):
        _ = mx.distributed.init(strict=True, backend="ring")
        cls.atol = 1e-6
        cls.rtol = 1e-4

    def test_groups(self):
        world = mx.distributed.init()
        self.assertTrue(0 <= world.rank() < world.size())

        world2 = mx.distributed.init()
        self.assertEqual(world.size(), world2.size())
        self.assertEqual(world.rank(), world2.rank())

        n = world.size()
        r = world.rank()

        sub = world.split(r % 2)
        members = [x for x in range(n) if x % 2 == r % 2]
        self.assertEqual(sub.size(), len(members))
        self.assertEqual(sub.rank(), members.index(r))
        self.assertEqual(
            mx.distributed.all_sum(mx.array(r), group=sub).item(), sum(members)
        )

        # The key decides the order within the subgroup.
        sub = world.split(r % 2, key=n - r)
        self.assertEqual(sub.rank(), list(reversed(members)).index(r))

        # A color nobody shares gives a group of one.
        solo = world.split(r)
        self.assertEqual(solo.size(), 1)
        self.assertEqual(solo.rank(), 0)

        # Subgroups split again.
        half = world.split(r < n // 2)
        members = [x for x in range(n) if (x < n // 2) == (r < n // 2)]
        self.assertEqual(half.size(), len(members))
        self.assertEqual(
            mx.distributed.all_sum(mx.array(r), group=half).item(), sum(members)
        )

    def test_all_reduce_extra(self):
        world = mx.distributed.init()
        dtypes = [
            (mx.int16, 0),
            (mx.uint16, 0),
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

                # All sum
                y = mx.distributed.all_sum(x[world.rank()])
                z = x.sum(0)
                maxrelerror = (y - z).abs()
                if rtol > 0:
                    maxrelerror /= z.abs()
                maxrelerror = maxrelerror.max()
                self.assertLessEqual(maxrelerror, rtol)

                # All max
                y = mx.distributed.all_max(x[world.rank()])
                z = x.max(0)
                self.assertTrue(mx.all(y == z))

                # All min
                y = mx.distributed.all_min(x[world.rank()])
                z = x.min(0)
                self.assertTrue(mx.all(y == z))

    def test_all_gather_extra(self):
        world = mx.distributed.init()
        dtypes = [
            mx.int16,
            mx.uint16,
            mx.complex64,
        ]
        for dt in dtypes:
            x = mx.ones((2, 2, 4), dtype=dt)
            y = mx.distributed.all_gather(x)
            self.assertEqual(y.shape, (world.size() * 2, 2, 4))
            self.assertTrue(mx.all(y == 1))

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

    def test_all_gather_vjp(self):
        def fun(x):
            return mx.distributed.all_gather(x)[0]

        dfdx = mx.grad(fun)(mx.array(1.0))
        if mx.distributed.init().rank() == 0:
            self.assertEqual(dfdx.item(), 1.0)
        else:
            self.assertEqual(dfdx.item(), 0.0)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
