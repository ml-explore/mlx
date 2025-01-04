# Copyright © 2024 Apple Inc.

import unittest

import mlx.core as mx
import mlx_tests
from mlx.nn.utils import average_gradients


class TestDistributed(mlx_tests.MLXTestCase):
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
            y = mx.distributed.all_sum(x)
            self.assertTrue(mx.all(y == world.size()))

        sub = world.split(world.rank() % 2)
        for dt in dtypes:
            x = mx.ones((2, 2, 4), dtype=dt)
            y = mx.distributed.all_sum(x, group=sub)
            self.assertTrue(mx.all(y == sub.size()))

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

    def test_average_gradients(self):
        original_all_sum = mx.distributed.all_sum
        n_calls = 0
        xtype = None

        def new_all_sum(x, **kwargs):
            nonlocal n_calls
            nonlocal xtype

            n_calls += 1
            if xtype is not None:
                self.assertEqual(xtype, x.dtype)

            return original_all_sum(x, **kwargs)

        mx.distributed.all_sum = new_all_sum

        try:
            grads = [mx.ones(10) for i in range(10)]
            new_grads = average_gradients(grads)
            mx.eval(new_grads)
            self.assertEqual(len(new_grads), 10)
            self.assertTrue(all(mx.all(g == 1) for g in new_grads))
            self.assertEqual(n_calls, 1)

            n_calls = 0
            new_grads = average_gradients(grads, all_reduce_size=4 * 50)
            mx.eval(new_grads)
            self.assertEqual(len(new_grads), 10)
            self.assertTrue(all(mx.all(g == 1) for g in new_grads))
            self.assertEqual(n_calls, 2)

            n_calls = 0
            new_grads = average_gradients(grads, all_reduce_size=0)
            mx.eval(new_grads)
            self.assertEqual(len(new_grads), 10)
            self.assertTrue(all(mx.all(g == 1) for g in new_grads))
            self.assertEqual(n_calls, 10)

            n_calls = 0
            xtype = mx.float16
            new_grads = average_gradients(
                grads, all_reduce_size=2 * 50, communication_type=mx.float16
            )
            mx.eval(new_grads)
            self.assertEqual(len(new_grads), 10)
            self.assertTrue(all(g.dtype == mx.float32 for g in new_grads))
            self.assertTrue(all(mx.all(g == 1) for g in new_grads))
            self.assertEqual(n_calls, 2)

        finally:
            mx.distributed.all_sum = original_all_sum


if __name__ == "__main__":
    unittest.main()
