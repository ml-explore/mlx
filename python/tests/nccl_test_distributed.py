# Copyright © 2024 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
import mlx_distributed_tests
import mlx_tests
from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten, tree_map


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

    def test_all_reduce_split(self):
        world = mx.distributed.init()
        dtypes = [
            (mx.float32, 1e-6),
            (mx.float16, 5e-3),
            (mx.bfloat16, 1e-1),
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
                x = (
                    mx.random.uniform(shape=(group.size(),) + sh, key=key) * 10
                ).astype(dt)

                # All sum
                y = mx.distributed.all_sum(x[group.rank()], group=group)
                z = x.sum(0)
                maxrelerror = (y - z).abs()
                if rtol > 0:
                    maxrelerror /= z.abs()
                maxrelerror = maxrelerror.max()
                self.assertLessEqual(maxrelerror, rtol)

                # All max
                y = mx.distributed.all_max(x[group.rank()], group=group)
                z = x.max(0)
                self.assertTrue(mx.all(y == z))

                # All min
                y = mx.distributed.all_min(x[group.rank()], group=group)
                z = x.min(0)
                self.assertTrue(mx.all(y == z))

    def test_all_gather_split(self):
        world = mx.distributed.init()
        dtypes = [mx.float32, mx.float16, mx.bfloat16]
        sub = world.split(world.rank() % 2)
        for dt in dtypes:
            x = mx.ones((2, 2, 4), dtype=dt)
            y = mx.distributed.all_gather(x, group=sub)
            self.assertEqual(y.shape, (sub.size() * 2, 2, 4))
            self.assertTrue(mx.all(y == 1))

    def test_fully_shard_grads(self):
        dtypes = [
            (mx.float32, 1e-6, 1e-6),
            (mx.bfloat16, 1e-3, 1e-3),
        ]

        world = mx.distributed.init()
        N = world.size()
        rank = world.rank()
        dims = 8 * N
        part = slice(rank * dims // N, (rank + 1) * dims // N)

        class MLP(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.l1 = nn.Linear(dims, dims)
                self.l2 = nn.Linear(dims, dims)

            def __call__(self, x):
                return self.l2(nn.relu(self.l1(x)))

        def loss_fn(model, x, y):
            logits = model(x).astype(mx.float32)
            return ((logits - y) ** 2).mean()

        for dtype, atol, rtol in dtypes:

            mx.random.seed(0xF0F0F0F0)

            kx, ky = mx.random.split(mx.random.key(rank))
            x = mx.random.normal((4, dims), dtype=dtype, key=kx)
            y = mx.random.normal((4, dims), key=ky)

            # DDP reference: replicated params, gradients averaged across ranks
            model = MLP(dims)
            params = model.trainable_parameters()

            # Cast parameters to dtype for forward
            model.update(tree_map(lambda p: p.astype(dtype), params))
            loss, grads = mx.value_and_grad(loss_fn)(model, x, y)
            grads = average_gradients(grads, group=world)
            loss = mx.distributed.all_sum(loss, group=world) / N
            mx.eval(loss, grads)

            # Shard the model
            model_sharded = MLP(dims)
            model_sharded.update(params)
            model_sharded = nn.fully_shard(model_sharded, compute_dtype=dtype)
            loss_sharded, grads_sharded = mx.value_and_grad(loss_fn)(
                model_sharded, x, y
            )

            loss_sharded = mx.distributed.all_sum(loss_sharded, group=world) / N
            mx.eval(loss_sharded, grads_sharded)
            grads_ref = dict(tree_flatten(grads))
            self.assertTrue(mx.allclose(loss, loss_sharded, atol=1e-4, rtol=1e-4))
            for k, gs in tree_flatten(grads_sharded["module"]):
                self.assertTrue(
                    mx.allclose(gs, grads_ref[k][part], atol=atol, rtol=rtol)
                )


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
