# Copyright © 2024 Apple Inc.

import mlx.core as mx
import mlx.optimizers as optim
import mlx_distributed_tests
import mlx_tests
from mlx.nn.utils import average_gradients, fsdp_apply_gradients


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

    def test_fsdp_apply_gradients(self):
        world = mx.distributed.init()
        N = world.size()

        params = {
            "w1": mx.ones((N * 10, 8)),
            "w2": mx.ones((N * 20,)),
        }
        grads = {
            "w1": mx.ones((N * 10, 8)) * 0.1,
            "w2": mx.ones((N * 20,)) * 0.1,
        }

        optimizer = optim.SGD(learning_rate=0.1)
        updated_params_fsdp = fsdp_apply_gradients(grads, params, optimizer)
        mx.eval(updated_params_fsdp)

        self.assertEqual(updated_params_fsdp["w1"].shape, (N * 10, 8))
        self.assertEqual(updated_params_fsdp["w2"].shape, (N * 20,))

        self.assertTrue(
            mx.allclose(
                updated_params_fsdp["w1"], mx.ones((N * 10, 8)) * 0.99, atol=1e-6
            )
        )
        self.assertTrue(
            mx.allclose(updated_params_fsdp["w2"], mx.ones((N * 20,)) * 0.99, atol=1e-6)
        )

        grads = {
            "w1": mx.ones((N * 10, 8)) * 10.0,
            "w2": mx.ones((N * 20,)) * 10.0,
        }

        new_params_clipped, grad_norm = fsdp_apply_gradients(
            grads, params, optimizer, max_norm=1.0
        )
        mx.eval(new_params_clipped, grad_norm)

        self.assertIsNotNone(grad_norm)
        expected_norm = mx.sqrt((N * 10 * 8 + N * 20) * 100.0)
        self.assertTrue(mx.allclose(grad_norm, expected_norm, atol=1e-4, rtol=1e-4))
        self.assertEqual(new_params_clipped["w1"].shape, (N * 10, 8))
        self.assertEqual(new_params_clipped["w2"].shape, (N * 20,))

        scale = 1.0 / expected_norm
        expected_update = 1.0 - 0.1 * 10.0 * scale
        self.assertTrue(
            mx.allclose(
                new_params_clipped["w1"],
                mx.ones((N * 10, 8)) * expected_update,
                atol=1e-4,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            mx.allclose(
                new_params_clipped["w2"],
                mx.ones((N * 20,)) * expected_update,
                atol=1e-4,
                rtol=1e-4,
            )
        )
        params = {"w": mx.ones((N * 4,))}
        grads = {"w": mx.ones((N * 4,)) * 0.5}

        optimizer_fsdp = optim.SGD(learning_rate=0.1)
        updated_params_fsdp = fsdp_apply_gradients(grads, params, optimizer_fsdp)

        optimizer_ddp = optim.SGD(learning_rate=0.1)
        avg_grads = average_gradients(grads)
        updated_params_ddp = optimizer_ddp.apply_gradients(avg_grads, params)
        mx.eval(updated_params_ddp, updated_params_fsdp)

        self.assertTrue(
            mx.allclose(
                updated_params_fsdp["w"], updated_params_ddp["w"], atol=1e-6, rtol=1e-4
            ),
        )

    def test_fsdp_peak_memory(self):
        world = mx.distributed.init()
        N = world.size()
        mx.random.seed(42)
        params = {
            "w1": mx.random.normal((N * 1024, 1024)),
            "w2": mx.random.normal((N * 2048, 512)),
        }
        grads = {
            "w1": mx.random.normal((N * 1024, 1024)),
            "w2": mx.random.normal((N * 2048, 512)),
        }
        mx.eval(params, grads)
        optimizer_ddp = optim.Adam(learning_rate=0.01)
        optimizer_fsdp = optim.Adam(learning_rate=0.01)

        def pseudo_step_ddp(grads, params, optimizer):
            grads = average_gradients(grads)
            grads, grad_norm = optim.clip_grad_norm(grads, max_norm=1.0)
            params = optimizer.apply_gradients(grads, params)
            return grad_norm, params

        def pseudo_step_fsdp(grads, params, optimizer):
            params, grad_norm = fsdp_apply_gradients(
                grads, params, optimizer, max_norm=1.0
            )
            return grad_norm, params

        mx.reset_peak_memory()

        for i in range(10):
            grad_norm, params = pseudo_step_ddp(grads, params, optimizer_ddp)
            mx.eval(grad_norm, params)

        ddp_peak_memory = mx.get_peak_memory()
        mx.reset_peak_memory()

        for i in range(10):
            grad_norm, params = pseudo_step_fsdp(grads, params, optimizer_fsdp)
            mx.eval(grad_norm, params)

        fsdp_peak_memory = mx.get_peak_memory()
        self.assertTrue(fsdp_peak_memory < ddp_peak_memory)


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
