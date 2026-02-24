# Copyright © 2024 Apple Inc.

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx_distributed_tests
import mlx_tests
from mlx.nn.utils import (
    all_gather_parameters,
    average_gradients,
    fsdp_update_parameters,
    reduce_scatter_gradients,
)


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

    def test_reduce_scatter_gradients(self):
        original_sum_scatter = mx.distributed.sum_scatter
        n_calls = 0
        xtype = None
        world = mx.distributed.init()
        N = world.size()

        def new_sum_scatter(x, **kwargs):
            nonlocal n_calls
            nonlocal xtype

            n_calls += 1
            if xtype is not None:
                self.assertEqual(xtype, x.dtype)

            return original_sum_scatter(x, **kwargs)

        mx.distributed.sum_scatter = new_sum_scatter

        try:
            grads = [mx.ones((N * 10,)) for i in range(10)]
            new_grads = reduce_scatter_gradients(grads)
            mx.eval(new_grads)
            self.assertEqual(len(new_grads), 10)
            self.assertTrue(all(g.shape[0] == 10 for g in new_grads))
            self.assertTrue(all(mx.all(g == 1) for g in new_grads))
            self.assertEqual(n_calls, 1)

            n_calls = 0
            new_grads = reduce_scatter_gradients(grads, reduce_scatter_size=4 * N * 50)
            mx.eval(new_grads)
            self.assertEqual(len(new_grads), 10)
            self.assertTrue(all(g.shape[0] == 10 for g in new_grads))
            self.assertTrue(all(mx.all(g == 1) for g in new_grads))
            self.assertEqual(n_calls, 2)

            n_calls = 0
            new_grads = reduce_scatter_gradients(grads, reduce_scatter_size=0)
            mx.eval(new_grads)
            self.assertEqual(len(new_grads), 10)
            self.assertTrue(all(g.shape[0] == 10 for g in new_grads))
            self.assertTrue(all(mx.all(g == 1) for g in new_grads))
            self.assertEqual(n_calls, 10)

            n_calls = 0
            xtype = mx.float16
            new_grads = reduce_scatter_gradients(
                grads, reduce_scatter_size=4 * N * 25, communication_type=mx.float16
            )
            mx.eval(new_grads)
            self.assertEqual(len(new_grads), 10)
            self.assertTrue(all(g.dtype == mx.float32 for g in new_grads))
            self.assertTrue(all(g.shape[0] == 10 for g in new_grads))
            self.assertTrue(all(mx.all(g == 1) for g in new_grads))
            self.assertEqual(n_calls, 2)

        finally:
            mx.distributed.sum_scatter = original_sum_scatter

    def test_all_gather_parameters(self):
        original_all_gather = mx.distributed.all_gather
        n_calls = 0
        xtype = None
        world = mx.distributed.init()
        N = world.size()

        def new_all_gather(x, **kwargs):
            nonlocal n_calls
            nonlocal xtype

            n_calls += 1
            if xtype is not None:
                self.assertEqual(xtype, x.dtype)

            return original_all_gather(x, **kwargs)

        mx.distributed.all_gather = new_all_gather

        try:
            params = [mx.ones((10,)) for i in range(10)]
            new_params = all_gather_parameters(params)
            mx.eval(new_params)
            self.assertEqual(len(new_params), 10)
            self.assertTrue(all(p.shape[0] == N * 10 for p in new_params))
            self.assertTrue(all(mx.all(p == 1) for p in new_params))
            self.assertEqual(n_calls, 1)

            n_calls = 0
            new_params = all_gather_parameters(params, all_gather_size=4 * 50)
            mx.eval(new_params)
            self.assertEqual(len(new_params), 10)
            self.assertTrue(all(p.shape[0] == N * 10 for p in new_params))
            self.assertTrue(all(mx.all(p == 1) for p in new_params))
            self.assertEqual(n_calls, 2)

            n_calls = 0
            new_params = all_gather_parameters(params, all_gather_size=0)
            mx.eval(new_params)
            self.assertEqual(len(new_params), 10)
            self.assertTrue(all(p.shape[0] == N * 10 for p in new_params))
            self.assertTrue(all(mx.all(p == 1) for p in new_params))
            self.assertEqual(n_calls, 10)

            n_calls = 0
            xtype = mx.float16
            new_params = all_gather_parameters(
                params, all_gather_size=4 * 25, communication_type=mx.float16
            )
            mx.eval(new_params)
            self.assertEqual(len(new_params), 10)
            self.assertTrue(all(p.dtype == mx.float32 for p in new_params))
            self.assertTrue(all(p.shape[0] == N * 10 for p in new_params))
            self.assertTrue(all(mx.all(p == 1) for p in new_params))
            self.assertEqual(n_calls, 2)

        finally:
            mx.distributed.all_gather = original_all_gather

    def test_fsdp(self):
        world = mx.distributed.init()
        N = world.size()

        tests = [
            [mx.random.normal((N * 10,)) for _ in range(10)],
            [mx.random.normal((N * 7, 8)) for _ in range(5)],
            {"a": mx.random.normal((N * 20,)), "b": mx.random.normal((N * 15, 4))},
        ]

        for grads in tests:
            averaged = average_gradients(grads)
            scattered = reduce_scatter_gradients(grads)
            gathered = all_gather_parameters(scattered)
            mx.eval(averaged, gathered)

            if isinstance(grads, list):
                self.assertEqual(len(averaged), len(gathered))
                for a, g in zip(averaged, gathered):
                    self.assertTrue(mx.allclose(a, g, atol=1e-6, rtol=1e-4))
            else:
                for key in grads.keys():
                    self.assertTrue(
                        mx.allclose(averaged[key], gathered[key], atol=1e-6, rtol=1e-4)
                    )

    def test_fsdp_update_parameters(self):
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
        updated_params_fsdp = fsdp_update_parameters(params, grads, optimizer)
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

        new_params_clipped, grad_norm = fsdp_update_parameters(
            params, grads, optimizer, max_norm=1.0
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
        updated_params_fsdp = fsdp_update_parameters(params, grads, optimizer_fsdp)

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
            params, grad_norm = fsdp_update_parameters(
                params, grads, optimizer, max_norm=1.0
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
