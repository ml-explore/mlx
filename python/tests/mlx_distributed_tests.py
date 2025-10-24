# Copyright © 2025 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx_tests
from mlx.nn.layers.distributed import shard_inplace, shard_linear
from mlx.nn.utils import average_gradients


class MLXDistributedCommonTestCase(mlx_tests.MLXTestCase):
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

    def test_donation(self):
        x = mx.random.normal((1024,))
        mx.eval(x)
        mx.synchronize()

        mx.reset_peak_memory()
        scale = mx.array(2.0)
        y = mx.distributed.all_sum(x)
        mx.eval(y)
        mx.synchronize()
        all_sum_only = mx.get_peak_memory()
        y = mx.distributed.all_sum(x) * scale
        mx.eval(y)
        mx.synchronize()
        all_sum_with_binary = mx.get_peak_memory()

        self.assertEqual(all_sum_only, all_sum_with_binary)

    def test_shard_linear(self):
        # Seed the prng to have the same inputs and weights generated everywhere
        mx.random.seed(0xF0F0F0F0)

        # Prepare inputs
        world = mx.distributed.init()
        part = (
            slice(None),
            slice(
                world.rank() * 1024 // world.size(),
                (world.rank() + 1) * 1024 // world.size(),
            ),
        )
        x = mx.random.normal((4, 1024))

        # Create and shard some linear layers
        lin = nn.Linear(1024, 1024, bias=True)
        slin1 = shard_linear(lin, "all-to-sharded")
        slin2 = shard_linear(lin, "sharded-to-all")
        y = lin(x)
        y1 = slin1(x)
        y2 = slin2(x[part])
        self.assertTrue(mx.allclose(y, y2, atol=1e-6, rtol=1e-4))
        self.assertTrue(mx.allclose(y[part], y1))

        # And their quant versions
        qlin = lin.to_quantized()
        slin1 = shard_linear(qlin, "all-to-sharded")
        slin2 = shard_linear(qlin, "sharded-to-all")
        y = qlin(x)
        y1 = slin1(x)
        y2 = slin2(x[part])
        self.assertTrue(mx.allclose(y, y2, atol=1e-6, rtol=1e-4))
        self.assertTrue(mx.allclose(y[part], y1))

        # Check the backward works as expected
        def dummy_loss(model, x, y):
            return (model(x) * y).sum()

        mod = nn.Sequential(
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
        )
        smod = nn.Sequential(
            shard_linear(mod.layers[0], "all-to-sharded"),
            shard_linear(mod.layers[1], "sharded-to-all"),
            shard_linear(mod.layers[2], "all-to-sharded"),
            shard_linear(mod.layers[3], "sharded-to-all"),
        )

        grad1 = nn.value_and_grad(mod, dummy_loss)
        grad2 = nn.value_and_grad(smod, dummy_loss)

        x = mx.random.normal((4, 128))
        y = mx.random.normal((4, 128))

        l1, g1 = grad1(mod, x, y)
        l2, g2 = grad2(smod, x, y)
        mx.eval(l1, g1, l2, g2)

        part = slice(
            world.rank() * 128 // world.size(), (world.rank() + 1) * 128 // world.size()
        )
        self.assertTrue(mx.allclose(l1, l2))
        self.assertTrue(
            mx.allclose(
                g1["layers"][0]["weight"][part],
                g2["layers"][0]["weight"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][2]["weight"][part],
                g2["layers"][2]["weight"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][1]["weight"][:, part],
                g2["layers"][1]["weight"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][3]["weight"][:, part],
                g2["layers"][3]["weight"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][0]["bias"][part],
                g2["layers"][0]["bias"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][2]["bias"][part],
                g2["layers"][2]["bias"],
                atol=1e-6,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][1]["bias"], g2["layers"][1]["bias"], atol=1e-6, rtol=1e-4
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][3]["bias"], g2["layers"][3]["bias"], atol=1e-6, rtol=1e-4
            )
        )

    def test_shard_predicate(self):
        mx.random.seed(0xF0F0F0F0)

        class MyConv(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.aggregate = kwargs.pop("aggregate", False)
                self.conv = nn.Conv2d(*args, **kwargs)

            def __call__(self, x):
                x = self.conv(x)
                if self.aggregate:
                    x = mx.distributed.all_sum(x)
                return x

        def sharding(path, weight):
            parts = path.split(".")
            even = int(parts[1]) % 2 == 0
            if even:
                return 0
            else:
                return -1 if parts[-1] != "bias" else None

        mod = nn.Sequential(
            MyConv(3, 128, kernel_size=3),
            MyConv(128, 128, kernel_size=3),
            MyConv(128, 128, kernel_size=3),
            MyConv(128, 3, kernel_size=3),
        )
        smod = nn.Sequential(
            MyConv(3, 128, kernel_size=3),
            MyConv(128, 128, kernel_size=3, aggregate=True),
            MyConv(128, 128, kernel_size=3),
            MyConv(128, 3, kernel_size=3, aggregate=True),
        )
        smod.update(mod.parameters())
        shard_inplace(smod, sharding)

        x = mx.random.normal((4, 16, 16, 3))
        y1 = mod(x)
        y2 = smod(x)
        self.assertTrue(mx.allclose(y1, y2, atol=1e-6, rtol=1e-4))
