# Copyright © 2025 Apple Inc.

import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx_tests
from mlx.nn.layers.distributed import shard_inplace, shard_linear
from mlx.nn.layers.moe import (
    MixtureOfExperts,
    expert_combine,
    expert_dispatch,
)
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

    def test_all_reduce(self):
        g = mx.distributed.init()
        dtypes = [
            (mx.int8, 0),
            (mx.uint8, 0),
            (mx.int32, 0),
            (mx.uint32, 0),
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

        for dt, rtol in dtypes:
            for sh in sizes:
                x = (mx.random.uniform(shape=(g.size(),) + sh, key=key) * 10).astype(dt)

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
        self.assertTrue(mx.allclose(y, y2, atol=self.atol, rtol=self.rtol))
        self.assertTrue(mx.allclose(y[part], y1, atol=self.atol, rtol=self.rtol))

        # And their quant versions (QuantizedMatmul is not supported on CUDA)
        if not mx.cuda.is_available():
            qlin = lin.to_quantized()
            slin1 = shard_linear(qlin, "all-to-sharded")
            slin2 = shard_linear(qlin, "sharded-to-all")
            y = qlin(x)
            y1 = slin1(x)
            y2 = slin2(x[part])
            self.assertTrue(mx.allclose(y, y2, atol=self.atol, rtol=self.rtol))
            self.assertTrue(mx.allclose(y[part], y1))

            # Test non-affine quantization modes (mxfp8)
            qlin_mxfp8 = lin.to_quantized(group_size=32, bits=8, mode="mxfp8")
            self.assertEqual(qlin_mxfp8.mode, "mxfp8")

            slin1_mxfp8 = shard_linear(qlin_mxfp8, "all-to-sharded")
            slin2_mxfp8 = shard_linear(qlin_mxfp8, "sharded-to-all")

            # Verify mode is propagated
            self.assertEqual(slin1_mxfp8.mode, "mxfp8")
            self.assertEqual(slin2_mxfp8.mode, "mxfp8")

            # Verify biases parameter is not set for mxfp8
            self.assertIsNone(slin1_mxfp8.get("biases"))
            self.assertIsNone(slin2_mxfp8.get("biases"))

            y = qlin_mxfp8(x)
            y1 = slin1_mxfp8(x)
            y2 = slin2_mxfp8(x[part])
            self.assertTrue(mx.allclose(y, y2, atol=self.atol, rtol=self.rtol))
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
                g1["layers"][1]["bias"],
                g2["layers"][1]["bias"],
                atol=self.atol,
                rtol=self.rtol,
            )
        )
        self.assertTrue(
            mx.allclose(
                g1["layers"][3]["bias"],
                g2["layers"][3]["bias"],
                atol=self.atol,
                rtol=self.rtol,
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

    def _skip_if_all_to_all_unsupported(self, group):
        try:
            test = mx.distributed.all_to_all(mx.zeros((group.size(),)), group=group)
            mx.eval(test)
        except RuntimeError:
            self.skipTest("all_to_all not supported on this backend")

    def test_all_to_all(self):
        group = mx.distributed.init()
        self._skip_if_all_to_all_unsupported(group)
        world_size = group.size()
        rank = group.rank()

        # Test multiple dtypes
        dtypes = [mx.float32, mx.float16, mx.bfloat16, mx.int32]

        for dt in dtypes:
            # Create a [world_size * 4, 8] tensor with rank-specific values
            rows = world_size * 4
            cols = 8
            x = (mx.ones((rows, cols), dtype=dt) * (rank * 100)) + mx.broadcast_to(
                mx.arange(rows).reshape(-1, 1), (rows, cols)
            ).astype(dt)

            y = mx.distributed.all_to_all(x, group=group)
            mx.eval(y)

            # Output shape should equal input shape
            self.assertEqual(y.shape, x.shape)

            if world_size == 1:
                # For single process: all_to_all is identity
                self.assertTrue(mx.array_equal(y, x).item())
            else:
                # For multi-process: verify the all-to-all permutation
                # Each rank's output chunk i should come from rank i's input chunk rank
                chunk_size = rows // world_size
                for src_rank in range(world_size):
                    out_chunk = y[src_rank * chunk_size : (src_rank + 1) * chunk_size]
                    # This chunk should be what src_rank sent to us (our rank-th chunk of src_rank's input)
                    expected_vals = (
                        mx.ones((chunk_size, cols), dtype=dt) * (src_rank * 100)
                    ) + mx.broadcast_to(
                        mx.arange(rank * chunk_size, (rank + 1) * chunk_size).reshape(
                            -1, 1
                        ),
                        (chunk_size, cols),
                    ).astype(
                        dt
                    )
                    self.assertTrue(mx.array_equal(out_chunk, expected_vals).item())

    def test_all_to_all_sizes(self):
        group = mx.distributed.init()
        self._skip_if_all_to_all_unsupported(group)
        world_size = group.size()

        # Test various input sizes
        sizes = [
            (world_size,),  # minimal 1D
            (world_size * 256, 64),  # medium 2D
            (world_size * 2, 3, 4, 5),  # multi-dimensional
        ]

        for sh in sizes:
            x = mx.ones(sh, dtype=mx.float32)
            y = mx.distributed.all_to_all(x, group=group)
            mx.eval(y)

            self.assertEqual(y.shape, x.shape)
            if world_size == 1:
                self.assertTrue(mx.array_equal(y, x).item())

    def test_all_to_all_non_contiguous(self):
        group = mx.distributed.init()
        self._skip_if_all_to_all_unsupported(group)
        world_size = group.size()

        # Create a non-contiguous input via transpose then slice
        base = mx.random.normal((8, world_size * 4))
        x_non_contig = base.T  # shape (world_size * 4, 8), non-contiguous

        # Create contiguous copy
        x_contig = mx.array(x_non_contig)

        y1 = mx.distributed.all_to_all(x_non_contig, group=group)
        y2 = mx.distributed.all_to_all(x_contig, group=group)
        mx.eval(y1, y2)

        self.assertTrue(mx.allclose(y1, y2).item())

    def test_all_to_all_vjp(self):
        group = mx.distributed.init()
        self._skip_if_all_to_all_unsupported(group)
        world_size = group.size()

        x = mx.random.normal((world_size * 4, 8))
        mx.eval(x)

        # Test mx.grad
        grad_fn = mx.grad(lambda x: mx.distributed.all_to_all(x, group=group).sum())
        g = grad_fn(x)
        mx.eval(g)

        if world_size == 1:
            # For single process: gradient of identity + sum is all ones
            self.assertTrue(mx.allclose(g, mx.ones_like(g)).item())

        # Test mx.value_and_grad
        val_grad_fn = mx.value_and_grad(
            lambda x: mx.distributed.all_to_all(x, group=group).sum()
        )
        val, g2 = val_grad_fn(x)
        mx.eval(val, g2)

        self.assertEqual(g2.shape, x.shape)

    @unittest.skipIf(mx.distributed.init().size() == 1, "requires world_size > 1")
    def test_all_to_all_shape_validation(self):
        group = mx.distributed.init()
        self._skip_if_all_to_all_unsupported(group)
        world_size = group.size()

        # Test that scalar input raises an exception
        scalar = mx.array(1.0)
        with self.assertRaises(Exception):
            mx.eval(mx.distributed.all_to_all(scalar, group=group))

        # Test that x.shape[0] % world_size != 0 raises (only meaningful for world_size > 1)
        if world_size > 1:
            bad = mx.ones((world_size * 4 + 1, 8))
            with self.assertRaises(Exception):
                mx.eval(mx.distributed.all_to_all(bad, group=group))

    def test_all_gather(self):
        world = mx.distributed.init()
        dtypes = [
            mx.int8,
            mx.uint8,
            mx.int32,
            mx.uint32,
            mx.float32,
            mx.float16,
            mx.bfloat16,
        ]
        for dt in dtypes:
            x = mx.ones((2, 2, 4), dtype=dt)
            y = mx.distributed.all_gather(x)
            self.assertEqual(y.shape, (world.size() * 2, 2, 4))
            self.assertTrue(mx.all(y == 1))

    def test_moe_ep_forward(self):
        group = mx.distributed.init()
        self._skip_if_all_to_all_unsupported(group)
        if group.size() != 2:
            self.skipTest("MoE EP tests require exactly 2 devices")

        world_size = group.size()
        rank = group.rank()

        mx.random.seed(42)
        hidden_dim = 16
        expert_dim = 32
        num_experts = 4
        top_k = 2
        num_tokens = 8

        moe = MixtureOfExperts(
            hidden_dim=hidden_dim,
            expert_dim=expert_dim,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=2.0,
        )
        mx.eval(moe.parameters())

        x = mx.random.normal((num_tokens, hidden_dim)) + rank * 1000
        output, aux_loss = moe(x)
        mx.eval(output, aux_loss)

        # Shape check
        self.assertEqual(output.shape, (num_tokens, hidden_dim))
        # Finiteness check
        self.assertTrue(mx.all(mx.isfinite(output)).item())
        self.assertTrue(mx.isfinite(aux_loss).item())
        # EP enabled check
        self.assertEqual(moe._world_size, world_size)
        # Local expert count
        self.assertEqual(len(moe.experts), num_experts // world_size)

    def test_moe_ep_uneven_batch(self):
        group = mx.distributed.init()
        self._skip_if_all_to_all_unsupported(group)
        if group.size() != 2:
            self.skipTest("MoE EP tests require exactly 2 devices")

        rank = group.rank()

        mx.random.seed(42)
        hidden_dim = 16
        expert_dim = 32
        num_experts = 4
        top_k = 2

        moe = MixtureOfExperts(
            hidden_dim=hidden_dim,
            expert_dim=expert_dim,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=2.0,
        )
        mx.eval(moe.parameters())

        if rank == 0:
            x = mx.random.normal((4, hidden_dim)) + rank * 1000
        else:
            x = mx.zeros((0, hidden_dim))

        output, aux_loss = moe(x)
        mx.eval(output, aux_loss)

        if rank == 0:
            self.assertEqual(output.shape, (4, hidden_dim))
            self.assertTrue(mx.all(mx.isfinite(output)).item())
        else:
            # Empty rank checks
            self.assertEqual(output.shape, (0, hidden_dim))
            self.assertTrue(mx.all(mx.isfinite(output)).item())
            self.assertEqual(aux_loss.item(), 0.0)

    def test_moe_ep_dispatch_combine_roundtrip(self):
        group = mx.distributed.init()
        self._skip_if_all_to_all_unsupported(group)
        if group.size() != 2:
            self.skipTest("MoE EP tests require exactly 2 devices")

        rank = group.rank()
        hidden_dim = 8
        num_tokens = 4
        num_experts = 4
        top_k = 1
        capacity_factor = 2.0

        # Rank-distinct input
        x = (mx.arange(num_tokens).reshape(-1, 1) + 1 + rank * 1000).astype(mx.float32)
        x = mx.broadcast_to(x, (num_tokens, hidden_dim))
        x = mx.array(x)  # make contiguous

        # Build routing: half local, half remote
        expert_indices = mx.zeros((num_tokens, top_k), dtype=mx.int32)
        for i in range(num_tokens):
            if i < num_tokens // 2:
                # Local expert
                expert_indices = expert_indices.at[i, 0].add(rank * 2 + i % 2)
            else:
                # Remote expert
                expert_indices = expert_indices.at[i, 0].add((1 - rank) * 2 + i % 2)

        weights = mx.ones((num_tokens, top_k), dtype=mx.float32)

        dispatched, meta = expert_dispatch(
            x,
            expert_indices,
            weights,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            group=group,
        )
        mx.eval(dispatched)

        # Identity: just pass through
        combined = expert_combine(dispatched, meta, x, group=group)
        mx.eval(combined)

        # Check valid-route tokens round-trip correctly
        has_valid = (meta.positions >= 0).any(axis=1)
        mx.eval(has_valid)
        for i in range(num_tokens):
            if has_valid[i].item():
                self.assertTrue(
                    mx.allclose(combined[i], x[i], atol=1e-5, rtol=1e-4).item(),
                    f"Token {i} on rank {rank} did not round-trip correctly",
                )

    def test_moe_ep_partial_overflow(self):
        group = mx.distributed.init()
        self._skip_if_all_to_all_unsupported(group)
        if group.size() != 2:
            self.skipTest("MoE EP tests require exactly 2 devices")

        rank = group.rank()
        hidden_dim = 8
        num_tokens = 4
        num_experts = 4
        top_k = 2
        capacity_factor = 0.5  # Force overflow

        # Rank-distinct input
        x = mx.ones((num_tokens, hidden_dim), dtype=mx.float32) * (rank + 1.0) * 100

        # Route all tokens to experts 0 and 1 -> overflow with low capacity
        expert_indices = mx.zeros((num_tokens, top_k), dtype=mx.int32)
        expert_indices = expert_indices.at[:, 1].add(
            1
        )  # k=0 -> expert 0, k=1 -> expert 1
        weights = mx.ones((num_tokens, top_k), dtype=mx.float32) * 0.5

        dispatched, meta = expert_dispatch(
            x,
            expert_indices,
            weights,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            group=group,
        )
        mx.eval(dispatched)

        combined = expert_combine(dispatched, meta, x, group=group)
        mx.eval(combined)

        has_valid_route = (meta.positions >= 0).any(axis=1)
        mx.eval(has_valid_route)

        for i in range(num_tokens):
            if has_valid_route[i].item():
                # Valid route tokens should be finite
                self.assertTrue(
                    mx.all(mx.isfinite(combined[i])).item(),
                    f"Token {i} on rank {rank} has non-finite values",
                )
            else:
                # Overflow tokens should fall back to residual (original input)
                self.assertTrue(
                    mx.array_equal(combined[i], x[i]).item(),
                    f"Overflow token {i} on rank {rank} did not use residual",
                )

    def test_moe_ep_gradient(self):
        group = mx.distributed.init()
        self._skip_if_all_to_all_unsupported(group)
        if group.size() != 2:
            self.skipTest("MoE EP tests require exactly 2 devices")

        rank = group.rank()

        mx.random.seed(42)
        hidden_dim = 16
        expert_dim = 32
        num_experts = 4
        top_k = 2
        num_tokens = 8

        moe = MixtureOfExperts(
            hidden_dim=hidden_dim,
            expert_dim=expert_dim,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=2.0,
            aux_loss_coeff=0.0,  # Exclude aux_loss to test all_to_all VJP path
        )
        mx.eval(moe.parameters())

        x = mx.random.normal((num_tokens, hidden_dim)) + rank * 1000

        def loss_fn(model, x):
            output, _aux = model(x)
            return output.sum()

        loss_and_grad = nn.value_and_grad(moe, loss_fn)
        loss, grads = loss_and_grad(moe, x)
        mx.eval(loss, grads)

        # Loss should be finite
        self.assertTrue(mx.isfinite(loss).item())

        # Router gate grad: must be finite
        gate_grad = grads["router"]["gate"]["weight"]
        self.assertTrue(mx.all(mx.isfinite(gate_grad)).item())
        gate_has_grad = mx.any(gate_grad != 0).item()

        # Check local expert grads: must be finite, at least one non-zero
        any_expert_has_grad = False
        for i in range(len(moe.experts)):
            w_gate_grad = grads["experts"][i]["w_gate"]["weight"]
            self.assertTrue(
                mx.all(mx.isfinite(w_gate_grad)).item(),
                f"Expert {i} w_gate grad has non-finite values on rank {rank}",
            )
            if mx.any(w_gate_grad != 0).item():
                any_expert_has_grad = True

        # At least gate or one expert must receive non-zero gradients
        self.assertTrue(
            gate_has_grad or any_expert_has_grad,
            f"Neither gate nor any expert received non-zero gradients on rank {rank}",
        )
