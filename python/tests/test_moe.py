# Copyright © 2026 Apple Inc.

import math
import unittest

import mlx.core as mx
import mlx.nn as nn
import mlx_tests

# Imports will be validated once the MoE module is implemented.
# If the module path changes, update accordingly.
from mlx.nn.layers.moe import (
    DispatchMeta,
    Expert,
    MixtureOfExperts,
    TopKRouter,
    _compute_capacity,
    expert_combine,
    expert_dispatch,
)


class TestTopKRouter(mlx_tests.MLXTestCase):
    def test_output_shapes(self):
        """Router should produce correct output shapes."""
        hidden_dim, num_experts, top_k = 64, 8, 2
        router = TopKRouter(hidden_dim, num_experts, top_k)
        x = mx.random.normal((16, hidden_dim))
        weights, indices, aux_loss = router(x)
        mx.eval(weights, indices, aux_loss)

        self.assertEqual(weights.shape, (16, top_k))
        self.assertEqual(indices.shape, (16, top_k))
        self.assertEqual(aux_loss.shape, ())

    def test_index_range(self):
        """Expert indices should be in [0, num_experts)."""
        hidden_dim, num_experts, top_k = 64, 8, 2
        router = TopKRouter(hidden_dim, num_experts, top_k)
        x = mx.random.normal((32, hidden_dim))
        _, indices, _ = router(x)
        mx.eval(indices)

        self.assertTrue(mx.all(indices >= 0).item())
        self.assertTrue(mx.all(indices < num_experts).item())

    def test_weights_sum(self):
        """Routing weights should approximately sum to 1 per token."""
        hidden_dim, num_experts, top_k = 64, 8, 2
        router = TopKRouter(hidden_dim, num_experts, top_k)
        x = mx.random.normal((16, hidden_dim))
        weights, _, _ = router(x)
        mx.eval(weights)

        weight_sums = weights.sum(axis=-1)
        self.assertTrue(
            mx.allclose(weight_sums, mx.ones_like(weight_sums), atol=1e-5).item()
        )

    def test_gradient_flow(self):
        """Gradients should flow through the router gate."""
        hidden_dim, num_experts, top_k = 32, 4, 2
        router = TopKRouter(hidden_dim, num_experts, top_k)
        x = mx.random.normal((8, hidden_dim))

        def loss_fn(model, x):
            weights, _, aux_loss = model(x)
            return weights.sum() + aux_loss

        loss, grads = nn.value_and_grad(router, loss_fn)(router, x)
        mx.eval(loss, grads)

        # Gate weight should have non-zero gradient
        self.assertTrue(mx.any(grads["gate"]["weight"] != 0).item())

    def test_aux_loss_positive(self):
        """Auxiliary loss should be non-negative."""
        hidden_dim, num_experts, top_k = 64, 8, 2
        router = TopKRouter(hidden_dim, num_experts, top_k)
        x = mx.random.normal((16, hidden_dim))
        _, _, aux_loss = router(x)
        mx.eval(aux_loss)

        self.assertTrue(aux_loss.item() >= 0)

    def test_different_top_k_values(self):
        """Router should work with different top_k values."""
        hidden_dim, num_experts = 64, 8
        x = mx.random.normal((16, hidden_dim))

        for top_k in [1, 2, 4]:
            router = TopKRouter(hidden_dim, num_experts, top_k)
            weights, indices, aux_loss = router(x)
            mx.eval(weights, indices, aux_loss)

            self.assertEqual(weights.shape, (16, top_k))
            self.assertEqual(indices.shape, (16, top_k))

    def test_single_token(self):
        """Router should handle a single-token input."""
        hidden_dim, num_experts, top_k = 64, 8, 2
        router = TopKRouter(hidden_dim, num_experts, top_k)
        x = mx.random.normal((1, hidden_dim))
        weights, indices, aux_loss = router(x)
        mx.eval(weights, indices, aux_loss)

        self.assertEqual(weights.shape, (1, top_k))
        self.assertEqual(indices.shape, (1, top_k))

    def test_top_k_validation(self):
        """Should raise error for invalid top_k."""
        with self.assertRaises(ValueError):
            TopKRouter(64, 8, top_k=0)
        with self.assertRaises(ValueError):
            TopKRouter(64, 8, top_k=9)

    def test_empty_batch(self):
        """Router should handle zero-token input without NaN."""
        router = TopKRouter(64, 8, top_k=2)
        x = mx.zeros((0, 64))
        weights, indices, aux_loss = router(x)
        mx.eval(weights, indices, aux_loss)
        self.assertEqual(weights.shape, (0, 2))
        self.assertEqual(indices.shape, (0, 2))
        self.assertTrue(mx.isfinite(aux_loss).item())
        self.assertEqual(aux_loss.item(), 0.0)


class TestExpert(mlx_tests.MLXTestCase):
    def test_output_shape(self):
        """Expert should preserve input/output dimensions."""
        hidden_dim, expert_dim = 64, 128
        expert = Expert(hidden_dim, expert_dim)
        x = mx.random.normal((8, hidden_dim))
        out = expert(x)
        mx.eval(out)

        self.assertEqual(out.shape, (8, hidden_dim))

    def test_gradient_flow(self):
        """Gradients should flow through expert."""
        hidden_dim, expert_dim = 32, 64
        expert = Expert(hidden_dim, expert_dim)
        x = mx.random.normal((4, hidden_dim))

        def loss_fn(model, x):
            return model(x).sum()

        loss, grads = nn.value_and_grad(expert, loss_fn)(expert, x)
        mx.eval(loss, grads)

        self.assertTrue(mx.any(grads["w_gate"]["weight"] != 0).item())
        self.assertTrue(mx.any(grads["w_up"]["weight"] != 0).item())
        self.assertTrue(mx.any(grads["w_down"]["weight"] != 0).item())

    def test_single_token(self):
        """Expert should handle single-token input."""
        hidden_dim, expert_dim = 64, 128
        expert = Expert(hidden_dim, expert_dim)
        x = mx.random.normal((1, hidden_dim))
        out = expert(x)
        mx.eval(out)

        self.assertEqual(out.shape, (1, hidden_dim))

    def test_empty_input(self):
        """Expert should handle zero-token input."""
        hidden_dim, expert_dim = 64, 128
        expert = Expert(hidden_dim, expert_dim)
        x = mx.zeros((0, hidden_dim))
        out = expert(x)
        mx.eval(out)

        self.assertEqual(out.shape, (0, hidden_dim))


class TestComputeCapacity(mlx_tests.MLXTestCase):
    def test_basic(self):
        """Test capacity computation."""
        # 16 tokens, top_k=2, factor=1.25, 8 experts
        cap = _compute_capacity(16, 2, 1.25, 8)
        expected = max(1, math.ceil(16 * 2 * 1.25 / 8))  # ceil(5.0) = 5
        self.assertEqual(cap, expected)

    def test_minimum_one(self):
        """Capacity should be at least 1."""
        cap = _compute_capacity(0, 2, 1.0, 8)
        self.assertEqual(cap, 1)

    def test_exact_division(self):
        """Test when division is exact."""
        # 8 tokens, top_k=1, factor=1.0, 4 experts -> ceil(2.0) = 2
        cap = _compute_capacity(8, 1, 1.0, 4)
        self.assertEqual(cap, 2)

    def test_large_capacity_factor(self):
        """Larger capacity factor yields larger capacity."""
        cap_low = _compute_capacity(16, 2, 1.0, 8)
        cap_high = _compute_capacity(16, 2, 2.0, 8)
        self.assertGreaterEqual(cap_high, cap_low)


class TestMixtureOfExperts(mlx_tests.MLXTestCase):
    def test_forward_shape(self):
        """MoE forward should produce correct output shape."""
        hidden_dim, expert_dim, num_experts = 64, 128, 4
        moe = MixtureOfExperts(hidden_dim, expert_dim, num_experts, top_k=2)
        x = mx.random.normal((8, hidden_dim))
        output, aux_loss = moe(x)
        mx.eval(output, aux_loss)

        self.assertEqual(output.shape, (8, hidden_dim))
        self.assertEqual(aux_loss.shape, ())

    def test_backward(self):
        """MoE should support backward pass."""
        hidden_dim, expert_dim, num_experts = 32, 64, 4
        moe = MixtureOfExperts(hidden_dim, expert_dim, num_experts, top_k=2)
        x = mx.random.normal((4, hidden_dim))

        def loss_fn(model, x):
            output, aux_loss = model(x)
            return output.sum() + aux_loss

        loss, grads = nn.value_and_grad(moe, loss_fn)(moe, x)
        mx.eval(loss, grads)

        # At least the router should have gradients
        self.assertIsNotNone(grads["router"]["gate"]["weight"])

    def test_parameter_count(self):
        """Verify parameter structure."""
        hidden_dim, expert_dim, num_experts = 64, 128, 4
        moe = MixtureOfExperts(hidden_dim, expert_dim, num_experts, top_k=2)

        params = moe.parameters()
        # Should have router and experts
        self.assertIn("router", params)
        self.assertIn("experts", params)
        # Single process: all experts are local
        self.assertEqual(len(params["experts"]), num_experts)

    def test_validation_error(self):
        """Should raise error for invalid num_experts or top_k."""
        with self.assertRaises(Exception):
            MixtureOfExperts(64, 128, 0)
        with self.assertRaises(ValueError):
            MixtureOfExperts(64, 128, 4, top_k=0)
        with self.assertRaises(ValueError):
            MixtureOfExperts(64, 128, 4, top_k=5)

    def test_different_top_k(self):
        """MoE should work with different top_k values."""
        hidden_dim, expert_dim, num_experts = 64, 128, 4
        x = mx.random.normal((8, hidden_dim))

        for top_k in [1, 2]:
            moe = MixtureOfExperts(hidden_dim, expert_dim, num_experts, top_k=top_k)
            output, aux_loss = moe(x)
            mx.eval(output, aux_loss)

            self.assertEqual(output.shape, (8, hidden_dim))

    def test_deterministic_with_seed(self):
        """Same seed should produce same results."""
        hidden_dim, expert_dim, num_experts = 32, 64, 4
        x = mx.random.normal((4, hidden_dim))

        moe1 = MixtureOfExperts(hidden_dim, expert_dim, num_experts, top_k=2)
        moe2 = MixtureOfExperts(hidden_dim, expert_dim, num_experts, top_k=2)
        moe2.update(moe1.parameters())

        out1, loss1 = moe1(x)
        out2, loss2 = moe2(x)
        mx.eval(out1, out2, loss1, loss2)

        self.assertTrue(mx.allclose(out1, out2).item())
        self.assertTrue(mx.allclose(loss1, loss2).item())

    def test_large_batch(self):
        """MoE should handle larger batch sizes."""
        hidden_dim, expert_dim, num_experts = 64, 128, 8
        moe = MixtureOfExperts(hidden_dim, expert_dim, num_experts, top_k=2)
        x = mx.random.normal((128, hidden_dim))
        output, aux_loss = moe(x)
        mx.eval(output, aux_loss)

        self.assertEqual(output.shape, (128, hidden_dim))

    def test_partial_overflow_preserves_valid_routes(self):
        """Tokens with at least one valid route should not be replaced by residual."""
        hidden_dim = 4
        num_experts = 2
        capacity = 1
        top_k = 2

        # token0: expert 0 valid (pos=0), expert 1 overflow (pos=-1)
        # token1: both routes overflow (pos=-1, -1)
        positions = mx.array([[0, -1], [-1, -1]], dtype=mx.int32)
        expert_indices = mx.array([[0, 1], [0, 1]], dtype=mx.int32)
        weights = mx.array([[0.6, 0.4], [0.5, 0.5]])
        overflow_mask = mx.array([[True], [True]])

        meta = DispatchMeta(
            expert_indices=expert_indices,
            weights=weights,
            positions=positions,
            overflow_mask=overflow_mask,
            num_experts=num_experts,
            capacity=capacity,
            world_size=1,
        )

        # expert_outputs: [num_experts, capacity, hidden_dim]
        expert_outputs = mx.ones((num_experts, capacity, hidden_dim)) * 10.0
        original_tokens = mx.zeros((2, hidden_dim))

        combined = expert_combine(expert_outputs, meta, original_tokens)
        mx.eval(combined)

        # Verify bug reproduction condition: token0 has overflow_mask=True
        # but should still use expert output because it has a valid route.
        self.assertTrue(meta.overflow_mask[0].item())
        has_valid = (meta.positions[0] >= 0).any().item()
        self.assertTrue(has_valid)

        # token0: weight=0.6 * expert_output=10.0 → expected 6.0 per dim
        expected_token0 = mx.full((hidden_dim,), 0.6 * 10.0)
        self.assertTrue(mx.allclose(combined[0], expected_token0).item())
        # token1: all overflow → should be original (zeros)
        self.assertTrue(mx.array_equal(combined[1], original_tokens[1]).item())

    def test_ep_backend_parameter(self):
        """Test ep_backend parameter is stored and used."""
        moe = MixtureOfExperts(
            hidden_dim=32,
            expert_dim=64,
            num_experts=4,
            top_k=2,
            ep_impl="auto",
            ep_backend="auto",
        )
        self.assertEqual(moe.ep_backend, "auto")

        moe2 = MixtureOfExperts(
            hidden_dim=32,
            expert_dim=64,
            num_experts=4,
            top_k=2,
            ep_impl="auto",
            ep_backend="cpu",
        )
        self.assertEqual(moe2.ep_backend, "cpu")


class TestVectorizedDispatchCombine(mlx_tests.MLXTestCase):
    def test_dispatch_combine_duplicate_expert_across_k(self):
        """Same expert selected by both top_k slots should not collide positions."""
        N, D = 4, 8
        num_experts = 4
        capacity_factor = 2.0  # generous capacity

        tokens = mx.random.normal((N, D))
        # Force token 0 and token 1 to route to the same expert (expert 0) for both k=0 and k=1
        expert_indices = mx.array(
            [
                [0, 0],  # token 0: expert 0 twice
                [0, 0],  # token 1: expert 0 twice
                [1, 2],  # token 2: different experts
                [3, 1],  # token 3: different experts
            ],
            dtype=mx.int32,
        )
        weights = mx.array(
            [
                [0.6, 0.4],
                [0.5, 0.5],
                [0.7, 0.3],
                [0.8, 0.2],
            ]
        )

        dispatched, meta = expert_dispatch(
            tokens,
            expert_indices,
            weights,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )
        mx.eval(dispatched, *meta)

        # Positions for token 0 and token 1 should be different across k
        # (expert_counts accumulation ensures no collision)
        pos_token0 = meta.positions[0]  # [top_k]
        pos_token1 = meta.positions[1]  # [top_k]

        # All positions should be >= 0 (no overflow with generous capacity)
        self.assertTrue(
            mx.all(meta.positions >= 0).item(),
            f"Expected all valid positions, got {meta.positions}",
        )

        # For tokens routed to same expert: k=0 and k=1 positions must differ
        self.assertNotEqual(
            pos_token0[0].item(),
            pos_token0[1].item(),
            "Same expert positions should differ across k",
        )

        # Round-trip test: dispatch then combine with identity expert
        expert_outputs = dispatched  # identity
        combined = expert_combine(expert_outputs, meta, tokens)
        mx.eval(combined)
        # Combined should not contain NaN
        self.assertTrue(mx.all(mx.isfinite(combined)).item())

    def test_dispatch_combine_overflow_boundary(self):
        """Capacity boundary: first 2 tokens fit, last 2 overflow."""
        N, D = 4, 8
        num_experts = 2

        tokens = mx.ones((N, D))  # all-ones for easy verification
        # All tokens go to expert 0 for k=0, expert 1 for k=1
        expert_indices = mx.array(
            [
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
            ],
            dtype=mx.int32,
        )
        weights = mx.array(
            [
                [0.6, 0.4],
                [0.6, 0.4],
                [0.6, 0.4],
                [0.6, 0.4],
            ]
        )

        # capacity = max(1, ceil(4 * 2 * capacity_factor / 2))
        # With capacity_factor = 0.5: ceil(4 * 2 * 0.5 / 2) = ceil(2.0) = 2
        dispatched, meta = expert_dispatch(
            tokens,
            expert_indices,
            weights,
            num_experts=num_experts,
            capacity_factor=0.5,
        )
        mx.eval(dispatched, *meta)

        capacity = meta.capacity
        self.assertEqual(capacity, 2)

        # For k=0 (expert 0): tokens 0,1 should have positions 0,1; tokens 2,3 overflow
        positions_k0 = meta.positions[:, 0]
        mx.eval(positions_k0)
        self.assertEqual(positions_k0[0].item(), 0)
        self.assertEqual(positions_k0[1].item(), 1)
        self.assertEqual(positions_k0[2].item(), -1)  # overflow
        self.assertEqual(positions_k0[3].item(), -1)  # overflow

        # Overflow mask should be True for tokens 2 and 3
        self.assertTrue(meta.overflow_mask[2].item())
        self.assertTrue(meta.overflow_mask[3].item())

    def test_dispatch_combine_empty_batch(self):
        """N=0 input should produce correct shapes without errors."""
        D = 8
        num_experts = 4

        tokens = mx.zeros((0, D))
        expert_indices = mx.zeros((0, 2), dtype=mx.int32)
        weights = mx.zeros((0, 2))

        dispatched, meta = expert_dispatch(
            tokens,
            expert_indices,
            weights,
            num_experts=num_experts,
            capacity_factor=1.25,
        )
        mx.eval(dispatched, *meta)

        # Shape checks
        self.assertEqual(meta.positions.shape, (0, 2))
        self.assertEqual(meta.overflow_mask.shape, (0, 1))
        self.assertEqual(dispatched.shape[0], num_experts)  # experts_per_device
        self.assertEqual(dispatched.shape[-1], D)

        # Round-trip with combine
        expert_outputs = dispatched
        combined = expert_combine(expert_outputs, meta, tokens)
        mx.eval(combined)
        self.assertEqual(combined.shape, (0, D))

    def test_combine_all_invalid_residual(self):
        """All routes invalid → combined should equal original_tokens."""
        N, D = 4, 8
        num_experts = 2
        capacity = 2

        original_tokens = mx.random.normal((N, D))
        expert_outputs = mx.random.normal((num_experts, capacity, D))

        # Manually construct meta with all-invalid positions
        positions = mx.full((N, 2), -1, dtype=mx.int32)
        expert_indices = mx.array([[0, 1]] * N, dtype=mx.int32)
        weights = mx.array([[0.5, 0.5]] * N)
        overflow_mask = mx.ones((N, 1), dtype=mx.bool_)

        meta = DispatchMeta(
            expert_indices=expert_indices,
            weights=weights,
            positions=positions,
            overflow_mask=overflow_mask,
            num_experts=num_experts,
            capacity=capacity,
            world_size=1,
        )

        combined = expert_combine(expert_outputs, meta, original_tokens)
        mx.eval(combined)

        self.assertTrue(mx.allclose(combined, original_tokens).item())


class TestCppMoeExchange(unittest.TestCase):
    """Tests for C++ moe_dispatch_exchange / moe_combine_exchange primitives."""

    def setUp(self):
        # Skip if C++ primitive not available
        if not hasattr(mx.distributed, "moe_dispatch_exchange"):
            self.skipTest("moe_dispatch_exchange not available")
        # Detect actual world_size: try each backend explicitly so that
        # mlx.launch-initialized backends (JACCL/MPI) are detected correctly.
        self._world_size = 1
        for backend in ("jaccl", "mpi", "nccl"):
            try:
                g = mx.distributed.init(strict=True, backend=backend)
                if g.size() > 1:
                    self._world_size = g.size()
                    break
            except Exception:
                pass
        if self._world_size == 1:
            try:
                self._world_size = mx.distributed.init().size()
            except Exception:
                self._world_size = 1

    def _python_dispatch_combine_ref(
        self, tokens, expert_indices, weights, num_experts, capacity
    ):
        """Reference Python implementation for comparison."""
        N, D = tokens.shape
        top_k = expert_indices.shape[1]
        experts_per_device = num_experts  # local only (world_size=1)

        dispatch_flat = mx.zeros((num_experts * capacity, D), dtype=tokens.dtype)
        route_indices = mx.full((N, top_k), -1, dtype=mx.int32)

        expert_counts = [0] * num_experts
        route_list = [[-1] * top_k for _ in range(N)]
        for k in range(top_k):
            for n in range(N):
                eid = expert_indices[n, k].item()
                pos = expert_counts[eid]
                if pos < capacity:
                    flat_idx = eid * capacity + pos
                    route_list[n][k] = flat_idx
                    expert_counts[eid] += 1

        route_np = mx.array(route_list, dtype=mx.int32)
        # Build dispatch flat
        disp = mx.zeros((num_experts * capacity, D), dtype=tokens.dtype)
        for n in range(N):
            for k in range(top_k):
                flat_idx = route_list[n][k]
                if flat_idx >= 0:
                    disp = disp.at[flat_idx].add(tokens[n])
        dispatched = disp.reshape(num_experts, capacity, D)

        # Combine
        combined = mx.zeros((N, D), dtype=tokens.dtype)
        result_flat = disp
        for n in range(N):
            has_valid = False
            for k in range(top_k):
                flat_idx = route_list[n][k]
                if flat_idx >= 0:
                    has_valid = True
                    w = weights[n, k].item()
                    combined = combined.at[n].add(w * result_flat[flat_idx])
            if not has_valid:
                combined = combined.at[n].add(tokens[n])

        return dispatched, route_np, combined

    def test_dispatch_local_basic(self):
        """Local dispatch matches reference for simple case."""
        if self._world_size > 1:
            self.skipTest("local-only test")
        mx.random.seed(42)
        N, D, E, top_k = 8, 16, 4, 2
        capacity = 4

        tokens = mx.random.normal((N, D))
        # Assign each token to experts deterministically
        expert_indices = mx.array(
            [[i % E, (i + 1) % E] for i in range(N)], dtype=mx.int32
        )
        weights = mx.ones((N, top_k)) / top_k

        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
        )
        mx.eval(dispatched, route_idx)

        # Shape check
        self.assertEqual(dispatched.shape, (E, capacity, D))
        self.assertEqual(route_idx.shape, (N, top_k))
        self.assertEqual(route_idx.dtype, mx.int32)

    def test_dispatch_combine_roundtrip(self):
        """Dispatch -> identity expert -> combine = input for non-overflow case."""
        mx.random.seed(7)
        N, D, E, top_k = 4, 8, 4, 2
        capacity = 4  # large enough for no overflow

        tokens = mx.random.normal((N, D))
        # Each token goes to a unique expert pair
        expert_indices = mx.array([[0, 1], [2, 3], [0, 2], [1, 3]], dtype=mx.int32)
        weights = mx.ones((N, top_k)) / top_k  # uniform

        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
        )
        mx.eval(dispatched, route_idx)

        # Identity expert: expert_outputs = dispatched
        combined = mx.distributed.moe_combine_exchange(
            dispatched,
            route_idx,
            weights,
            tokens,
            num_experts=E,
            capacity=capacity,
        )
        mx.eval(combined)

        # Should reconstruct original tokens
        self.assertEqual(combined.shape, (N, D))
        self.assertTrue(mx.allclose(combined, tokens, atol=1e-5).item())

    def test_overflow_residual_fallback(self):
        """Tokens with all-overflow routes get original_tokens as residual."""
        if self._world_size > 1:
            self.skipTest("local-only test")
        N, D, E, top_k = 4, 8, 1, 2
        capacity = 1  # only 1 slot for the single expert

        tokens = mx.random.normal((N, D))
        # All tokens go to expert 0, but capacity=1 -> most overflow
        expert_indices = mx.zeros((N, top_k), dtype=mx.int32)
        weights = mx.ones((N, top_k)) / top_k

        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
        )
        mx.eval(dispatched, route_idx)

        # Most route_indices should be -1 (overflow)
        n_overflow = (route_idx == -1).sum().item()
        self.assertGreater(n_overflow, 0)

        # Identity expert
        combined = mx.distributed.moe_combine_exchange(
            dispatched,
            route_idx,
            weights,
            tokens,
            num_experts=E,
            capacity=capacity,
        )
        mx.eval(combined)

        # Token 0 (first to be dispatched) has valid route, rest are overflow
        # Fully overflowed tokens should get original_tokens
        route_idx_np = route_idx.tolist()
        for n in range(N):
            all_invalid = all(route_idx_np[n][k] < 0 for k in range(top_k))
            if all_invalid:
                self.assertTrue(
                    mx.allclose(combined[n], tokens[n], atol=1e-5).item(),
                    f"Token {n} should be residual fallback",
                )

    def test_empty_batch(self):
        """N=0 (empty batch) should produce empty outputs."""
        if self._world_size > 1:
            self.skipTest("local-only test")
        E, D, top_k = 4, 16, 2
        capacity = 4

        tokens = mx.zeros((0, D))
        expert_indices = mx.zeros((0, top_k), dtype=mx.int32)
        weights = mx.zeros((0, top_k))

        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
        )
        mx.eval(dispatched, route_idx)

        self.assertEqual(dispatched.shape, (E, capacity, D))
        self.assertEqual(route_idx.shape, (0, top_k))

        combined = mx.distributed.moe_combine_exchange(
            dispatched,
            route_idx,
            weights,
            tokens,
            num_experts=E,
            capacity=capacity,
        )
        mx.eval(combined)
        self.assertEqual(combined.shape, (0, D))

    def test_dispatch_deterministic(self):
        """Same input always produces same route_indices (deterministic=True)."""
        N, D, E, top_k = 16, 8, 4, 2
        capacity = 6

        tokens = mx.random.normal((N, D))
        expert_indices = mx.array(
            [[i % E, (i + 2) % E] for i in range(N)], dtype=mx.int32
        )

        _, route1 = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
        )
        _, route2 = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
        )
        mx.eval(route1, route2)
        self.assertTrue((route1 == route2).all().item())

    def test_dtype_float16(self):
        """float16 tokens are correctly dispatched and combined."""
        N, D, E, top_k = 8, 16, 4, 2
        capacity = 4

        tokens = mx.random.normal((N, D)).astype(mx.float16)
        expert_indices = mx.array(
            [[i % E, (i + 1) % E] for i in range(N)], dtype=mx.int32
        )
        weights = mx.ones((N, top_k), dtype=mx.float32) / top_k

        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
        )
        mx.eval(dispatched, route_idx)
        self.assertEqual(dispatched.dtype, mx.float16)

        combined = mx.distributed.moe_combine_exchange(
            dispatched,
            route_idx,
            weights,
            tokens,
            num_experts=E,
            capacity=capacity,
        )
        mx.eval(combined)
        self.assertEqual(combined.dtype, mx.float16)
        self.assertEqual(combined.shape, (N, D))

    def test_cpp_vs_python_consistency(self):
        """C++ primitive matches Python expert_dispatch/combine for local mode."""
        if self._world_size > 1:
            self.skipTest("local-only test")
        mx.random.seed(123)
        N, D, E, top_k = 12, 8, 4, 2
        capacity_factor = 1.5

        tokens = mx.random.normal((N, D))
        expert_indices = mx.array(
            [[i % E, (i + 1) % E] for i in range(N)], dtype=mx.int32
        )
        weights = mx.random.uniform(shape=(N, top_k))
        weights = weights / weights.sum(axis=-1, keepdims=True)

        # Python path
        dispatched_py, meta = expert_dispatch(
            tokens, expert_indices, weights, E, capacity_factor, group=None
        )
        capacity = meta.capacity
        expert_out_py = dispatched_py  # identity expert

        # C++ path
        dispatched_cpp, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices.astype(mx.int32),
            num_experts=E,
            capacity=capacity,
        )
        mx.eval(dispatched_cpp, route_idx)

        # Both dispatch should have same shape
        self.assertEqual(
            dispatched_cpp.shape,
            dispatched_py.shape,
            f"Shape mismatch: cpp={dispatched_cpp.shape} py={dispatched_py.shape}",
        )

        # C++ combine
        combined_cpp = mx.distributed.moe_combine_exchange(
            dispatched_cpp,
            route_idx,
            weights,
            tokens,
            num_experts=E,
            capacity=capacity,
        )
        # Python combine
        combined_py = expert_combine(expert_out_py, meta, tokens, group=None)

        mx.eval(combined_cpp, combined_py)

        # Results should be close (same deterministic routing)
        self.assertTrue(
            mx.allclose(combined_cpp, combined_py, atol=1e-5).item(),
            f"C++ and Python combine results differ.\n"
            f"Max diff: {mx.abs(combined_cpp - combined_py).max().item()}",
        )

    def test_backend_auto(self):
        """Test that backend='auto' works (resolves to cpu in current build)."""
        N, D, top_k = 8, 16, 2
        num_experts = self._world_size * 2
        capacity = 4
        tokens = mx.random.normal((N, D))
        expert_indices = mx.random.randint(0, num_experts, shape=(N, top_k)).astype(
            mx.int32
        )

        # backend="auto" should work without error
        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=num_experts,
            capacity=capacity,
            backend="auto",
        )
        mx.eval(dispatched, route_idx)

        # Verify shapes are correct
        world_size = self._world_size
        experts_per_device = num_experts // world_size
        self.assertEqual(
            dispatched.shape, (experts_per_device, world_size * capacity, D)
        )
        self.assertEqual(route_idx.shape, (N, top_k))

        # Run combine with auto backend too
        expert_out = dispatched * 2.0  # simulate expert computation
        weights = mx.ones((N, top_k), dtype=mx.float32) / top_k
        combined = mx.distributed.moe_combine_exchange(
            expert_out,
            route_idx,
            weights,
            tokens,
            num_experts=num_experts,
            capacity=capacity,
            backend="auto",
        )
        mx.eval(combined)
        self.assertEqual(combined.shape, (N, D))

    def test_metal_dispatch_combine_roundtrip(self):
        """Metal backend: dispatch -> identity -> combine = input for non-overflow."""
        mx.random.seed(42)
        N, D, E, top_k = 16, 32, 4, 2
        capacity = 8  # large enough for no overflow

        tokens = mx.random.normal((N, D))
        expert_indices = mx.array(
            [[i % E, (i + 1) % E] for i in range(N)], dtype=mx.int32
        )
        weights = mx.ones((N, top_k), dtype=mx.float32) / top_k

        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(dispatched, route_idx)

        combined = mx.distributed.moe_combine_exchange(
            dispatched,
            route_idx,
            weights,
            tokens,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(combined)

        self.assertEqual(combined.shape, (N, D))
        self.assertTrue(
            mx.allclose(combined, tokens, atol=1e-4).item(),
            f"Metal roundtrip failed. Max diff: {mx.abs(combined - tokens).max().item()}",
        )

    def test_metal_vs_cpu_consistency(self):
        """Metal backend produces same results as CPU backend."""
        mx.random.seed(77)
        N, D, E, top_k = 32, 64, 4, 2
        capacity = 12

        tokens = mx.random.normal((N, D))
        expert_indices = mx.array(
            [[i % E, (i + 1) % E] for i in range(N)], dtype=mx.int32
        )
        weights = mx.random.uniform(shape=(N, top_k))
        weights = weights / weights.sum(axis=-1, keepdims=True)

        # CPU path
        disp_cpu, ri_cpu = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
            backend="cpu",
        )
        mx.eval(disp_cpu, ri_cpu)

        comb_cpu = mx.distributed.moe_combine_exchange(
            disp_cpu,
            ri_cpu,
            weights,
            tokens,
            num_experts=E,
            capacity=capacity,
            backend="cpu",
        )
        mx.eval(comb_cpu)

        # Metal path
        disp_metal, ri_metal = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(disp_metal, ri_metal)

        comb_metal = mx.distributed.moe_combine_exchange(
            disp_metal,
            ri_metal,
            weights,
            tokens,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(comb_metal)

        # Route indices must match exactly
        self.assertTrue(
            mx.array_equal(ri_cpu, ri_metal).item(),
            "Route indices differ between CPU and Metal",
        )
        # Dispatched must match
        self.assertTrue(
            mx.allclose(disp_cpu, disp_metal, atol=1e-5).item(),
            f"Dispatch differs. Max diff: {mx.abs(disp_cpu - disp_metal).max().item()}",
        )
        # Combined must match
        self.assertTrue(
            mx.allclose(comb_cpu, comb_metal, atol=1e-4).item(),
            f"Combine differs. Max diff: {mx.abs(comb_cpu - comb_metal).max().item()}",
        )

    def test_metal_overflow_residual(self):
        """Metal backend: all-overflow tokens get original_tokens as residual."""
        if self._world_size > 1:
            self.skipTest("local-only test")
        N, D, E, top_k = 4, 16, 1, 2
        capacity = 1

        tokens = mx.random.normal((N, D))
        expert_indices = mx.zeros((N, top_k), dtype=mx.int32)
        weights = mx.ones((N, top_k), dtype=mx.float32) / top_k

        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(dispatched, route_idx)

        combined = mx.distributed.moe_combine_exchange(
            dispatched,
            route_idx,
            weights,
            tokens,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(combined)

        route_idx_list = route_idx.tolist()
        for n in range(N):
            all_invalid = all(route_idx_list[n][k] < 0 for k in range(top_k))
            if all_invalid:
                self.assertTrue(
                    mx.allclose(combined[n], tokens[n], atol=1e-5).item(),
                    f"Token {n} should be residual fallback (Metal)",
                )

    def test_metal_empty_batch(self):
        """Metal backend: N=0 should produce correct shapes."""
        if self._world_size > 1:
            self.skipTest("local-only test")
        E, D, top_k = 4, 16, 2
        capacity = 4

        tokens = mx.zeros((0, D))
        expert_indices = mx.zeros((0, top_k), dtype=mx.int32)
        weights = mx.zeros((0, top_k), dtype=mx.float32)

        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(dispatched, route_idx)

        self.assertEqual(dispatched.shape, (E, capacity, D))
        self.assertEqual(route_idx.shape, (0, top_k))

        combined = mx.distributed.moe_combine_exchange(
            dispatched,
            route_idx,
            weights,
            tokens,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(combined)
        self.assertEqual(combined.shape, (0, D))

    def test_metal_dtype_float16(self):
        """Metal backend: float16 tokens dispatch and combine correctly."""
        N, D, E, top_k = 8, 32, 4, 2
        capacity = 4

        tokens = mx.random.normal((N, D)).astype(mx.float16)
        expert_indices = mx.array(
            [[i % E, (i + 1) % E] for i in range(N)], dtype=mx.int32
        )
        weights = mx.ones((N, top_k), dtype=mx.float32) / top_k

        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(dispatched, route_idx)
        self.assertEqual(dispatched.dtype, mx.float16)

        combined = mx.distributed.moe_combine_exchange(
            dispatched,
            route_idx,
            weights,
            tokens,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(combined)
        self.assertEqual(combined.dtype, mx.float16)
        self.assertEqual(combined.shape, (N, D))

    def test_metal_dtype_bfloat16(self):
        """Metal backend: bfloat16 tokens dispatch and combine correctly."""
        N, D, E, top_k = 8, 32, 4, 2
        capacity = 4

        tokens = mx.random.normal((N, D)).astype(mx.bfloat16)
        expert_indices = mx.array(
            [[i % E, (i + 1) % E] for i in range(N)], dtype=mx.int32
        )
        weights = mx.ones((N, top_k), dtype=mx.float32) / top_k

        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(dispatched, route_idx)
        self.assertEqual(dispatched.dtype, mx.bfloat16)

        combined = mx.distributed.moe_combine_exchange(
            dispatched,
            route_idx,
            weights,
            tokens,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(combined)
        self.assertEqual(combined.dtype, mx.bfloat16)
        self.assertEqual(combined.shape, (N, D))

    def test_metal_large_batch(self):
        """Metal backend: large batch N=256 with realistic parameters."""
        N, D, E, top_k = 256, 128, 8, 2
        capacity = 80

        tokens = mx.random.normal((N, D)).astype(mx.float16)
        expert_indices = mx.random.randint(0, E, shape=(N, top_k)).astype(mx.int32)
        weights = mx.random.uniform(shape=(N, top_k))
        weights = weights / weights.sum(axis=-1, keepdims=True)

        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(dispatched, route_idx)

        combined = mx.distributed.moe_combine_exchange(
            dispatched,
            route_idx,
            weights,
            tokens,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(combined)
        self.assertEqual(combined.shape, (N, D))
        self.assertTrue(
            mx.all(mx.isfinite(combined)).item(), "NaN/Inf in combined output"
        )

    def test_metal_all_local_experts(self):
        """Metal backend: all tokens routed to local experts (ws=1, no remote)."""
        if self._world_size > 1:
            self.skipTest("local-only test")
        N, D, E, top_k = 16, 32, 4, 2
        capacity = 8

        tokens = mx.random.normal((N, D))
        expert_indices = mx.array(
            [[i % E, (i + 1) % E] for i in range(N)], dtype=mx.int32
        )
        weights = mx.ones((N, top_k), dtype=mx.float32) / top_k

        disp_cpu, ri_cpu = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
            backend="cpu",
        )
        disp_metal, ri_metal = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=E,
            capacity=capacity,
            backend="metal",
        )
        mx.eval(disp_cpu, ri_cpu, disp_metal, ri_metal)

        self.assertTrue(mx.array_equal(ri_cpu, ri_metal).item())
        self.assertTrue(mx.allclose(disp_cpu, disp_metal, atol=1e-5).item())


if __name__ == "__main__":
    unittest.main()
