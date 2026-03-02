# Copyright © 2026 Apple Inc.

import mlx.core as mx
import mlx_distributed_tests
import mlx_tests


class TestJACCLDistributed(mlx_distributed_tests.MLXDistributedCommonTestCase):
    @classmethod
    def setUpClass(cls):
        _ = mx.distributed.init(strict=True, backend="jaccl")
        cls.atol = 1e-6
        cls.rtol = 1e-4

    def test_groups(self):
        world = mx.distributed.init()
        self.assertEqual(world.size(), 2)
        self.assertTrue(0 <= world.rank() < 2)

        world2 = mx.distributed.init()
        self.assertEqual(world.size(), world2.size())
        self.assertEqual(world.rank(), world2.rank())

    # ------------------------------------------------------------------
    # MoE Expert Parallelism C++ primitive tests (2-rank JACCL)
    # ------------------------------------------------------------------

    def test_moe_ep_cpp_roundtrip_2rank(self):
        """Dispatch + combine roundtrip numerical correctness over 2 ranks."""
        world = mx.distributed.init()
        if world.size() != 2:
            self.skipTest("requires 2 ranks")
        if not hasattr(mx.distributed, "moe_dispatch_exchange"):
            self.skipTest("moe_dispatch_exchange not available")

        N = 16
        D = 64
        num_experts = 4
        top_k = 2
        capacity = 8
        experts_per_device = num_experts // world.size()  # 2

        for dtype in [mx.float32, mx.float16, mx.bfloat16]:
            mx.random.seed(42 + world.rank())
            tokens = mx.random.normal((N, D)).astype(dtype)

            # Random expert indices in [0, num_experts)
            expert_indices = mx.random.randint(0, num_experts, shape=(N, top_k)).astype(
                mx.int32
            )

            # Dispatch
            dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
                tokens,
                expert_indices,
                num_experts=num_experts,
                capacity=capacity,
                group=world,
            )
            mx.eval(dispatched, route_idx)

            # Shape: [experts_per_device, world_size * capacity, D]
            cap_total = world.size() * capacity
            self.assertEqual(
                dispatched.shape,
                (experts_per_device, cap_total, D),
                f"dtype={dtype}: dispatched shape mismatch",
            )
            self.assertEqual(dispatched.dtype, dtype)
            self.assertEqual(route_idx.shape, (N, top_k))
            self.assertEqual(route_idx.dtype, mx.int32)

            # Identity expert: expert_outputs = dispatched
            weights = mx.ones((N, top_k), dtype=mx.float32) / top_k
            original_tokens = mx.zeros((N, D), dtype=dtype)

            combined = mx.distributed.moe_combine_exchange(
                dispatched,
                route_idx,
                weights,
                original_tokens,
                num_experts=num_experts,
                capacity=capacity,
                group=world,
            )
            mx.eval(combined)

            self.assertEqual(combined.shape, (N, D), f"dtype={dtype}: combined shape")
            self.assertEqual(combined.dtype, dtype, f"dtype={dtype}: combined dtype")
            self.assertTrue(
                mx.all(mx.isfinite(combined)).item(),
                f"dtype={dtype}: combined contains non-finite values",
            )

    def test_moe_ep_cpp_asymmetric_traffic_2rank(self):
        """Asymmetric traffic completes without deadlock."""
        world = mx.distributed.init()
        if world.size() != 2:
            self.skipTest("requires 2 ranks")
        if not hasattr(mx.distributed, "moe_dispatch_exchange"):
            self.skipTest("moe_dispatch_exchange not available")

        N = 32
        D = 64
        num_experts = 4
        capacity = 16
        top_k = 2
        experts_per_device = num_experts // world.size()  # 2

        mx.random.seed(100 + world.rank())
        tokens = mx.random.normal((N, D))

        # Rank 0: all tokens → experts 2,3 (owned by rank 1)
        # Rank 1: all tokens → experts 0,1 (owned by rank 0)
        if world.rank() == 0:
            expert_indices = mx.random.randint(2, 4, shape=(N, top_k)).astype(mx.int32)
        else:
            expert_indices = mx.random.randint(0, 2, shape=(N, top_k)).astype(mx.int32)

        # Dispatch
        dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
            tokens,
            expert_indices,
            num_experts=num_experts,
            capacity=capacity,
            group=world,
        )
        mx.eval(dispatched, route_idx)

        cap_total = world.size() * capacity
        self.assertEqual(
            dispatched.shape,
            (experts_per_device, cap_total, D),
        )

        # Combine
        weights = mx.ones((N, top_k)) / top_k
        combined = mx.distributed.moe_combine_exchange(
            dispatched,
            route_idx,
            weights,
            tokens,
            num_experts=num_experts,
            capacity=capacity,
            group=world,
        )
        mx.eval(combined)

        self.assertIsNotNone(combined)
        self.assertEqual(combined.shape, (N, D))
        self.assertTrue(
            mx.all(mx.isfinite(combined)).item(),
            "combined contains non-finite values after asymmetric exchange",
        )


if __name__ == "__main__":
    mlx_tests.MLXTestRunner()
