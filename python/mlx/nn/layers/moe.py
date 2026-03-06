# Copyright © 2026 Apple Inc.

import math
from typing import NamedTuple, Optional

import mlx.core as mx
from mlx.nn.layers.activations import silu
from mlx.nn.layers.base import Module
from mlx.nn.layers.linear import Linear


class DispatchMeta(NamedTuple):
    """Metadata for expert dispatch/combine round-trip."""

    expert_indices: mx.array  # [N, top_k] expert assignments
    weights: mx.array  # [N, top_k] routing weights
    positions: mx.array  # [N, top_k] slot positions in dispatch buffer
    overflow_mask: mx.array  # [N, 1] True if token overflowed capacity
    num_experts: int
    capacity: int
    world_size: int


class TopKRouter(Module):
    """Top-K expert router with load balancing auxiliary loss.

    Routes each token to the top-k experts based on a learned gate.

    Args:
        hidden_dim: Input hidden dimension.
        num_experts: Total number of experts.
        top_k: Number of experts per token. Default: ``2``.
        capacity_factor: Capacity scaling factor. Default: ``1.25``.
        aux_loss_coeff: Coefficient for load balancing loss. Default: ``0.01``.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coeff: float = 0.01,
    ):
        super().__init__()
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) must not exceed num_experts ({num_experts})"
            )
        self.gate = Linear(hidden_dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.aux_loss_coeff = aux_loss_coeff

    def __call__(self, x: mx.array):
        """Route tokens to experts.

        Args:
            x: Input tensor of shape ``[N, hidden_dim]``.

        Returns:
            Tuple of (weights, indices, aux_loss):
                - weights: ``[N, top_k]`` normalized routing weights
                - indices: ``[N, top_k]`` expert indices (integers in [0, num_experts))
                - aux_loss: scalar load balancing loss
        """
        # x: [N, D] -> logits: [N, num_experts]
        logits = self.gate(x)
        probs = mx.softmax(logits, axis=-1)

        # Get top-k experts per token
        # mx.argpartition gives indices of top-k elements (unordered within top-k)
        # We negate probs to get the largest values
        neg_probs = -probs
        top_k_indices = mx.argpartition(neg_probs, kth=self.top_k - 1, axis=-1)[
            :, : self.top_k
        ]

        # Stop gradient on discrete routing decisions
        expert_indices = mx.stop_gradient(top_k_indices)

        # Gather the weights for selected experts
        # Use take_along_axis to gather probs at the top-k indices
        weights = mx.take_along_axis(probs, expert_indices, axis=-1)

        # Normalize weights so they sum to 1 per token
        weights = weights / mx.maximum(weights.sum(axis=-1, keepdims=True), 1e-6)

        # Compute auxiliary load balancing loss
        aux_loss = self._load_balance_loss(probs, expert_indices)

        return weights, expert_indices, aux_loss

    def _load_balance_loss(self, probs: mx.array, expert_indices: mx.array) -> mx.array:
        """GShard load balancing loss: num_experts * sum(f_e * P_e).

        Args:
            probs: [N, num_experts] routing probabilities.
            expert_indices: [N, top_k] selected expert indices.

        Returns:
            Scalar auxiliary loss.
        """
        num_tokens = probs.shape[0]
        if num_tokens == 0:
            return mx.zeros((), dtype=probs.dtype)

        # f_e: fraction of tokens routed to each expert
        # Create one-hot and sum across top_k selections
        one_hot = mx.zeros_like(probs)
        for k in range(self.top_k):
            indices_k = expert_indices[:, k]  # [N]
            rows = mx.arange(num_tokens)
            one_hot = one_hot.at[rows, indices_k].add(1.0)
        f_e = mx.mean(one_hot, axis=0) / self.top_k  # [num_experts]

        # P_e: mean routing probability per expert
        P_e = mx.mean(probs, axis=0)  # [num_experts]

        # GShard loss
        loss = self.aux_loss_coeff * self.num_experts * mx.sum(f_e * P_e)
        return loss


def _compute_capacity(
    num_tokens: int,
    top_k: int,
    capacity_factor: float,
    num_experts: int,
) -> int:
    """Compute expert buffer capacity."""
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    return max(1, math.ceil(num_tokens * top_k * capacity_factor / num_experts))


class Expert(Module):
    """Single expert network with SwiGLU activation.

    Args:
        hidden_dim: Input/output hidden dimension.
        expert_dim: Expert intermediate dimension.
    """

    def __init__(self, hidden_dim: int, expert_dim: int):
        super().__init__()
        self.w_gate = Linear(hidden_dim, expert_dim, bias=False)
        self.w_up = Linear(hidden_dim, expert_dim, bias=False)
        self.w_down = Linear(expert_dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with SwiGLU activation.

        Args:
            x: Input tensor of shape ``[..., hidden_dim]``.

        Returns:
            Output tensor of shape ``[..., hidden_dim]``.
        """
        return self.w_down(silu(self.w_gate(x)) * self.w_up(x))


def expert_dispatch(
    tokens: mx.array,
    expert_indices: mx.array,
    weights: mx.array,
    num_experts: int,
    capacity_factor: float,
    group: Optional["mx.distributed.Group"] = None,
) -> tuple:
    """Dispatch tokens to experts across devices.

    Args:
        tokens: [N, D] input tokens.
        expert_indices: [N, top_k] expert assignments.
        weights: [N, top_k] routing weights.
        num_experts: Total number of experts across all devices.
        capacity_factor: Capacity scaling factor.
        group: Distributed group. If None, uses local-only dispatch.

    Returns:
        Tuple of (dispatched, meta):
            - dispatched: [experts_per_device, capacity, D] expert inputs for this device
            - meta: DispatchMeta for use with expert_combine
    """
    num_tokens, hidden_dim = tokens.shape
    top_k = expert_indices.shape[1]

    world_size = group.size() if group is not None else 1
    if world_size > 1 and num_experts % world_size != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by "
            f"world_size ({world_size})"
        )
    experts_per_device = num_experts // world_size
    capacity = _compute_capacity(num_tokens, top_k, capacity_factor, num_experts)

    # In distributed mode, synchronize capacity across ranks so all ranks
    # use the same buffer size for all_to_all. Different ranks may have
    # different token counts (e.g. uneven last batch).
    if world_size > 1 and group is not None:
        cap_arr = mx.array(capacity, dtype=mx.int32)
        cap_arr = mx.distributed.all_max(cap_arr, group=group)
        mx.eval(cap_arr)
        capacity = cap_arr.item()

    total_slots = world_size * experts_per_device * capacity
    slots_per_device = experts_per_device * capacity
    dispatch_flat = mx.zeros((total_slots, hidden_dim), dtype=tokens.dtype)

    expert_range = mx.arange(num_experts)
    overflow_mask = mx.zeros((num_tokens, 1), dtype=mx.bool_)
    expert_counts = mx.zeros((num_experts,), dtype=mx.int32)
    pos_columns = []

    zero_idx = mx.array(0, dtype=mx.int32)
    neg_one = mx.array(-1, dtype=mx.int32)
    zero_tokens = mx.zeros_like(tokens)

    for k in range(top_k):
        indices_k = expert_indices[:, k]

        one_hot = indices_k.reshape(-1, 1) == expert_range.reshape(1, -1)
        one_hot_int = one_hot.astype(mx.int32)

        # Per-expert cumulative position, offset by counts from prior k columns
        cum = mx.cumsum(one_hot_int, axis=0) - 1 + expert_counts.reshape(1, -1)

        pos = mx.take_along_axis(cum, indices_k.reshape(-1, 1), axis=1).squeeze(1)

        valid = pos < capacity
        overflow_mask = overflow_mask | (~valid).reshape(-1, 1)

        pos_columns.append(mx.where(valid, pos, neg_one))

        d = indices_k // experts_per_device
        le = indices_k % experts_per_device
        flat_idx = d * slots_per_device + le * capacity + pos
        flat_idx = mx.where(valid, flat_idx, zero_idx).astype(mx.int32)
        scatter_vals = mx.where(valid.reshape(-1, 1), tokens, zero_tokens)
        dispatch_flat = dispatch_flat.at[flat_idx].add(scatter_vals)

        expert_counts = expert_counts + one_hot_int.sum(axis=0)

    positions = mx.stack(pos_columns, axis=1)

    dispatch_buffer = dispatch_flat.reshape(
        world_size, experts_per_device, capacity, hidden_dim
    )

    meta = DispatchMeta(
        expert_indices=expert_indices,
        weights=weights,
        positions=positions,
        overflow_mask=overflow_mask,
        num_experts=num_experts,
        capacity=capacity,
        world_size=world_size,
    )

    # All-to-all exchange if distributed
    if world_size > 1 and group is not None:
        # Materialize scatter graph before distributed exchange
        mx.eval(dispatch_buffer)
        flat = dispatch_buffer.reshape(world_size, -1)
        exchanged = mx.distributed.all_to_all(flat, group=group)
        dispatched = exchanged.reshape(
            world_size, experts_per_device, capacity, hidden_dim
        )
        # Each device processes experts_per_device experts,
        # data from all devices combined
        # Reshape: [world_size, experts_per_device, capacity, D] -> [experts_per_device, world_size * capacity, D]
        dispatched = mx.transpose(dispatched, axes=(1, 0, 2, 3)).reshape(
            experts_per_device, world_size * capacity, hidden_dim
        )
    else:
        # Local only: [1, experts_per_device, capacity, D] -> [experts_per_device, capacity, D]
        dispatched = dispatch_buffer.squeeze(0)

    return dispatched, meta


def expert_combine(
    expert_outputs: mx.array,
    meta: DispatchMeta,
    original_tokens: mx.array,
    group: Optional["mx.distributed.Group"] = None,
) -> mx.array:
    """Combine expert outputs back to token order.

    Args:
        expert_outputs: [experts_per_device, capacity_total, D] expert output tokens.
        meta: DispatchMeta from expert_dispatch.
        original_tokens: [N, D] original input tokens for residual.
        group: Distributed group.

    Returns:
        [N, D] combined output tokens.
    """
    world_size = meta.world_size
    experts_per_device = meta.num_experts // world_size
    capacity = meta.capacity
    hidden_dim = original_tokens.shape[-1]
    num_tokens = original_tokens.shape[0]

    if world_size > 1 and group is not None:
        # Reshape back for all_to_all: [experts_per_device, world_size * capacity, D]
        # -> [world_size, experts_per_device, capacity, D]
        reshaped = expert_outputs.reshape(
            experts_per_device, world_size, capacity, hidden_dim
        )
        reshaped = mx.transpose(reshaped, axes=(1, 0, 2, 3))
        flat = reshaped.reshape(world_size, -1)
        exchanged = mx.distributed.all_to_all(flat, group=group)
        result_buffer = exchanged.reshape(
            world_size, experts_per_device, capacity, hidden_dim
        )
    else:
        result_buffer = expert_outputs.reshape(
            1, experts_per_device, capacity, hidden_dim
        )

    result_flat = result_buffer.reshape(-1, hidden_dim)
    combined = mx.zeros_like(original_tokens)
    top_k = meta.expert_indices.shape[1]
    slots_per_device = experts_per_device * capacity
    zero_idx = mx.array(0, dtype=mx.int32)

    for k in range(top_k):
        indices_k = meta.expert_indices[:, k]
        positions_k = meta.positions[:, k]
        weights_k = meta.weights[:, k]
        device_idx = indices_k // experts_per_device
        local_expert = indices_k % experts_per_device

        flat_idx = device_idx * slots_per_device + local_expert * capacity + positions_k
        valid = positions_k >= 0
        flat_idx = mx.where(valid, flat_idx, zero_idx).astype(mx.int32)

        gathered = result_flat[flat_idx]
        safe_gathered = mx.where(
            valid.reshape(-1, 1), gathered, mx.zeros_like(gathered)
        )
        combined = combined + weights_k.reshape(-1, 1) * safe_gathered

    has_valid_route = (meta.positions >= 0).any(axis=1, keepdims=True)
    combined = mx.where(has_valid_route, combined, original_tokens)

    return combined


class MixtureOfExperts(Module):
    """Mixture of Experts layer with Expert Parallelism support.

    Args:
        hidden_dim: Input/output hidden dimension.
        expert_dim: Expert intermediate dimension.
        num_experts: Total number of experts.
        top_k: Number of experts per token. Default: ``2``.
        capacity_factor: Capacity scaling factor. Default: ``1.25``.
        aux_loss_coeff: Load balance loss coefficient. Default: ``0.01``.
        ep_impl: Expert parallelism implementation to use. One of ``"auto"``,
            ``"python"``, or ``"cpp"``. ``"auto"`` uses the Python vectorized
            path (safe for training; C++ VJP not yet available). ``"cpp"``
            uses the fused C++ primitive (inference-only, no gradient support).
            ``"python"`` always uses the Python vectorized path. Default:
            ``"auto"``.
        ep_backend: Backend for the C++ MoE exchange. One of ``"auto"``,
            ``"cpu"``, or ``"metal"``. ``"auto"`` selects based on workload
            size. Only used when ``ep_impl`` is ``"cpp"`` or ``"auto"`` with
            C++ path active. Default: ``"auto"``.

    Limitations:
        - Inference only: VJP/backward not implemented for C++ fused
          dispatch/combine primitives.
        - ws=2 optimized: world_size > 2 automatically downgrades Metal to
          CPU backend and uses the fixed all_to_all CPU path (one-time
          warning emitted).
        - Expert parallelism is opt-in via ``ep_impl="cpp"`` parameter.
    """

    def __init__(
        self,
        hidden_dim: int,
        expert_dim: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        aux_loss_coeff: float = 0.01,
        ep_impl: str = "auto",
        ep_backend: str = "auto",
    ):
        super().__init__()

        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) must not exceed num_experts ({num_experts})"
            )

        # Determine distributed context
        self._world_size = 1
        self._group = None
        try:
            group = mx.distributed.init(strict=False)
            if group.size() > 1:
                # Probe all_to_all support; some backends (ring, NCCL)
                # do not implement it and would crash on every forward pass.
                try:
                    test = mx.distributed.all_to_all(
                        mx.zeros((group.size(),)), group=group
                    )
                    mx.eval(test)
                    self._world_size = group.size()
                    self._group = group
                except RuntimeError:
                    # Backend doesn't support all_to_all, fall back to local-only
                    pass
        except Exception:
            pass

        if num_experts % self._world_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by "
                f"world_size ({self._world_size})"
            )

        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.ep_impl = ep_impl
        self.ep_backend = ep_backend

        # Router
        self.router = TopKRouter(
            hidden_dim, num_experts, top_k, capacity_factor, aux_loss_coeff
        )

        # Local experts for this device
        experts_per_device = num_experts // self._world_size
        self.experts = [
            Expert(hidden_dim, expert_dim) for _ in range(experts_per_device)
        ]

    def __call__(self, x: mx.array):
        """Forward pass.

        Args:
            x: Input tensor of shape ``[N, hidden_dim]``.

        Returns:
            Tuple of (output, aux_loss):
                - output: ``[N, hidden_dim]`` combined expert outputs
                - aux_loss: scalar load balancing loss
        """
        # Route
        weights, expert_indices, aux_loss = self.router(x)

        # Determine implementation to use
        # auto: use Python path (safe for training; C++ VJP not yet implemented)
        # cpp: use C++ primitive (inference-only, no grad support)
        # python: always use Python vectorized path
        use_cpp = (
            self.ep_impl == "cpp"
            and self._group is not None
            and hasattr(mx.distributed, "moe_dispatch_exchange")
        )

        if use_cpp:
            # C++ fused primitive path (inference-only)
            capacity = _compute_capacity(
                x.shape[0], self.router.top_k, self.capacity_factor, self.num_experts
            )
            # Synchronize capacity across ranks
            if self._world_size > 1:
                cap_arr = mx.array(capacity, dtype=mx.int32)
                cap_arr = mx.distributed.all_max(cap_arr, group=self._group)
                mx.eval(cap_arr)
                capacity = cap_arr.item()

            dispatched, route_idx = mx.distributed.moe_dispatch_exchange(
                x,
                expert_indices.astype(mx.int32),
                num_experts=self.num_experts,
                capacity=capacity,
                group=self._group,
                backend=self.ep_backend,
            )
            route_idx = mx.stop_gradient(route_idx)
            expert_out = self._run_local_experts(dispatched)
            weights_f32 = weights.astype(mx.float32)
            output = mx.distributed.moe_combine_exchange(
                expert_out,
                route_idx,
                weights_f32,
                x,
                num_experts=self.num_experts,
                capacity=capacity,
                group=self._group,
                backend=self.ep_backend,
            )
        else:
            # Python vectorized path (supports grad)
            dispatched, meta = expert_dispatch(
                x,
                expert_indices,
                weights,
                self.num_experts,
                self.capacity_factor,
                group=self._group,
            )
            expert_out = self._run_local_experts(dispatched)
            output = expert_combine(
                expert_out,
                meta,
                x,
                group=self._group,
            )

        return output, aux_loss

    def _run_local_experts(self, dispatched: mx.array) -> mx.array:
        """Run local experts on dispatched tokens.

        Args:
            dispatched: [experts_per_device, capacity_total, D] dispatched inputs.

        Returns:
            [experts_per_device, capacity_total, D] expert outputs.
        """
        outputs = []
        for i, expert in enumerate(self.experts):
            expert_input = dispatched[i]  # [capacity_total, D]
            expert_output = expert(expert_input)  # [capacity_total, D]
            outputs.append(expert_output)
        return mx.stack(outputs, axis=0)
