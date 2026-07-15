// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

#include <cstddef>
#include <vector>

namespace mlx::core::rocm {

/* Check if the ROCm backend is available. */
MLX_API bool is_available();

// Deterministic bump arena (shared decode/train region). Opt-in for future
// HIP-graph train capture — does NOT enable graphs by itself.
// capacity_bytes: backing HBM region; returns false on alloc failure.
MLX_API bool train_arena_begin(size_t capacity_bytes);
MLX_API void train_arena_reset();
MLX_API void train_arena_end();
MLX_API bool train_arena_active();
MLX_API size_t train_arena_high_water();
MLX_API bool train_arena_overflowed();

// Fused sorted-MoE SwiGLU (one D2H sync for the whole gate/up/silu/down).
// x: [T,D] bf16
// w_gate, w_up: [E,D,I] bf16  (lemonseed gather_mm layout after swapaxes)
// w_down: [E,I,D] bf16
// expert_ids: [T] uint32 sorted by expert id
// returns y: [T,D] bf16
MLX_API array moe_swiglu_sorted(
    const array& x,
    const array& w_gate,
    const array& w_up,
    const array& w_down,
    const array& expert_ids,
    StreamOrDevice s = {});

// Fused sorted-MoE SwiGLU VJP (one D2H sync for recompute + all grads).
// x: [T,D]  w_gate/up: [E,D,I]  w_down: [E,I,D]  ids: [T]  dy: [T,D]  (bf16)
// returns {dx[T,D], dw_gate[E,D,I], dw_up[E,D,I], dw_down[E,I,D]}
MLX_API std::vector<array> moe_swiglu_sorted_vjp(
    const array& x,
    const array& w_gate,
    const array& w_up,
    const array& w_down,
    const array& expert_ids,
    const array& dy,
    StreamOrDevice s = {});

} // namespace mlx::core::rocm
