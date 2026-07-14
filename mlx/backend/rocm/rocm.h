// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/api.h"

#include <cstddef>

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

} // namespace mlx::core::rocm
