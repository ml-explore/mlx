// Copyright © 2025 Apple Inc.

#pragma once

// ROCm/HIP specific configuration
#define ROCM_MAX_THREADS_PER_BLOCK 1024
#define ROCM_WARP_SIZE 64
#define ROCM_MAX_BLOCKS_PER_GRID 65535

namespace mlx::core::rocm {
constexpr int kMaxThreadsPerBlock = ROCM_MAX_THREADS_PER_BLOCK;
constexpr int kWarpSize = ROCM_WARP_SIZE;
constexpr int kMaxBlocksPerGrid = ROCM_MAX_BLOCKS_PER_GRID;
} // namespace mlx::core::rocm