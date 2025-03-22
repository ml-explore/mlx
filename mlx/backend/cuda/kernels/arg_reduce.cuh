// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/kernels/utils.cuh"

#include <cooperative_groups.h>

namespace mlx::core::mxcuda {

namespace cg = cooperative_groups;

template <typename U>
struct IndexValPair {
  uint32_t index;
  U val;
};

} // namespace mlx::core::mxcuda
