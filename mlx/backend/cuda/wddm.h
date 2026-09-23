// Copyright © 2026 Apple Inc.

#pragma once

#include <cuda_runtime.h>

#include <limits>

namespace mlx::core::cu {

inline size_t memory_limit_from_budget(
    uint64_t budget,
    uint64_t usage,
    uint64_t pool_reserved) {
  auto non_pool_usage = usage > pool_reserved ? usage - pool_reserved : 0;
  auto margin = budget / 20;
  if ((non_pool_usage >= budget) || (margin >= budget - non_pool_usage)) {
    return 0;
  }
  uint64_t size_limit = std::numeric_limits<size_t>::max();
  return std::min(budget - non_pool_usage - margin, size_limit);
}

size_t get_wddm_memory_limit(int device, cudaMemPool_t pool);

} // namespace mlx::core::cu
