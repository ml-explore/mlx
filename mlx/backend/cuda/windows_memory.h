// Copyright © 2026 Apple Inc.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace mlx::core::cu {

size_t compute_wddm_memory_limit(
    size_t memory_limit,
    uint64_t budget,
    uint64_t usage,
    uint64_t pool_reserved);
size_t get_windows_memory_limit(
    size_t hard_memory_limit,
    int device,
    cudaMemPool_t pool);

} // namespace mlx::core::cu
