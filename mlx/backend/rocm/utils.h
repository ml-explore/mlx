// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// Utility function to check HIP errors
void check_hip_error(const char* msg, hipError_t error);

} // namespace mlx::core::rocm