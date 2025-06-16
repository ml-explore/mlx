// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/utils.h"
#include <sstream>
#include <stdexcept>

namespace mlx::core::rocm {

void check_hip_error(const char* msg, hipError_t error) {
  if (error != hipSuccess) {
    std::ostringstream oss;
    oss << "[ROCm] " << msg << ": " << hipGetErrorString(error);
    throw std::runtime_error(oss.str());
  }
}

} // namespace mlx::core::rocm