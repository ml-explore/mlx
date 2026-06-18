// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/rocm.h"
#include "mlx/fast.h"

namespace mlx::core {

namespace rocm {

bool is_available() {
  return false;
}

} // namespace rocm

namespace fast {

CustomKernelFunction hip_kernel(
    const std::string&,
    const std::vector<std::string>&,
    const std::vector<std::string>&,
    const std::string&,
    const std::string&,
    bool,
    int,
    std::vector<std::pair<int, int>>) {
  throw std::runtime_error("[hip_kernel] No ROCm back-end.");
}

} // namespace fast

} // namespace mlx::core
