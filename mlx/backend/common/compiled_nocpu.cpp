// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/common/compiled.h"

namespace mlx::core {

// GPU compile is always available if the GPU is available and since we are in
// this file CPU compile is not available so check if the device is a GPU
// device.
namespace detail {
bool compile_available_for_device(const Device& device) {
  return device == Device::gpu;
}
} // namespace detail

void Compiled::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error(
      "[Compiled::eval_cpu] CPU compialtion not supported on the platform.");
}

} // namespace mlx::core
