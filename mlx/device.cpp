// Copyright © 2023 Apple Inc.

#include <stdexcept>

#include "mlx/backend/cpu/available.h"
#include "mlx/backend/gpu/available.h"
#include "mlx/device.h"

#ifdef MLX_USE_ROCM
#include "mlx/backend/rocm/rocm.h"
#endif

namespace mlx::core {

Device& mutable_default_device() {
  Device::DeviceType default_type = Device::cpu;
  if (gpu::is_available()) {
    default_type = Device::gpu;
  }
#ifdef MLX_USE_ROCM
  else if (rocm::is_available()) {
    default_type = Device::gpu; // ROCm devices use the generic gpu type
  }
#endif
  static Device default_device{default_type};
  return default_device;
}

const Device& default_device() {
  return mutable_default_device();
}

void set_default_device(const Device& d) {
  if (!gpu::is_available() && d == Device::gpu) {
    throw std::invalid_argument(
        "[set_default_device] Cannot set gpu device without gpu backend.");
  }
  mutable_default_device() = d;
}

bool operator==(const Device& lhs, const Device& rhs) {
  return lhs.type == rhs.type && lhs.index == rhs.index;
}

bool operator!=(const Device& lhs, const Device& rhs) {
  return !(lhs == rhs);
}

bool is_available(const Device& d) {
  switch (d.type) {
    case Device::cpu:
      return cpu::is_available();
    case Device::gpu:
#ifdef MLX_USE_ROCM
      return gpu::is_available() || rocm::is_available();
#else
      return gpu::is_available();
#endif
  }
  // appease compiler
  return false;
}

} // namespace mlx::core
