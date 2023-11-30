// Copyright © 2023 Apple Inc.

#include "mlx/device.h"
#include "mlx/backend/metal/metal.h"

namespace mlx::core {

static Device default_device_{
    metal::is_available() ? Device::gpu : Device::cpu};

const Device& default_device() {
  return default_device_;
}

void set_default_device(const Device& d) {
  if (!metal::is_available() && d == Device::gpu) {
    throw std::invalid_argument(
        "[set_default_device] Cannot set gpu device without gpu backend.");
  }
  default_device_ = d;
}

bool operator==(const Device& lhs, const Device& rhs) {
  return lhs.type == rhs.type && lhs.index == rhs.index;
}

bool operator!=(const Device& lhs, const Device& rhs) {
  return !(lhs == rhs);
}

} // namespace mlx::core
