// Copyright © 2023 Apple Inc.

#include "mlx/device.h"
#include "mlx/backend/metal/metal.h"

namespace mlx::core {

Device& mutable_default_device() {
  static Device default_device{
      metal::is_available() ? Device::gpu : Device::cpu};
  return default_device;
}

const Device& default_device() {
  return mutable_default_device();
}

void set_default_device(const Device& d) {
  if (!metal::is_available() && d == Device::gpu) {
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

} // namespace mlx::core
