// Copyright © 2026 Apple Inc.

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/metal/metal.h"

namespace mlx::core::gpu {

bool is_available() {
  return metal::is_available();
}

int device_count() {
  return 1;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int /* device_index */) {
  return metal::device_info();
}

} // namespace mlx::core::gpu
