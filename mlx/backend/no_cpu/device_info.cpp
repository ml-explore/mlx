// Copyright © 2026 Apple Inc.

#include "mlx/backend/cpu/device_info.h"

namespace mlx::core::cpu {

bool is_available() {
  return false;
}

int device_count() {
  return 0;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int /* device_index */) {
  static std::unordered_map<std::string, std::variant<std::string, size_t>>
      empty;
  return empty;
}

} // namespace mlx::core::cpu
