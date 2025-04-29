// Copyright © 2025 Apple Inc.

#include <stdexcept>

#include "mlx/backend/metal/metal.h"

namespace mlx::core::metal {

bool is_available() {
  return false;
}

void start_capture(std::string) {}
void stop_capture() {}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  throw std::runtime_error(
      "[metal::device_info] Cannot get device info without metal backend");
};

} // namespace mlx::core::metal
