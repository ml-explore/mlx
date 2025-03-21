// Copyright © 2023-2024 Apple Inc.

#include <stdexcept>

#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/metal_impl.h"
namespace mlx::core::metal {

bool is_available() {
  return false;
}

void new_stream(Stream) {}

std::unique_ptr<void, std::function<void(void*)>> new_scoped_memory_pool() {
  return nullptr;
}

void eval(array&) {
  throw std::runtime_error(
      "[metal::eval] Cannot eval on GPU without metal backend");
}

void finalize(Stream) {
  throw std::runtime_error(
      "[metal::finalize] Cannot finalize GPU without metal backend");
}

void synchronize(Stream) {
  throw std::runtime_error(
      "[metal::synchronize] Cannot synchronize GPU without metal backend");
}

void start_capture(std::string) {}
void stop_capture() {}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  throw std::runtime_error(
      "[metal::device_info] Cannot get device info without metal backend");
};

} // namespace mlx::core::metal
