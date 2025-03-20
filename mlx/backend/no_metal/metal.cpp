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

// No-ops when Metal is not available.
size_t get_active_memory() {
  return 0;
}
size_t get_peak_memory() {
  return 0;
}
void reset_peak_memory() {}
size_t get_cache_memory() {
  return 0;
}
size_t set_memory_limit(size_t) {
  return 0;
}
size_t get_memory_limit() {
  return 0;
}
size_t set_cache_limit(size_t) {
  return 0;
}
size_t set_wired_limit(size_t) {
  return 0;
}

void start_capture(std::string) {}
void stop_capture() {}
void clear_cache() {}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  throw std::runtime_error(
      "[metal::device_info] Cannot get device info without metal backend");
};

} // namespace mlx::core::metal
