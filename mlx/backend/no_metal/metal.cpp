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

std::function<void()> make_task(array arr, bool signal) {
  throw std::runtime_error(
      "[metal::make_task] Cannot make GPU task without metal backend");
}

std::function<void()> make_synchronize_task(
    Stream s,
    std::shared_ptr<std::promise<void>> p) {
  throw std::runtime_error(
      "[metal::make_synchronize_task] Cannot synchronize GPU"
      " without metal backend");
}

// No-ops when Metal is not available.
size_t get_active_memory() {
  return 0;
}
size_t get_peak_memory() {
  return 0;
}
size_t get_cache_memory() {
  return 0;
}
size_t set_memory_limit(size_t, bool) {
  return 0;
}
size_t set_cache_limit(size_t) {
  return 0;
}
void start_capture(std::string path) {}
void stop_capture() {}
void clear_cache() {}

} // namespace mlx::core::metal
