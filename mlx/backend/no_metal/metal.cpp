// Copyright © 2023-2024 Apple Inc.

#include <stdexcept>

#include "mlx/backend/metal/metal.h"

namespace mlx::core::metal {

bool is_available() {
  return false;
}

void new_stream(Stream) {}
std::shared_ptr<void> new_scoped_memory_pool() {
  return nullptr;
}

std::function<void()> make_task(
    array& arr,
    std::vector<std::shared_future<void>> deps,
    std::shared_ptr<std::promise<void>> p) {
  throw std::runtime_error(
      "[metal::make_task] Cannot make GPU task without metal backend");
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

} // namespace mlx::core::metal
