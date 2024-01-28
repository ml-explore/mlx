// Copyright © 2023 Apple Inc.

#include <stdexcept>

#include "mlx/backend/metal/metal.h"

namespace mlx::core::metal {

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

// No cache for CPU only
bool cache_enabled(void) {
  return false;
}
void set_cache_enabled(bool) {}

} // namespace mlx::core::metal
