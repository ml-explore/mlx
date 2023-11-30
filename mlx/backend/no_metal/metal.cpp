// Copyright © 2023 Apple Inc.

#include <stdexcept>

#include "mlx/backend/metal/metal.h"

namespace mlx::core::metal {

void new_stream(Stream) {}

std::function<void()> make_task(
    array& arr,
    std::vector<std::shared_future<void>> deps,
    std::shared_ptr<std::promise<void>> p,
    bool retain_graph) {
  throw std::runtime_error(
      "[metal::make_task] Cannot make GPU task without metal backend");
}

} // namespace mlx::core::metal
