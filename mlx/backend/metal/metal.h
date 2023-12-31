// Copyright © 2023 Apple Inc.

#pragma once

#include <future>
#include <memory>
#include <vector>

#include "mlx/graph.h"
#include "mlx/stream.h"

namespace mlx::core::metal {

constexpr bool is_available() {
#ifdef _METAL_
  return true;
#else
  return false;
#endif
}

void new_stream(Stream stream);
std::shared_ptr<void> new_scoped_memory_pool();

std::function<void()> make_task(
    GraphNode g,
    std::vector<std::shared_future<void>> deps,
    std::shared_ptr<std::promise<void>> p);

} // namespace mlx::core::metal
