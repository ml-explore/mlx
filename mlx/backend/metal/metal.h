// Copyright © 2023 Apple Inc.

#pragma once

#include <future>
#include <memory>
#include <vector>

#include "mlx/array.h"
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

std::function<void()> make_task(
    array& arr,
    std::vector<std::shared_future<void>> deps,
    std::shared_ptr<std::promise<void>> p,
    bool retain_graph);

} // namespace mlx::core::metal
