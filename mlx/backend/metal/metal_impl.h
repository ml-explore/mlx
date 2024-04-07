// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <future>
#include <memory>
#include <vector>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::metal {

void new_stream(Stream stream);
std::shared_ptr<void> new_scoped_memory_pool();

std::function<void()> make_task(
    array& arr,
    std::vector<std::shared_future<void>> deps,
    std::shared_ptr<std::promise<void>> p);

} // namespace mlx::core::metal
