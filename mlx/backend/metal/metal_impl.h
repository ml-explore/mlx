// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <future>
#include <memory>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::metal {

void new_stream(Stream stream);

std::unique_ptr<void, std::function<void(void*)>> new_scoped_memory_pool();

std::function<void()> make_task(array arr, bool signal);

std::function<void()> make_synchronize_task(
    Stream s,
    std::shared_ptr<std::promise<void>> p);

} // namespace mlx::core::metal
