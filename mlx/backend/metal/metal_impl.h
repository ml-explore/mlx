// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <memory>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::metal {

void new_stream(Stream stream);
std::shared_ptr<void> new_scoped_memory_pool();

std::function<void()> make_task(array arr, bool flush);

} // namespace mlx::core::metal
