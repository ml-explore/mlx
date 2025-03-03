// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <future>
#include <memory>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::metal {

void new_stream(Stream stream);

std::unique_ptr<void, std::function<void(void*)>> new_scoped_memory_pool();

void eval(array& arr);
void finalize(Stream s);
void synchronize(Stream s);

} // namespace mlx::core::metal
