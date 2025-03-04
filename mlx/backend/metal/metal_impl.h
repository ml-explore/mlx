// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <future>
#include <memory>
#include <unordered_set>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::metal {

void new_stream(Stream stream);

std::unique_ptr<void, std::function<void(void*)>> new_scoped_memory_pool();

void eval(array& arr);
void finalize(
    Stream s,
    std::unordered_set<std::shared_ptr<array::Data>> retain_buffers,
    bool force_commit);
void synchronize(Stream s);

} // namespace mlx::core::metal
