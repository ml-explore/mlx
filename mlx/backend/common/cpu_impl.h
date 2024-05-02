// Copyright © 2024 Apple Inc.

#pragma once

#include <functional>
#include <future>
#include <memory>

#include "mlx/array.h"

namespace mlx::core::cpu {

std::function<void()> make_task(array arr, bool signal);
std::function<void()> make_synchronize_task(
    Stream s,
    std::shared_ptr<std::promise<void>> p);

} // namespace mlx::core::cpu
