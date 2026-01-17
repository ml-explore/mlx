// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx {
namespace core {
class array;
} // namespace core
} // namespace mlx

namespace mlx::core::cpu {

void eval(array& arr);

} // namespace mlx::core::cpu
