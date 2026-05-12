// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::cpu {

void new_stream(Stream s);
void eval(array& arr);
void clear_streams();

} // namespace mlx::core::cpu
