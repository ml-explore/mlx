// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core {

namespace cu {

constexpr int RADIX_BITS = 8;
constexpr int RADIX = 1 << RADIX_BITS;
constexpr int N_PER_THREAD = 4;
constexpr int BLOCK_THREADS = 256;
constexpr int TILE_SIZE = BLOCK_THREADS * N_PER_THREAD;

} // namespace cu

void radix_argsort(const Stream& s, const array& in, array& out);

} // namespace mlx::core
