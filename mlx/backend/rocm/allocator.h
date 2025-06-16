// Copyright © 2025 Apple Inc.

#pragma once

#include <cstddef>

namespace mlx::core::rocm {

void* allocate(size_t size);
void deallocate(void* ptr);

} // namespace mlx::core::rocm