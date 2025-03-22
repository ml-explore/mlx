// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"

#include <cuda_runtime.h>
#include <type_traits>

namespace mlx::core {

template <typename T, typename U>
inline auto ceil_div(T a, U b) {
  return (a + (b - 1)) / b;
}

dim3 get_2d_num_blocks(
    const Shape& shape,
    const Strides& strides,
    size_t num_threads);

std::string get_primitive_string(Primitive* primitive);

void check_cuda_error(const char* name, cudaError_t err);

// Throw exception if the cuda API does not succeed.
#define CHECK_CUDA_ERROR(cmd) check_cuda_error(#cmd, (cmd))

} // namespace mlx::core
