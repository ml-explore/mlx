// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"

#include <cuda_runtime.h>
#include <type_traits>

namespace mlx::core {

template <typename T>
inline constexpr bool is_floating_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

// Throw exception if the cuda API does not succeed.
void check_cuda_error(const char* name, cudaError_t err);

#define CHECK_CUDA_ERROR(cmd) check_cuda_error(#cmd, (cmd))

// Return the 3d block_dim fit for total_threads.
dim3 get_block_dim(dim3 total_threads, int pow2 = 10);

std::string get_primitive_string(Primitive* primitive);

} // namespace mlx::core
