// Copyright © 2025 Apple Inc.

// This file includes host-only utilies for writing CUDA kernels, the difference
// from backend/cuda/kernels/utils.cuh is that the latter file only include
// device-only code.

#pragma once

#include "mlx/array.h"

#include <cuComplex.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace mlx::core {

// Maps CPU types to CUDA types.
template <typename T>
struct CTypeToCudaType {
  using type = T;
};

template <>
struct CTypeToCudaType<float16_t> {
  using type = __half;
};

template <>
struct CTypeToCudaType<bfloat16_t> {
  using type = __nv_bfloat16;
};

template <>
struct CTypeToCudaType<complex64_t> {
  using type = cuComplex;
};

template <typename T>
using cuda_type_t = typename CTypeToCudaType<T>::type;

// Compute the grid and block dimensions, check backend/common/utils.h for docs.
dim3 get_block_dims(int dim0, int dim1, int dim2, int pow2 = 10);
dim3 get_2d_grid_dims(const Shape& shape, const Strides& strides);
dim3 get_2d_grid_dims(
    const Shape& shape,
    const Strides& strides,
    size_t divisor);

} // namespace mlx::core
