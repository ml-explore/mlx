// Copyright © 2025 Apple Inc.

#pragma once

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

} // namespace mlx::core
