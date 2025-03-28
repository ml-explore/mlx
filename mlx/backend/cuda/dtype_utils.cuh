// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/dtype_utils.h"

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

// Like MLX_SWITCH_XXX_TYPES but use CUDA types.
#define MLX_SWITCH_CUDA_TYPES(TYPE, CTYPE_ALIAS, ...)           \
  MLX_SWITCH_ALL_TYPES(TYPE, CTYPE_NATIVE, [&]() {              \
    using CTYPE_ALIAS = ::mlx::core::cuda_type_t<CTYPE_NATIVE>; \
    return __VA_ARGS__();                                       \
  })

} // namespace mlx::core
