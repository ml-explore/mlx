// Copyright © 2025 Apple Inc.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

namespace mlx::core::cu {

///////////////////////////////////////////////////////////////////////////////
// Unary ops for half types.
///////////////////////////////////////////////////////////////////////////////

#if CUDART_VERSION < 12000 && __CUDA_ARCH__ < 800
#define MLX_DEFINE_UNARY_OP(NAME, HALF_OP)           \
  template <typename T>                              \
  __forceinline__ __device__ auto NAME(T x) {        \
    if constexpr (cuda::std::is_same_v<T, __half>) { \
      return HALF_OP(x);                             \
    } else {                                         \
      return ::NAME(x);                              \
    }                                                \
  }
#else
#define MLX_DEFINE_UNARY_OP(NAME, HALF_OP)                         \
  template <typename T>                                            \
  __forceinline__ __device__ auto NAME(T x) {                      \
    if constexpr (cuda::std::is_same_v<T, __half>) {               \
      return HALF_OP(x);                                           \
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) { \
      return HALF_OP(x);                                           \
    } else {                                                       \
      return ::NAME(x);                                            \
    }                                                              \
  }
#endif

#define MLX_DEFINE_UNARY_OP_FALLBCK(NAME)                          \
  template <typename T>                                            \
  __forceinline__ __device__ auto NAME(T x) {                      \
    if constexpr (cuda::std::is_same_v<T, __half>) {               \
      return ::NAME(__half2float(x));                              \
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) { \
      return ::NAME(__bfloat162float(x));                          \
    } else {                                                       \
      return ::NAME(x);                                            \
    }                                                              \
  }

MLX_DEFINE_UNARY_OP(abs, __habs)
MLX_DEFINE_UNARY_OP(ceil, hceil)
MLX_DEFINE_UNARY_OP(cos, hcos)
MLX_DEFINE_UNARY_OP(exp, hexp)
MLX_DEFINE_UNARY_OP(floor, hfloor)
MLX_DEFINE_UNARY_OP(isnan, __hisnan)
MLX_DEFINE_UNARY_OP(log, hlog)
MLX_DEFINE_UNARY_OP(log2, hlog2)
MLX_DEFINE_UNARY_OP(log10, hlog10)
MLX_DEFINE_UNARY_OP(rint, hrint)
MLX_DEFINE_UNARY_OP(rsqrt, hrsqrt)
MLX_DEFINE_UNARY_OP(sin, hsin)
MLX_DEFINE_UNARY_OP(sqrt, hsqrt)
MLX_DEFINE_UNARY_OP_FALLBCK(acos)
MLX_DEFINE_UNARY_OP_FALLBCK(acosh)
MLX_DEFINE_UNARY_OP_FALLBCK(asin)
MLX_DEFINE_UNARY_OP_FALLBCK(asinh)
MLX_DEFINE_UNARY_OP_FALLBCK(atan)
MLX_DEFINE_UNARY_OP_FALLBCK(atanh)
MLX_DEFINE_UNARY_OP_FALLBCK(cosh)
MLX_DEFINE_UNARY_OP_FALLBCK(log1p)
MLX_DEFINE_UNARY_OP_FALLBCK(sinh)
MLX_DEFINE_UNARY_OP_FALLBCK(tan)
#if __CUDA_ARCH__ >= 1280
MLX_DEFINE_UNARY_OP(tanh, htanh)
#else
MLX_DEFINE_UNARY_OP_FALLBCK(tanh)
#endif

#undef MLX_DEFINE_UNARY_OP
#undef MLX_DEFINE_UNARY_OP_FALLBCK

///////////////////////////////////////////////////////////////////////////////
// Additional C++ operator overrides between half types and native types.
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U>
constexpr bool is_integral_except =
    cuda::std::is_integral_v<T> && !cuda::std::is_same_v<T, U>;

template <typename T, typename U>
constexpr bool is_arithmetic_except =
    cuda::std::is_arithmetic_v<T> && !cuda::std::is_same_v<T, U>;

#define MLX_DEFINE_HALF_OP(HALF, HALF2FLOAT, FLOAT2HALF, OP)          \
  template <                                                          \
      typename T,                                                     \
      typename = cuda::std::enable_if_t<is_integral_except<T, HALF>>> \
  __forceinline__ __device__ HALF operator OP(HALF x, T y) {          \
    return FLOAT2HALF(HALF2FLOAT(x) OP static_cast<float>(y));        \
  }                                                                   \
  template <                                                          \
      typename T,                                                     \
      typename = cuda::std::enable_if_t<is_integral_except<T, HALF>>> \
  __forceinline__ __device__ HALF operator OP(T x, HALF y) {          \
    return FLOAT2HALF(static_cast<float>(x) OP HALF2FLOAT(y));        \
  }

#define MLX_DEFINE_HALF_CMP(HALF, HALF2FLOAT, OP)                       \
  template <                                                            \
      typename T,                                                       \
      typename = cuda::std::enable_if_t<is_arithmetic_except<T, HALF>>> \
  __forceinline__ __device__ bool operator OP(HALF x, T y) {            \
    return HALF2FLOAT(x) OP static_cast<float>(y);                      \
  }                                                                     \
  template <                                                            \
      typename T,                                                       \
      typename = cuda::std::enable_if_t<is_arithmetic_except<T, HALF>>> \
  __forceinline__ __device__ bool operator OP(T x, HALF y) {            \
    return static_cast<float>(y) OP HALF2FLOAT(x);                      \
  }

MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, +)
MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, -)
MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, *)
MLX_DEFINE_HALF_OP(__half, __half2float, __float2half, /)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, +)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, -)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, *)
MLX_DEFINE_HALF_OP(__nv_bfloat16, __bfloat162float, __float2bfloat16, /)
MLX_DEFINE_HALF_CMP(__half, __half2float, <)
MLX_DEFINE_HALF_CMP(__half, __half2float, >)
MLX_DEFINE_HALF_CMP(__half, __half2float, <=)
MLX_DEFINE_HALF_CMP(__half, __half2float, >=)
MLX_DEFINE_HALF_CMP(__half, __half2float, ==)
MLX_DEFINE_HALF_CMP(__half, __half2float, !=)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, <)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, >)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, <=)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, >=)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, ==)
MLX_DEFINE_HALF_CMP(__nv_bfloat16, __bfloat162float, !=)

#undef MLX_DEFINE_HALF_OP
#undef MLX_DEFINE_HALF_CMP

} // namespace mlx::core::cu
