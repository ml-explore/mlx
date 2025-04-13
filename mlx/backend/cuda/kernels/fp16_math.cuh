// Copyright © 2025 Apple Inc.

#pragma once

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
// Binary ops for half types.
///////////////////////////////////////////////////////////////////////////////

#if CUDART_VERSION < 12000 && __CUDA_ARCH__ < 800
#define MLX_DEFINE_BINARY_OP(NAME, HALF_OP)          \
  template <typename T>                              \
  __forceinline__ __device__ auto NAME(T x, T y) {   \
    if constexpr (cuda::std::is_same_v<T, __half>) { \
      return HALF_OP(x, y);                          \
    } else {                                         \
      return ::NAME(x, y);                           \
    }                                                \
  }
#else
#define MLX_DEFINE_BINARY_OP(NAME, HALF_OP)                        \
  template <typename T>                                            \
  __forceinline__ __device__ auto NAME(T x, T y) {                 \
    if constexpr (cuda::std::is_same_v<T, __half>) {               \
      return HALF_OP(x, y);                                        \
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) { \
      return HALF_OP(x, y);                                        \
    } else {                                                       \
      return ::NAME(x, y);                                         \
    }                                                              \
  }
#endif

MLX_DEFINE_BINARY_OP(max, __hmax)
MLX_DEFINE_BINARY_OP(min, __hmin)

#undef MLX_DEFINE_BINARY_OP

template <typename T>
__forceinline__ __device__ T fmod(T x, T y) {
  if constexpr (cuda::std::is_same_v<T, __half>) {
    return __float2half(::fmod(__half2float(x), __half2float(y)));
#if CUDART_VERSION >= 12000 || __CUDA_ARCH__ >= 800
  } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
    return __float2bfloat16(::fmod(__bfloat162float(x), __bfloat162float(y)));
#endif
  } else {
    return ::fmod(x, y);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Missing C++ operator overrides for CUDA 7.
///////////////////////////////////////////////////////////////////////////////

#if CUDART_VERSION < 12000 && __CUDA_ARCH__ < 800

#define MLX_DEFINE_BF16_OP(OP)                                           \
  __forceinline__ __device__ __nv_bfloat16 operator OP(                  \
      __nv_bfloat16 x, __nv_bfloat16 y) {                                \
    return __float2bfloat16(__bfloat162float(x) OP __bfloat162float(y)); \
  }

#define MLX_DEFINE_BF16_CMP(OP)                                          \
  __forceinline__ __device__ bool operator OP(                           \
      __nv_bfloat16 x, __nv_bfloat16 y) {                                \
    return __float2bfloat16(__bfloat162float(x) OP __bfloat162float(y)); \
  }

MLX_DEFINE_BF16_OP(+)
MLX_DEFINE_BF16_OP(-)
MLX_DEFINE_BF16_OP(*)
MLX_DEFINE_BF16_OP(/)
MLX_DEFINE_BF16_CMP(>)
MLX_DEFINE_BF16_CMP(<)
MLX_DEFINE_BF16_CMP(>=)
MLX_DEFINE_BF16_CMP(<=)

#undef MLX_DEFINE_BF16_OP
#undef MLX_DEFINE_BF16_CMP

#endif // CUDART_VERSION < 12000 && __CUDA_ARCH__ < 800

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
