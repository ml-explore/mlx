#pragma once

#include "mlx/backend/cpu/simd/base_simd.h"

#if MLX_SIMD_LIBRARY_VERSION < 6
#include "mlx/backend/cpu/simd/neon_fp16_simd.h"
#endif

namespace mlx::core::simd {

#if MLX_SIMD_LIBRARY_VERSION >= 6
constexpr int N = 8;
template <int N>
struct ScalarT<float16_t, N> {
  using v = _Float16;
};
#endif

template <>
inline constexpr int max_size<float16_t> = N;

#define SIMD_FP16_DEFAULT_UNARY(op)                    \
  template <>                                          \
  inline Simd<float16_t, N> op(Simd<float16_t, N> v) { \
    Simd<float, N> in = v;                             \
    return op(in);                                     \
  }

SIMD_FP16_DEFAULT_UNARY(acos)
SIMD_FP16_DEFAULT_UNARY(acosh)
SIMD_FP16_DEFAULT_UNARY(asin)
SIMD_FP16_DEFAULT_UNARY(asinh)
SIMD_FP16_DEFAULT_UNARY(atan)
SIMD_FP16_DEFAULT_UNARY(atanh)
SIMD_FP16_DEFAULT_UNARY(cosh)
SIMD_FP16_DEFAULT_UNARY(expm1)
SIMD_FP16_DEFAULT_UNARY(log)
SIMD_FP16_DEFAULT_UNARY(log2)
SIMD_FP16_DEFAULT_UNARY(log10)
SIMD_FP16_DEFAULT_UNARY(log1p)
SIMD_FP16_DEFAULT_UNARY(sinh)
SIMD_FP16_DEFAULT_UNARY(tan)
SIMD_FP16_DEFAULT_UNARY(tanh)

#define SIMD_FP16_DEFAULT_BINARY(op)                                         \
  template <>                                                                \
  inline Simd<float16_t, N> op(Simd<float16_t, N> x, Simd<float16_t, N> y) { \
    Simd<float, N> a = x;                                                    \
    Simd<float, N> b = y;                                                    \
    return op(a, b);                                                         \
  }
SIMD_FP16_DEFAULT_BINARY(atan2)
SIMD_FP16_DEFAULT_BINARY(remainder)
SIMD_FP16_DEFAULT_BINARY(pow)

} // namespace mlx::core::simd
