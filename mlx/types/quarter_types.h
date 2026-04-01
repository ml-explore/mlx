// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/types/half_types.h"

#include "mlx/types/fp8.h"
namespace mlx::core {
typedef struct _MLX_Float8 float8_t;
} // namespace mlx::core

#include "mlx/types/bf8.h"
namespace mlx::core {
typedef struct _MLX_BFloat8 bfloat8_t;
} // namespace mlx::core

namespace mlx::core {

// clang-format off
#define fp8_bf8_binop_helper(__op__, __operator__)                    \
  inline float __operator__(float8_t lhs, bfloat8_t rhs) {            \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs);    \
  }                                                                    \
  inline float __operator__(bfloat8_t lhs, float8_t rhs) {            \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs);    \
  }

fp8_bf8_binop_helper(+, operator+)
fp8_bf8_binop_helper(-, operator-)
fp8_bf8_binop_helper(*, operator*)
fp8_bf8_binop_helper(/, operator/)

// Cross-type ops: float8_t <-> float16_t
#define fp8_fp16_binop_helper(__op__, __operator__)                    \
  inline float __operator__(float8_t lhs, float16_t rhs) {            \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs);    \
  }                                                                    \
  inline float __operator__(float16_t lhs, float8_t rhs) {            \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs);    \
  }

fp8_fp16_binop_helper(+, operator+)
fp8_fp16_binop_helper(-, operator-)
fp8_fp16_binop_helper(*, operator*)
fp8_fp16_binop_helper(/, operator/)

// Cross-type ops: float8_t <-> bfloat16_t
#define fp8_bf16_binop_helper(__op__, __operator__)                    \
  inline float __operator__(float8_t lhs, bfloat16_t rhs) {           \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs);    \
  }                                                                    \
  inline float __operator__(bfloat16_t lhs, float8_t rhs) {           \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs);    \
  }

fp8_bf16_binop_helper(+, operator+)
fp8_bf16_binop_helper(-, operator-)
fp8_bf16_binop_helper(*, operator*)
fp8_bf16_binop_helper(/, operator/)

// Cross-type ops: bfloat8_t <-> float16_t
#define bf8_fp16_binop_helper(__op__, __operator__)                    \
  inline float __operator__(bfloat8_t lhs, float16_t rhs) {           \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs);    \
  }                                                                    \
  inline float __operator__(float16_t lhs, bfloat8_t rhs) {           \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs);    \
  }

bf8_fp16_binop_helper(+, operator+)
bf8_fp16_binop_helper(-, operator-)
bf8_fp16_binop_helper(*, operator*)
bf8_fp16_binop_helper(/, operator/)

// Cross-type ops: bfloat8_t <-> bfloat16_t
#define bf8_bf16_binop_helper(__op__, __operator__)                    \
  inline float __operator__(bfloat8_t lhs, bfloat16_t rhs) {          \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs);    \
  }                                                                    \
  inline float __operator__(bfloat16_t lhs, bfloat8_t rhs) {          \
    return static_cast<float>(lhs) __op__ static_cast<float>(rhs);    \
  }

bf8_bf16_binop_helper(+, operator+)
bf8_bf16_binop_helper(-, operator-)
bf8_bf16_binop_helper(*, operator*)
bf8_bf16_binop_helper(/, operator/)
// clang-format on

} // namespace mlx::core
