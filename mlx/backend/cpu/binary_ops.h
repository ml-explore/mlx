// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/backend/cpu/simd/simd.h"

namespace mlx::core::detail {

using namespace mlx::core::simd;

#define BINARY_SINGLE()                                 \
  template <typename T>                                 \
  T operator()(T x, T y) {                              \
    return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; \
  }

#define DEFAULT_BINARY_OP(Op, op)                       \
  struct Op {                                           \
    template <int N, typename T>                        \
    Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { \
      return op(x, y);                                  \
    }                                                   \
    BINARY_SINGLE()                                     \
  };

DEFAULT_BINARY_OP(Add, operator+)
DEFAULT_BINARY_OP(ArcTan2, atan2)
DEFAULT_BINARY_OP(Divide, operator/)
DEFAULT_BINARY_OP(Multiply, operator*)
DEFAULT_BINARY_OP(Subtract, operator-)
DEFAULT_BINARY_OP(LogicalAnd, operator&&)
DEFAULT_BINARY_OP(LogicalOr, operator||)
DEFAULT_BINARY_OP(BitwiseAnd, operator&)
DEFAULT_BINARY_OP(BitwiseOr, operator|)
DEFAULT_BINARY_OP(BitwiseXor, operator^)
DEFAULT_BINARY_OP(LeftShift, operator<<)
DEFAULT_BINARY_OP(RightShift, operator>>)
DEFAULT_BINARY_OP(Remainder, remainder)
DEFAULT_BINARY_OP(Maximum, maximum)
DEFAULT_BINARY_OP(Minimum, minimum)
DEFAULT_BINARY_OP(Power, pow)

#define DEFAULT_BOOL_OP(Op, op)                            \
  struct Op {                                              \
    template <int N, typename T>                           \
    Simd<bool, N> operator()(Simd<T, N> x, Simd<T, N> y) { \
      return op(x, y);                                     \
    }                                                      \
    template <typename T>                                  \
    bool operator()(T x, T y) {                            \
      return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value;  \
    }                                                      \
  };

DEFAULT_BOOL_OP(Equal, operator==)
DEFAULT_BOOL_OP(Greater, operator>)
DEFAULT_BOOL_OP(GreaterEqual, operator>=)
DEFAULT_BOOL_OP(Less, operator<)
DEFAULT_BOOL_OP(LessEqual, operator<=)
DEFAULT_BOOL_OP(NotEqual, operator!=)

struct NaNEqual {
  template <int N, typename T>
  Simd<bool, N> operator()(Simd<T, N> x, Simd<T, N> y) {
    return x == y || (isnan(x) && isnan(y));
  }
  template <typename T>
  bool operator()(T x, T y) {
    return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value;
  }
};

struct LogAddExp {
  template <int N, typename T>
  Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) {
    auto maxval = maximum(x, y);
    auto minval = minimum(x, y);
    auto mask = minval == -inf || maxval == inf;
    auto out = maxval + log1p(exp(minval - maxval));
    return select(mask, Simd<T, N>(maxval), Simd<T, N>(out));
  }
  BINARY_SINGLE()
};

struct Select {
  template <typename T>
  T operator()(bool condition, T x, T y) {
    return (*this)(Simd<bool, 1>(condition), Simd<T, 1>(x), Simd<T, 1>(y))
        .value;
  }

  template <int N, typename T>
  Simd<T, N> operator()(Simd<bool, N> condition, Simd<T, N> x, Simd<T, N> y) {
    return select(condition, x, y);
  }
};

} // namespace mlx::core::detail
