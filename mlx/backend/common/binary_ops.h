// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/backend/common/simd/simd.h"

namespace mlx::core::detail {

using namespace mlx::core::simd;

#define BINARY_SINGLE()                                 \
  template <typename T>                                 \
  T operator()(T x, T y) {                              \
    return (*this)(Simd<T, 1>(x), Simd<T, 1>(y)).value; \
  }

#define DEFAULT_OP(Op, op)                              \
  struct Op {                                           \
    template <int N, typename T>                        \
    Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) { \
      return op(x, y);                                  \
    }                                                   \
    BINARY_SINGLE()                                     \
  };

DEFAULT_OP(Add, operator+)
DEFAULT_OP(ArcTan2, atan2)
DEFAULT_OP(Divide, operator/)
DEFAULT_OP(Multiply, operator*)
DEFAULT_OP(Subtract, operator-)
DEFAULT_OP(LogicalAnd, operator&&)
DEFAULT_OP(LogicalOr, operator||)
DEFAULT_OP(BitwiseAnd, operator&)
DEFAULT_OP(BitwiseOr, operator|)
DEFAULT_OP(BitwiseXor, operator^)
DEFAULT_OP(LeftShift, operator<<)
DEFAULT_OP(RightShift, operator>>)
DEFAULT_OP(Remainder, remainder)
DEFAULT_OP(Maximum, maximum)
DEFAULT_OP(Minimum, minimum)
DEFAULT_OP(Power, pow)

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
  template <typename T>
  T operator()(T x, T y) {
    constexpr float inf = std::numeric_limits<float>::infinity();
    auto maxval = Maximum()(x, y);
    auto minval = Minimum()(x, y);
    return (minval == -inf || maxval == inf)
        ? maxval
        : static_cast<T>(maxval + std::log1p(fast_exp(minval - maxval)));
  }

  template <int N, typename T>
  Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) {
    auto maxval = maximum(x, y);
    auto minval = minimum(x, y);
    auto mask = minval == -inf || maxval == inf;
    auto out = maxval + log1p(exp(minval - maxval));
    return select(mask, maxval, out);
  }
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
