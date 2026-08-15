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
// A shift amount that is negative or at least the operand width is undefined
// in C++, so the scalar and vectorized paths disagreed about it and the answer
// depended on the array length. Clamp the amount instead, which gives what
// numpy and pytorch return: zero for a left shift, and the sign bit for a
// right shift.
template <int N, typename T>
Simd<bool, N> shift_in_range(Simd<T, N> y) {
  auto in_range = y < Simd<T, N>(sizeof(T) * 8);
  if constexpr (std::is_signed_v<T>) {
    in_range = in_range && y >= Simd<T, N>(0);
  }
  return in_range;
}

struct LeftShift {
  template <int N, typename T>
  Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) {
    Simd<T, N> amount = y & Simd<T, N>(sizeof(T) * 8 - 1);
    Simd<T, N> shifted = x << amount;
    return select(shift_in_range(y), shifted, Simd<T, N>(0));
  }
  BINARY_SINGLE()
};

struct RightShift {
  template <int N, typename T>
  Simd<T, N> operator()(Simd<T, N> x, Simd<T, N> y) {
    auto in_range = shift_in_range(y);
    Simd<T, N> amount = select(in_range, y, Simd<T, N>(sizeof(T) * 8 - 1));
    Simd<T, N> shifted = x >> amount;
    if constexpr (std::is_signed_v<T>) {
      // Shifting a negative value past the width leaves the sign bit behind.
      return shifted;
    } else {
      return select(in_range, shifted, Simd<T, N>(0));
    }
  }
  BINARY_SINGLE()
};

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
