// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_integer>
#include <metal_math>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/utils.h"

struct Add {
  template <typename T>
  T operator()(T x, T y) {
    return x + y;
  }
};

struct FloorDivide {
  template <typename T>
  T operator()(T x, T y) {
    return x / y;
  }
  template <>
  float operator()(float x, float y) {
    return trunc(x / y);
  }
  template <>
  half operator()(half x, half y) {
    return trunc(x / y);
  }
  template <>
  bfloat16_t operator()(bfloat16_t x, bfloat16_t y) {
    return trunc(x / y);
  }
};

struct Divide {
  template <typename T>
  T operator()(T x, T y) {
    return x / y;
  }
};

struct Remainder {
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T> & !metal::is_signed_v<T>, T>
  operator()(T x, T y) {
    return x % y;
  }
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T> & metal::is_signed_v<T>, T>
  operator()(T x, T y) {
    auto r = x % y;
    if (r != 0 && (r < 0 != y < 0)) {
      r += y;
    }
    return r;
  }
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    T r = fmod(x, y);
    if (r != 0 && (r < 0 != y < 0)) {
      r += y;
    }
    return r;
  }
  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    return x % y;
  }
};

struct Equal {
  template <typename T>
  bool operator()(T x, T y) {
    return x == y;
  }
};

struct NaNEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x == y || (metal::isnan(x) && metal::isnan(y));
  }
  template <>
  bool operator()(complex64_t x, complex64_t y) {
    return x == y ||
        (metal::isnan(x.real) && metal::isnan(y.real) && metal::isnan(x.imag) &&
         metal::isnan(y.imag)) ||
        (x.real == y.real && metal::isnan(x.imag) && metal::isnan(y.imag)) ||
        (metal::isnan(x.real) && metal::isnan(y.real) && x.imag == y.imag);
  }
};

struct Greater {
  template <typename T>
  bool operator()(T x, T y) {
    return x > y;
  }
};

struct GreaterEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x >= y;
  }
};

struct Less {
  template <typename T>
  bool operator()(T x, T y) {
    return x < y;
  }
};

struct LessEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x <= y;
  }
};

struct LogAddExp {
  template <typename T>
  T operator()(T x, T y) {
    if (metal::isnan(x) || metal::isnan(y)) {
      return metal::numeric_limits<T>::quiet_NaN();
    }
    constexpr T inf = metal::numeric_limits<T>::infinity();
    T maxval = metal::max(x, y);
    T minval = metal::min(x, y);
    return (minval == -inf || maxval == inf)
        ? maxval
        : (maxval + log1p(metal::exp(minval - maxval)));
  };
};

struct Maximum {
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T x, T y) {
    return metal::max(x, y);
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    if (metal::isnan(x)) {
      return x;
    }
    return x > y ? x : y;
  }

  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    if (metal::isnan(x.real) || metal::isnan(x.imag)) {
      return x;
    }
    return x > y ? x : y;
  }
};

struct Minimum {
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T x, T y) {
    return metal::min(x, y);
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    if (metal::isnan(x)) {
      return x;
    }
    return x < y ? x : y;
  }

  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    if (metal::isnan(x.real) || metal::isnan(x.imag)) {
      return x;
    }
    return x < y ? x : y;
  }
};

struct Multiply {
  template <typename T>
  T operator()(T x, T y) {
    return x * y;
  }
};

struct NotEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x != y;
  }
  template <>
  bool operator()(complex64_t x, complex64_t y) {
    return x.real != y.real || x.imag != y.imag;
  }
};

struct Power {
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T base, T exp) {
    return metal::pow(base, exp);
  }

  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T base, T exp) {
    T res = 1;
    while (exp) {
      if (exp & 1) {
        res *= base;
      }
      exp >>= 1;
      base *= base;
    }
    return res;
  }

  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    auto x_theta = metal::atan(x.imag / x.real);
    auto x_ln_r = 0.5 * metal::log(x.real * x.real + x.imag * x.imag);
    auto mag = metal::exp(y.real * x_ln_r - y.imag * x_theta);
    auto phase = y.imag * x_ln_r + y.real * x_theta;
    return {mag * metal::cos(phase), mag * metal::sin(phase)};
  }
};

struct Subtract {
  template <typename T>
  T operator()(T x, T y) {
    return x - y;
  }
};

struct LogicalAnd {
  template <typename T>
  T operator()(T x, T y) {
    return x && y;
  };
};

struct LogicalOr {
  template <typename T>
  T operator()(T x, T y) {
    return x || y;
  };
};

struct BitwiseAnd {
  template <typename T>
  T operator()(T x, T y) {
    return x & y;
  };
};

struct BitwiseOr {
  template <typename T>
  T operator()(T x, T y) {
    return x | y;
  };
};

struct BitwiseXor {
  template <typename T>
  T operator()(T x, T y) {
    return x ^ y;
  };
};

struct LeftShift {
  template <typename T>
  T operator()(T x, T y) {
    return x << y;
  };
};

struct RightShift {
  template <typename T>
  T operator()(T x, T y) {
    return x >> y;
  };
};

struct ArcTan2 {
  template <typename T>
  T operator()(T y, T x) {
    return metal::precise::atan2(y, x);
  }
};

struct DivMod {
  template <typename T>
  metal::array<T, 2> operator()(T x, T y) {
    return {FloorDivide{}(x, y), Remainder{}(x, y)};
  };
};

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_ss(
    device const T* a,
    device const T* b,
    device U* c,
    uint index [[thread_position_in_grid]]) {
  c[index] = Op()(a[0], b[0]);
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_sv(
    device const T* a,
    device const T* b,
    device U* c,
    uint index [[thread_position_in_grid]]) {
  c[index] = Op()(a[0], b[index]);
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_vs(
    device const T* a,
    device const T* b,
    device U* c,
    uint index [[thread_position_in_grid]]) {
  c[index] = Op()(a[index], b[0]);
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_vv(
    device const T* a,
    device const T* b,
    device U* c,
    uint index [[thread_position_in_grid]]) {
  c[index] = Op()(a[index], b[index]);
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_g_nd1(
    device const T* a,
    device const T* b,
    device U* c,
    constant const size_t& a_stride,
    constant const size_t& b_stride,
    uint index [[thread_position_in_grid]]) {
  auto a_idx = elem_to_loc_1(index, a_stride);
  auto b_idx = elem_to_loc_1(index, b_stride);
  c[index] = Op()(a[a_idx], b[b_idx]);
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_g_nd2(
    device const T* a,
    device const T* b,
    device U* c,
    constant const size_t a_strides[2],
    constant const size_t b_strides[2],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_2(index, a_strides);
  auto b_idx = elem_to_loc_2(index, b_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * index.y;
  c[out_idx] = Op()(a[a_idx], b[b_idx]);
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_g_nd3(
    device const T* a,
    device const T* b,
    device U* c,
    constant const size_t a_strides[3],
    constant const size_t b_strides[3],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_3(index, a_strides);
  auto b_idx = elem_to_loc_3(index, b_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  c[out_idx] = Op()(a[a_idx], b[b_idx]);
}

template <typename T, typename U, typename Op, int DIM>
[[kernel]] void binary_op_g_nd(
    device const T* a,
    device const T* b,
    device U* c,
    constant const int shape[DIM],
    constant const size_t a_strides[DIM],
    constant const size_t b_strides[DIM],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_2_nd<DIM>(index, shape, a_strides, b_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  c[out_idx] = Op()(a[idx.x], b[idx.y]);
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_g(
    device const T* a,
    device const T* b,
    device U* c,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_2_nd(index, shape, a_strides, b_strides, ndim);
  size_t out_idx = index.x + grid_dim.x * (index.y + grid_dim.y * index.z);
  c[out_idx] = Op()(a[idx.x], b[idx.y]);
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_s2s(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    uint index [[thread_position_in_grid]]) {
  auto out = Op()(a[0], b[0]);
  c[index] = out[0];
  d[index] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_ss(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    uint index [[thread_position_in_grid]]) {
  auto out = Op()(a[0], b[0]);
  c[index] = out[0];
  d[index] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_sv(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    uint index [[thread_position_in_grid]]) {
  auto out = Op()(a[0], b[index]);
  c[index] = out[0];
  d[index] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_vs(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    uint index [[thread_position_in_grid]]) {
  auto out = Op()(a[index], b[0]);
  c[index] = out[0];
  d[index] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_vv(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    uint index [[thread_position_in_grid]]) {
  auto out = Op()(a[index], b[index]);
  c[index] = out[0];
  d[index] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_g_nd1(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const size_t& a_stride,
    constant const size_t& b_stride,
    uint index [[thread_position_in_grid]]) {
  auto a_idx = elem_to_loc_1(index, a_stride);
  auto b_idx = elem_to_loc_1(index, b_stride);
  auto out = Op()(a[a_idx], b[b_idx]);
  c[index] = out[0];
  d[index] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_g_nd2(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const size_t a_strides[2],
    constant const size_t b_strides[2],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_2(index, a_strides);
  auto b_idx = elem_to_loc_2(index, b_strides);
  size_t out_idx = index.x + (size_t)grid_dim.x * index.y;
  auto out = Op()(a[a_idx], b[b_idx]);
  c[out_idx] = out[0];
  d[out_idx] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_g_nd3(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const size_t a_strides[3],
    constant const size_t b_strides[3],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto a_idx = elem_to_loc_3(index, a_strides);
  auto b_idx = elem_to_loc_3(index, b_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  auto out = Op()(a[a_idx], b[b_idx]);
  c[out_idx] = out[0];
  d[out_idx] = out[1];
}

template <typename T, typename U, typename Op, int DIM>
[[kernel]] void binary_op_g_nd(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const int shape[DIM],
    constant const size_t a_strides[DIM],
    constant const size_t b_strides[DIM],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_2_nd<DIM>(index, shape, a_strides, b_strides);
  size_t out_idx =
      index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
  auto out = Op()(a[idx.x], b[idx.y]);
  c[out_idx] = out[0];
  d[out_idx] = out[1];
}

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_g(
    device const T* a,
    device const T* b,
    device U* c,
    device U* d,
    constant const int* shape,
    constant const size_t* a_strides,
    constant const size_t* b_strides,
    constant const int& ndim,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  auto idx = elem_to_loc_2_nd(index, shape, a_strides, b_strides, ndim);
  size_t out_idx = index.x + grid_dim.x * (index.y + grid_dim.y * index.z);
  auto out = Op()(a[idx.x], b[idx.y]);
  c[out_idx] = out[0];
  d[out_idx] = out[1];
}
>>>>>>> 31caf139 (try cpp 20 for compile)
