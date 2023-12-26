// Copyright © 2023 Apple Inc.

#include <metal_integer>
#include <metal_math>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/bf16.h"

struct Add {
  template <typename T> T operator()(T x, T y) { return x + y; }
};

struct Divide {
  template <typename T> T operator()(T x, T y) { return x / y; }
};

struct Remainder {
  template <typename T> T operator()(T x, T y) { return x % y; }
  template <> float operator()(float x, float y) { return fmod(x, y); }
  template <> half operator()(half x, half y) { return fmod(x, y); }
  template <> bfloat16_t operator()(bfloat16_t x, bfloat16_t y) { return fmod(x, y); }
};

struct Equal {
  template <typename T> bool operator()(T x, T y) { return x == y; }
};

struct NaNEqual {
  template <typename T> bool operator()(T x, T y) {
    return x == y || (metal::isnan(x) && metal::isnan(y));
  }
  template <>
  bool operator()(complex64_t x, complex64_t y) {
    return x == y ||
      (metal::isnan(x.real) && metal::isnan(y.real)
       && metal::isnan(x.imag) && metal::isnan(y.imag)) ||
      (x.real == y.real && metal::isnan(x.imag) && metal::isnan(y.imag)) ||
      (metal::isnan(x.real) && metal::isnan(y.real) && x.imag == y.imag);
  }
};

struct Greater {
  template <typename T> bool operator()(T x, T y) { return x > y; }
};

struct GreaterEqual {
  template <typename T> bool operator()(T x, T y) { return x >= y; }
};

struct Less {
  template <typename T> bool operator()(T x, T y) { return x < y; }
};

struct LessEqual {
  template <typename T> bool operator()(T x, T y) { return x <= y; }
};

struct LogAddExp {
  template <typename T>
  T operator()(T x, T y) {
    constexpr T inf = metal::numeric_limits<T>::infinity();
    T maxval = metal::max(x, y);
    T minval = metal::min(x, y);
    return (minval == -inf || maxval == inf) ? maxval :
      (maxval + log1p(metal::exp(minval - maxval)));
  };
};

struct Maximum {
  template <typename T> T operator()(T x, T y) { return metal::max(x, y); }

  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    return x >= y ? x : y;
  }
};

struct Minimum {
  template <typename T> T operator()(T x, T y) { return metal::min(x, y); }

  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    return x <= y ? x : y;
  }
};

struct Multiply {
  template <typename T> T operator()(T x, T y) { return x * y; }
};

struct NotEqual {
  template <typename T> bool operator()(T x, T y) { return x != y; }
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
  template <typename T> T operator()(T x, T y) { return x - y; }
};

template <typename T, typename U, typename Op>
[[kernel]] void binary_op_s2s(
    device const T* a,
    device const T* b,
    device U* c,
    uint index [[thread_position_in_grid]]) {
  c[index] = Op()(a[0], b[0]);
}


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
  size_t out_idx = index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
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
  size_t out_idx = index.x + (size_t)grid_dim.x * (index.y + (size_t)grid_dim.y * index.z);
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

#define instantiate_binary(name, itype, otype, op, bopt) \
  template [[host_name(name)]] \
  [[kernel]] void binary_op_##bopt<itype, otype, op>( \
      device const itype* a, \
      device const itype* b, \
      device otype* c, \
      uint index [[thread_position_in_grid]]);

#define instantiate_binary_g_dim(name, itype, otype, op, dims) \
  template [[host_name(name "_" #dims)]] \
  [[kernel]] void binary_op_g_nd<itype, otype, op, dims>( \
      device const itype* a, \
      device const itype* b, \
      device otype* c, \
      constant const int shape[dims], \
      constant const size_t a_strides[dims], \
      constant const size_t b_strides[dims], \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]);

#define instantiate_binary_g_nd(name, itype, otype, op) \
  template [[host_name(name "_1")]] \
  [[kernel]] void binary_op_g_nd1<itype, otype, op>( \
      device const itype* a, \
      device const itype* b, \
      device otype* c, \
      constant const size_t& a_stride, \
      constant const size_t& b_stride, \
      uint index [[thread_position_in_grid]]); \
  template [[host_name(name "_2")]] \
  [[kernel]] void binary_op_g_nd2<itype, otype, op>( \
      device const itype* a, \
      device const itype* b, \
      device otype* c, \
      constant const size_t a_strides[2], \
      constant const size_t b_strides[2], \
      uint2 index [[thread_position_in_grid]], \
      uint2 grid_dim [[threads_per_grid]]); \
  template [[host_name(name "_3")]] \
  [[kernel]] void binary_op_g_nd3<itype, otype, op>( \
      device const itype* a, \
      device const itype* b, \
      device otype* c, \
      constant const size_t a_strides[3], \
      constant const size_t b_strides[3], \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]); \
  instantiate_binary_g_dim(name, itype, otype, op, 4) \
  instantiate_binary_g_dim(name, itype, otype, op, 5)


#define instantiate_binary_g(name, itype, otype, op) \
  template [[host_name(name)]] \
  [[kernel]] void binary_op_g<itype, otype, op>( \
      device const itype* a, \
      device const itype* b, \
      device otype* c, \
      constant const int* shape, \
      constant const size_t* a_strides, \
      constant const size_t* b_strides, \
      constant const int& ndim, \
      uint3 index [[thread_position_in_grid]], \
      uint3 grid_dim [[threads_per_grid]]);

#define instantiate_binary_all(name, tname, itype, otype, op) \
  instantiate_binary("ss" #name #tname, itype, otype, op, ss) \
  instantiate_binary("sv" #name #tname, itype, otype, op, sv) \
  instantiate_binary("vs" #name #tname, itype, otype, op, vs) \
  instantiate_binary("vv" #name #tname, itype, otype, op, vv) \
  instantiate_binary_g("g" #name #tname, itype, otype, op) \
  instantiate_binary_g_nd("g" #name #tname, itype, otype, op)

#define instantiate_binary_float(name, op) \
  instantiate_binary_all(name, float16, half, half, op) \
  instantiate_binary_all(name, float32, float, float, op) \
  instantiate_binary_all(name, bfloat16, bfloat16_t, bfloat16_t, op)

#define instantiate_binary_types(name, op) \
  instantiate_binary_all(name, bool_, bool, bool, op) \
  instantiate_binary_all(name, uint8, uint8_t, uint8_t, op) \
  instantiate_binary_all(name, uint16, uint16_t, uint16_t, op) \
  instantiate_binary_all(name, uint32, uint32_t, uint32_t, op) \
  instantiate_binary_all(name, uint64, uint64_t, uint64_t, op) \
  instantiate_binary_all(name, int8, int8_t, int8_t, op) \
  instantiate_binary_all(name, int16, int16_t, int16_t, op) \
  instantiate_binary_all(name, int32, int32_t, int32_t, op) \
  instantiate_binary_all(name, int64, int64_t, int64_t, op) \
  instantiate_binary_all(name, complex64, complex64_t, complex64_t, op) \
  instantiate_binary_float(name, op)

#define instantiate_binary_types_bool(name, op) \
  instantiate_binary_all(name, bool_, bool, bool, op) \
  instantiate_binary_all(name, uint8, uint8_t, bool, op) \
  instantiate_binary_all(name, uint16, uint16_t, bool, op) \
  instantiate_binary_all(name, uint32, uint32_t, bool, op) \
  instantiate_binary_all(name, uint64, uint64_t, bool, op) \
  instantiate_binary_all(name, int8, int8_t, bool, op) \
  instantiate_binary_all(name, int16, int16_t, bool, op) \
  instantiate_binary_all(name, int32, int32_t, bool, op) \
  instantiate_binary_all(name, int64, int64_t, bool, op) \
  instantiate_binary_all(name, float16, half, bool, op) \
  instantiate_binary_all(name, float32, float, bool, op) \
  instantiate_binary_all(name, bfloat16, bfloat16_t, bool, op) \
  instantiate_binary_all(name, complex64, complex64_t, bool, op)

instantiate_binary_types(add, Add)
instantiate_binary_types(div, Divide)
instantiate_binary_types_bool(eq, Equal)
instantiate_binary_types_bool(ge, Greater)
instantiate_binary_types_bool(geq, GreaterEqual)
instantiate_binary_types_bool(le, Less)
instantiate_binary_types_bool(leq, LessEqual)
instantiate_binary_types_bool(neq, NotEqual)
instantiate_binary_float(lae, LogAddExp)
instantiate_binary_types(max, Maximum)
instantiate_binary_types(min, Minimum)
instantiate_binary_types(mul, Multiply)
instantiate_binary_types(sub, Subtract)
instantiate_binary_types(pow, Power)
instantiate_binary_types(rem, Remainder)

// NaNEqual only needed for floating point types with boolean output
instantiate_binary_all(naneq, float16, half, bool, NaNEqual)
instantiate_binary_all(naneq, float32, float, bool, NaNEqual)
instantiate_binary_all(naneq, bfloat16, bfloat16_t, bool, NaNEqual)
instantiate_binary_all(naneq, complex64, complex64_t, bool, NaNEqual)
