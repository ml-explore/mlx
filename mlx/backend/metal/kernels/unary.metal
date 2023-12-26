// Copyright © 2023 Apple Inc.

#include <metal_integer>
#include <metal_math>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/erf.h"
#include "mlx/backend/metal/kernels/bf16.h"

struct Abs {
  template <typename T> T operator()(T x) { return metal::abs(x); };
  template <> uint8_t operator()(uint8_t x) { return x; };
  template <> uint16_t operator()(uint16_t x) { return x; };
  template <> uint32_t operator()(uint32_t x) { return x; };
  template <> uint64_t operator()(uint64_t x) { return x; };
  template <> bool operator()(bool x) { return x; };
  template <> complex64_t operator()(complex64_t x) {
      return {metal::precise::sqrt(x.real * x.real + x.imag * x.imag), 0};
  };
};

struct ArcCos {
  template <typename T> T operator()(T x) { return metal::precise::acos(x); };
};

struct ArcCosh {
  template <typename T> T operator()(T x) { return metal::precise::acosh(x); };
};

struct ArcSin {
  template <typename T> T operator()(T x) { return metal::precise::asin(x); };
};

struct ArcSinh {
  template <typename T> T operator()(T x) { return metal::precise::asinh(x); };
};

struct ArcTan {
  template <typename T> T operator()(T x) { return metal::precise::atan(x); };
};

struct ArcTanh {
  template <typename T> T operator()(T x) { return metal::precise::atanh(x); };
};

struct Ceil {
  template <typename T> T operator()(T x) { return metal::ceil(x); };
  template <> int8_t operator()(int8_t x) { return x; };
  template <> int16_t operator()(int16_t x) { return x; };
  template <> int32_t operator()(int32_t x) { return x; };
  template <> int64_t operator()(int64_t x) { return x; };
  template <> uint8_t operator()(uint8_t x) { return x; };
  template <> uint16_t operator()(uint16_t x) { return x; };
  template <> uint32_t operator()(uint32_t x) { return x; };
  template <> uint64_t operator()(uint64_t x) { return x; };
  template <> bool operator()(bool x) { return x; };
};

struct Cos {
  template <typename T> T operator()(T x) { return metal::precise::cos(x); };

  template <>
  complex64_t operator()(complex64_t x) {
    return {
      metal::precise::cos(x.real) * metal::precise::cosh(x.imag),
      -metal::precise::sin(x.real) * metal::precise::sinh(x.imag)
    };
  };
};

struct Cosh {
  template <typename T> T operator()(T x) { return metal::precise::cosh(x); };

  template <>
  complex64_t operator()(complex64_t x) {
    return {
      metal::precise::cosh(x.real) * metal::precise::cos(x.imag),
      metal::precise::sinh(x.real) * metal::precise::sin(x.imag)
    };
  };
};

struct Erf {
  template <typename T> T operator()(T x) { return static_cast<T>(erf(static_cast<float>(x))); };
};

struct ErfInv {
  template <typename T> T operator()(T x) { return static_cast<T>(erfinv(static_cast<float>(x))); };
};

struct Exp {
  template <typename T> T operator()(T x) { return metal::precise::exp(x); };
  template <> complex64_t operator()(complex64_t x) {
    auto m = metal::precise::exp(x.real);
    return {m * metal::precise::cos(x.imag), m * metal::precise::sin(x.imag)};
  }
};

struct Floor {
  template <typename T> T operator()(T x) { return metal::floor(x); };
  template <> int8_t operator()(int8_t x) { return x; };
  template <> int16_t operator()(int16_t x) { return x; };
  template <> int32_t operator()(int32_t x) { return x; };
  template <> int64_t operator()(int64_t x) { return x; };
  template <> uint8_t operator()(uint8_t x) { return x; };
  template <> uint16_t operator()(uint16_t x) { return x; };
  template <> uint32_t operator()(uint32_t x) { return x; };
  template <> uint64_t operator()(uint64_t x) { return x; };
  template <> bool operator()(bool x) { return x; };
};

struct Log {
  template <typename T> T operator()(T x) { return metal::precise::log(x); };
};

struct Log2 {
  template <typename T> T operator()(T x) { return metal::precise::log2(x); };
};

struct Log10 {
  template <typename T> T operator()(T x) { return metal::precise::log10(x); };
};

struct Log1p {
  template <typename T> T operator()(T x) { return log1p(x); };
};

struct LogicalNot {
  template <typename T> T operator()(T x) { return !x; };
};

struct Negative {
  template <typename T> T operator()(T x) { return -x; };
};

struct Round {
  template <typename T> T operator()(T x) { return metal::round(x); };
  template <> complex64_t operator()(complex64_t x) { return {metal::round(x.real), metal::round(x.imag)}; };
};

struct Sigmoid {
  template <typename T>
  T operator()(T x) {
    auto y = 1 / (1 + metal::exp(-metal::abs(x)));
    return (x < 0) ? 1 - y : y;
  }
};

struct Sign {
  template <typename T> T operator()(T x) { return (x > T(0)) - (x < T(0)); };
  template <> uint32_t operator()(uint32_t x) { return x != 0; };
};

struct Sin {
  template <typename T> T operator()(T x) { return metal::precise::sin(x); };

  template <>
  complex64_t operator()(complex64_t x) {
    return {
      metal::precise::sin(x.real) * metal::precise::cosh(x.imag),
      metal::precise::cos(x.real) * metal::precise::sinh(x.imag)
    };
  };
};

struct Sinh {
  template <typename T> T operator()(T x) { return metal::precise::sinh(x); };

  template <>
  complex64_t operator()(complex64_t x) {
    return {
      metal::precise::sinh(x.real) * metal::precise::cos(x.imag),
      metal::precise::cosh(x.real) * metal::precise::sin(x.imag)
    };
  };
};

struct Square {
  template <typename T> T operator()(T x) { return x * x; };
};

struct Sqrt {
  template <typename T> T operator()(T x) { return metal::precise::sqrt(x); };
};

struct Rsqrt {
  template <typename T> T operator()(T x) { return metal::precise::rsqrt(x); };
};

struct Tan {
  template <typename T> T operator()(T x) { return metal::precise::tan(x); };

  template <>
  complex64_t operator()(complex64_t x) {
    float tan_a = metal::precise::tan(x.real);
    float tanh_b = metal::precise::tanh(x.imag);
    float t1 = tan_a * tanh_b;
    float denom = 1. + t1 * t1;
    return {
      (tan_a - tanh_b * t1) / denom,
      (tanh_b + tan_a * t1) / denom
    };
  };
};

struct Tanh {
  template <typename T> T operator()(T x) { return metal::precise::tanh(x); };

  template <>
  complex64_t operator()(complex64_t x) {
    float tanh_a = metal::precise::tanh(x.real);
    float tan_b = metal::precise::tan(x.imag);
    float t1 = tanh_a * tan_b;
    float denom = 1. + t1 * t1;
    return {
      (tanh_a + tan_b * t1) / denom,
      (tan_b - tanh_a * t1) / denom
    };
  };
};

template <typename T, typename Op>
[[kernel]] void unary_op_v(
    device const T* in,
    device T* out,
    uint index [[thread_position_in_grid]]) {
  out[index] = Op()(in[index]);
}

template <typename T, typename Op>
[[kernel]] void unary_op_g(
    device const T* in,
    device T* out,
    device const int* in_shape,
    device const size_t* in_strides,
    device const int& ndim,
    uint index [[thread_position_in_grid]]) {
  auto idx = elem_to_loc(index, in_shape, in_strides, ndim);
  out[index] = Op()(in[idx]);
}

#define instantiate_unary_v(name, type, op) \
  template [[host_name(name)]] \
  [[kernel]] void unary_op_v<type, op>( \
      device const type* in, \
      device type* out, \
      uint index [[thread_position_in_grid]]);

#define instantiate_unary_g(name, type, op) \
  template [[host_name(name)]] \
  [[kernel]] void unary_op_g<type, op>( \
      device const type* in, \
      device type* out, \
      device const int* in_shape, \
      device const size_t* in_strides, \
      device const int& ndim, \
      uint index [[thread_position_in_grid]]);

#define instantiate_unary_all(name, tname, type, op) \
  instantiate_unary_v("v" #name #tname, type, op) \
  instantiate_unary_g("g" #name #tname, type, op)

#define instantiate_unary_float(name, op) \
  instantiate_unary_all(name, float16, half, op) \
  instantiate_unary_all(name, float32, float, op) \
  instantiate_unary_all(name, bfloat16, bfloat16_t, op) \

#define instantiate_unary_types(name, op) \
  instantiate_unary_all(name, bool_, bool, op) \
  instantiate_unary_all(name, uint8, uint8_t, op) \
  instantiate_unary_all(name, uint16, uint16_t, op) \
  instantiate_unary_all(name, uint32, uint32_t, op) \
  instantiate_unary_all(name, uint64, uint64_t, op) \
  instantiate_unary_all(name, int8, int8_t, op) \
  instantiate_unary_all(name, int16, int16_t, op) \
  instantiate_unary_all(name, int32, int32_t, op) \
  instantiate_unary_all(name, int64, int64_t, op) \
  instantiate_unary_float(name, op)

instantiate_unary_types(abs, Abs)
instantiate_unary_float(arccos, ArcCos)
instantiate_unary_float(arccosh, ArcCosh)
instantiate_unary_float(arcsin, ArcSin)
instantiate_unary_float(arcsinh, ArcSinh)
instantiate_unary_float(arctan, ArcTan)
instantiate_unary_float(arctanh, ArcTanh)
instantiate_unary_types(ceil, Ceil)
instantiate_unary_float(cos, Cos)
instantiate_unary_float(cosh, Cosh)
instantiate_unary_float(exp, Exp)
instantiate_unary_types(floor, Floor)
instantiate_unary_float(log, Log)
instantiate_unary_float(log2, Log2)
instantiate_unary_float(log10, Log10)
instantiate_unary_float(log1p, Log1p)
instantiate_unary_types(neg, Negative)
instantiate_unary_float(sigmoid, Sigmoid)
instantiate_unary_float(erf, Erf)
instantiate_unary_float(erfinv, ErfInv)
instantiate_unary_types(sign, Sign)
instantiate_unary_float(sin, Sin)
instantiate_unary_float(sinh, Sinh)
instantiate_unary_types(square, Square)
instantiate_unary_float(sqrt, Sqrt)
instantiate_unary_float(rsqrt, Rsqrt)
instantiate_unary_float(tan, Tan)
instantiate_unary_float(tanh, Tanh)
instantiate_unary_float(round, Round)

instantiate_unary_all(abs, complex64, complex64_t, Abs)
instantiate_unary_all(cos, complex64, complex64_t, Cos)
instantiate_unary_all(cosh, complex64, complex64_t, Cosh)
instantiate_unary_all(exp, complex64, complex64_t, Exp)
instantiate_unary_all(neg, complex64, complex64_t, Negative)
instantiate_unary_all(sin, complex64, complex64_t, Sin)
instantiate_unary_all(sinh, complex64, complex64_t, Sinh)
instantiate_unary_all(tan, complex64, complex64_t, Tan)
instantiate_unary_all(tanh, complex64, complex64_t, Tanh)
instantiate_unary_all(round, complex64, complex64_t, Round)

instantiate_unary_all(lnot, bool_, bool, LogicalNot)
