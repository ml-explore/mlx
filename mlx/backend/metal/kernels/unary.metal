// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/metal/kernels/unary.h"

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
