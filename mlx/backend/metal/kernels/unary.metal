// Copyright © 2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/unary_ops.h"
#include "mlx/backend/metal/kernels/unary.h"

#define instantiate_unary_all(op, tname, type)                  \
  instantiate_kernel("v_" #op #tname, unary_v, type, op)        \
  instantiate_kernel("v2_" #op #tname, unary_v2, type, op)      \
  instantiate_kernel("gn4_" #op #tname, unary_g, type, op, 4)

#define instantiate_unary_float(op)               \
  instantiate_unary_all(op, float16, half)        \
  instantiate_unary_all(op, float32, float)       \
  instantiate_unary_all(op, bfloat16, bfloat16_t)

#define instantiate_unary_types(op)           \
  instantiate_unary_all(op, bool_, bool)      \
  instantiate_unary_all(op, uint8, uint8_t)   \
  instantiate_unary_all(op, uint16, uint16_t) \
  instantiate_unary_all(op, uint32, uint32_t) \
  instantiate_unary_all(op, uint64, uint64_t) \
  instantiate_unary_all(op, int8, int8_t)     \
  instantiate_unary_all(op, int16, int16_t)   \
  instantiate_unary_all(op, int32, int32_t)   \
  instantiate_unary_all(op, int64, int64_t)   \
  instantiate_unary_float(op)

instantiate_unary_types(Abs)
instantiate_unary_float(ArcCos)
instantiate_unary_float(ArcCosh)
instantiate_unary_float(ArcSin)
instantiate_unary_float(ArcSinh)
instantiate_unary_float(ArcTan)
instantiate_unary_float(ArcTanh)
instantiate_unary_types(Ceil)
instantiate_unary_float(Cos)
instantiate_unary_float(Cosh)
instantiate_unary_float(Exp)
instantiate_unary_float(Expm1)
instantiate_unary_types(Floor)
instantiate_unary_float(Log)
instantiate_unary_float(Log2)
instantiate_unary_float(Log10)
instantiate_unary_float(Log1p)
instantiate_unary_types(Negative)
instantiate_unary_float(Sigmoid)
instantiate_unary_float(Erf)
instantiate_unary_float(ErfInv)
instantiate_unary_types(Sign)
instantiate_unary_float(Sin)
instantiate_unary_float(Sinh)
instantiate_unary_types(Square)
instantiate_unary_float(Sqrt)
instantiate_unary_float(Rsqrt)
instantiate_unary_float(Tan)
instantiate_unary_float(Tanh)
instantiate_unary_float(Round)

instantiate_unary_all(Abs, complex64, complex64_t)
instantiate_unary_all(Conjugate, complex64, complex64_t)
instantiate_unary_all(Cos, complex64, complex64_t)
instantiate_unary_all(Cosh, complex64, complex64_t)
instantiate_unary_all(Exp, complex64, complex64_t)
instantiate_unary_all(Negative, complex64, complex64_t)
instantiate_unary_all(Sign, complex64, complex64_t)
instantiate_unary_all(Sin, complex64, complex64_t)
instantiate_unary_all(Sinh, complex64, complex64_t)
instantiate_unary_all(Tan, complex64, complex64_t)
instantiate_unary_all(Tanh, complex64, complex64_t)
instantiate_unary_all(Round, complex64, complex64_t)

instantiate_unary_all(LogicalNot, bool_, bool) // clang-format on
