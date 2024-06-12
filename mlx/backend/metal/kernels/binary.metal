// Copyright © 2024 Apple Inc.

#include <metal_integer>
#include <metal_math>

// clang-format off
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/binary_ops.h"
#include "mlx/backend/metal/kernels/binary.h"

#define instantiate_binary_all(op, tname, itype, otype)                 \
  instantiate_kernel("ss" #op #tname, binary_ss, itype, otype, op)      \
  instantiate_kernel("sv" #op #tname, binary_sv, itype, otype, op)      \
  instantiate_kernel("vs" #op #tname, binary_vs, itype, otype, op)      \
  instantiate_kernel("vv" #op #tname, binary_vv, itype, otype, op)      \
  instantiate_kernel("gn" #op #tname, binary_g, itype, otype, op)       \
  instantiate_kernel("g1" #op #tname, binary_g_nd1, itype, otype, op)   \
  instantiate_kernel("g2" #op #tname, binary_g_nd2, itype, otype, op)   \
  instantiate_kernel("g3" #op #tname, binary_g_nd3, itype, otype, op)   \
  instantiate_kernel("g4" #op #tname, binary_g_nd, itype, otype, op, 4) \
  instantiate_kernel("g5" #op #tname, binary_g_nd,  itype, otype, op, 5)

#define instantiate_binary_integer(op)                   \
  instantiate_binary_all(op, uint8, uint8_t, uint8_t)    \
  instantiate_binary_all(op, uint16, uint16_t, uint16_t) \
  instantiate_binary_all(op, uint32, uint32_t, uint32_t) \
  instantiate_binary_all(op, uint64, uint64_t, uint64_t) \
  instantiate_binary_all(op, int8, int8_t, int8_t)       \
  instantiate_binary_all(op, int16, int16_t, int16_t)    \
  instantiate_binary_all(op, int32, int32_t, int32_t)    \
  instantiate_binary_all(op, int64, int64_t, int64_t)

#define instantiate_binary_float(op)                \
  instantiate_binary_all(op, float16, half, half)   \
  instantiate_binary_all(op, float32, float, float) \
  instantiate_binary_all(op, bfloat16, bfloat16_t, bfloat16_t)

#define instantiate_binary_types(op)                              \
  instantiate_binary_all(op, bool_, bool, bool)                   \
  instantiate_binary_integer(op)                                  \
  instantiate_binary_all(op, complex64, complex64_t, complex64_t) \
  instantiate_binary_float(op)

#define instantiate_binary_types_bool(op)                \
  instantiate_binary_all(op, bool_, bool, bool)          \
  instantiate_binary_all(op, uint8, uint8_t, bool)       \
  instantiate_binary_all(op, uint16, uint16_t, bool)     \
  instantiate_binary_all(op, uint32, uint32_t, bool)     \
  instantiate_binary_all(op, uint64, uint64_t, bool)     \
  instantiate_binary_all(op, int8, int8_t, bool)         \
  instantiate_binary_all(op, int16, int16_t, bool)       \
  instantiate_binary_all(op, int32, int32_t, bool)       \
  instantiate_binary_all(op, int64, int64_t, bool)       \
  instantiate_binary_all(op, float16, half, bool)        \
  instantiate_binary_all(op, float32, float, bool)       \
  instantiate_binary_all(op, bfloat16, bfloat16_t, bool) \
  instantiate_binary_all(op, complex64, complex64_t, bool)

instantiate_binary_types(Add)
instantiate_binary_types(Divide)
instantiate_binary_types_bool(Equal)
instantiate_binary_types_bool(Greater)
instantiate_binary_types_bool(GreaterEqual)
instantiate_binary_types_bool(Less)
instantiate_binary_types_bool(LessEqual)
instantiate_binary_types_bool(NotEqual)
instantiate_binary_float(LogAddExp)
instantiate_binary_types(Maximum)
instantiate_binary_types(Minimum)
instantiate_binary_types(Multiply)
instantiate_binary_types(Subtract)
instantiate_binary_types(Power)
instantiate_binary_types(Remainder)
instantiate_binary_float(ArcTan2)

// NaNEqual only needed for floating point types with boolean output
instantiate_binary_all(NaNEqual, float16, half, bool)
instantiate_binary_all(NaNEqual, float32, float, bool)
instantiate_binary_all(NaNEqual, bfloat16, bfloat16_t, bool)
instantiate_binary_all(NaNEqual, complex64, complex64_t, bool)

instantiate_binary_all(LogicalOr, bool_, bool, bool)
instantiate_binary_all(LogicalAnd, bool_, bool, bool)

// Bitwise ops only need integer types and bool (except for l/r shift)
instantiate_binary_integer(BitwiseAnd)
instantiate_binary_all(BitwiseAnd, bool_, bool, bool)
instantiate_binary_integer(BitwiseOr)
instantiate_binary_all(BitwiseOr, bool_, bool, bool)
instantiate_binary_integer(BitwiseXor)
instantiate_binary_all(BitwiseXor, bool_, bool, bool)
instantiate_binary_integer(LeftShift)
instantiate_binary_integer(RightShift) // clang-format on
