// Copyright © 2024 Apple Inc.
#include <metal_integer>
#include <metal_math>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/binary_ops.h"
#include "mlx/backend/metal/kernels/binary_two.h"

#define instantiate_binary_all(op, tname, itype, otype)                     \
  instantiate_kernel("ss_" #op #tname, binary_ss, itype, otype, op)         \
  instantiate_kernel("sv_" #op #tname, binary_sv, itype, otype, op)         \
  instantiate_kernel("vs_" #op #tname, binary_vs, itype, otype, op)         \
  instantiate_kernel("vv_" #op #tname, binary_vv, itype, otype, op)         \
  instantiate_kernel("sv2_" #op #tname, binary_sv2, itype, otype, op)       \
  instantiate_kernel("vs2_" #op #tname, binary_vs2, itype, otype, op)       \
  instantiate_kernel("vv2_" #op #tname, binary_vv2, itype, otype, op)       \
  instantiate_kernel("gn2_" #op #tname, binary_g, itype, otype, op, 2, int) \
  instantiate_kernel("gn4large_" #op #tname, binary_g, itype, otype, op, 4) \
  instantiate_kernel("g1_" #op #tname, binary_g_nd1, itype, otype, op, int) \
  instantiate_kernel("g2_" #op #tname, binary_g_nd2, itype, otype, op, int) \
  instantiate_kernel("g3_" #op #tname, binary_g_nd3, itype, otype, op, int) \
  instantiate_kernel("g1large_" #op #tname, binary_g_nd1, itype, otype, op) \
  instantiate_kernel("g2large_" #op #tname, binary_g_nd2, itype, otype, op) \
  instantiate_kernel("g3large_" #op #tname, binary_g_nd3, itype, otype, op)

#define instantiate_binary_float(op)                \
  instantiate_binary_all(op, float16, half, half)   \
  instantiate_binary_all(op, float32, float, float) \
  instantiate_binary_all(op, bfloat16, bfloat16_t, bfloat16_t)

#define instantiate_binary_types(op)                              \
  instantiate_binary_all(op, bool_, bool, bool)                   \
  instantiate_binary_all(op, uint8, uint8_t, uint8_t)             \
  instantiate_binary_all(op, uint16, uint16_t, uint16_t)          \
  instantiate_binary_all(op, uint32, uint32_t, uint32_t)          \
  instantiate_binary_all(op, uint64, uint64_t, uint64_t)          \
  instantiate_binary_all(op, int8, int8_t, int8_t)                \
  instantiate_binary_all(op, int16, int16_t, int16_t)             \
  instantiate_binary_all(op, int32, int32_t, int32_t)             \
  instantiate_binary_all(op, int64, int64_t, int64_t)             \
  instantiate_binary_all(op, complex64, complex64_t, complex64_t) \
  instantiate_binary_float(op)

instantiate_binary_types(DivMod) // clang-format on
