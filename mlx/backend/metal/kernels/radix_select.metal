// Copyright © 2025 Apple Inc.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/radix_select.h"

///////////////////////////////////////////////////////////////////////////////
// Radix Select Kernel Instantiations
///////////////////////////////////////////////////////////////////////////////

#define instantiate_radix_select(name, itname, itype, otname, otype, arg_part, bn, tn) \
  instantiate_kernel(                                                                   \
      "c" #name "_" #itname "_" #otname "_bn" #bn "_tn" #tn,                            \
      radix_select_partition,                                                           \
      itype,                                                                            \
      otype,                                                                            \
      arg_part,                                                                         \
      bn,                                                                               \
      tn)                                                                               \
  instantiate_kernel(                                                                   \
      "nc" #name "_" #itname "_" #otname "_bn" #bn "_tn" #tn,                           \
      radix_select_partition_nc,                                                        \
      itype,                                                                            \
      otype,                                                                            \
      arg_part,                                                                         \
      bn,                                                                               \
      tn)

#define instantiate_radix_select_arg(itname, itype, bn, tn) \
  instantiate_radix_select(                                 \
      arg_radix_select, itname, itype, uint32, uint32_t, true, bn, tn)

#define instantiate_radix_select_val(itname, itype, bn, tn) \
  instantiate_radix_select(                                 \
      _radix_select, itname, itype, itname, itype, false, bn, tn)

#define instantiate_radix_select_tn(itname, itype, bn) \
  instantiate_radix_select_arg(itname, itype, bn, 8)   \
  instantiate_radix_select_val(itname, itype, bn, 8)

#define instantiate_radix_select_bn(itname, itype) \
  instantiate_radix_select_tn(itname, itype, 256)

// Instantiate for all supported types
instantiate_radix_select_bn(uint8, uint8_t)
instantiate_radix_select_bn(uint16, uint16_t)
instantiate_radix_select_bn(uint32, uint32_t)
instantiate_radix_select_bn(int8, int8_t)
instantiate_radix_select_bn(int16, int16_t)
instantiate_radix_select_bn(int32, int32_t)
instantiate_radix_select_bn(float16, half)
instantiate_radix_select_bn(float32, float)
instantiate_radix_select_bn(bfloat16, bfloat16_t)

// 64-bit types with smaller block size due to memory constraints
#define instantiate_radix_select_long(itname, itype) \
  instantiate_radix_select_tn(itname, itype, 128)

instantiate_radix_select_long(uint64, uint64_t)
instantiate_radix_select_long(int64, int64_t)
// clang-format on
