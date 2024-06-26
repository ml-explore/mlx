// Copyright © 2023-2024 Apple Inc.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sort.h"

#define instantiate_block_sort(                                          \
    name, itname, itype, otname, otype, arg_sort, bn, tn)                \
  instantiate_kernel("c" #name "_" #itname "_" #otname "_bn" #bn "_tn" #tn, \
                     block_sort, itype, otype, arg_sort, bn, tn) \
  instantiate_kernel("nc" #name "_" #itname "_" #otname "_bn" #bn "_tn" #tn, \
                     block_sort_nc, itype, otype, arg_sort, bn, tn)

#define instantiate_arg_block_sort_base(itname, itype, bn, tn) \
  instantiate_block_sort(                                      \
      arg_block_sort, itname, itype, uint32, uint32_t, true, bn, tn)

#define instantiate_block_sort_base(itname, itype, bn, tn) \
  instantiate_block_sort(                                  \
      _block_sort, itname, itype, itname, itype, false, bn, tn)

#define instantiate_block_sort_tn(itname, itype, bn) \
  instantiate_block_sort_base(itname, itype, bn, 8)  \
  instantiate_arg_block_sort_base(itname, itype, bn, 8)

#define instantiate_block_sort_bn(itname, itype) \
  instantiate_block_sort_tn(itname, itype, 128)  \
  instantiate_block_sort_tn(itname, itype, 256)  \
  instantiate_block_sort_tn(itname, itype, 512)

instantiate_block_sort_bn(uint8, uint8_t)
instantiate_block_sort_bn(uint16, uint16_t)
instantiate_block_sort_bn(uint32, uint32_t)
instantiate_block_sort_bn(int8, int8_t)
instantiate_block_sort_bn(int16, int16_t)
instantiate_block_sort_bn(int32, int32_t)
instantiate_block_sort_bn(float16, half)
instantiate_block_sort_bn(float32, float)
instantiate_block_sort_bn(bfloat16, bfloat16_t)

#define instantiate_block_sort_long(itname, itype) \
  instantiate_block_sort_tn(itname, itype, 128)    \
  instantiate_block_sort_tn(itname, itype, 256)

instantiate_block_sort_long(uint64, uint64_t)
instantiate_block_sort_long(int64, int64_t)

#define instantiate_multi_block_sort(                                      \
    vtname, vtype, itname, itype, arg_sort, bn, tn)                        \
  instantiate_kernel("sort_mbsort_" #vtname "_" #itname "_bn" #bn "_tn" #tn, \
                     mb_block_sort, vtype, itype, arg_sort, bn, tn) \
  instantiate_kernel("partition_mbsort_" #vtname "_" #itname "_bn" #bn "_tn" #tn, \
                     mb_block_partition, vtype, itype, arg_sort, bn, tn) \
  instantiate_kernel("merge_mbsort_" #vtname "_" #itname "_bn" #bn "_tn" #tn, \
                     mb_block_merge, vtype, itype, arg_sort, bn, tn)

#define instantiate_multi_block_sort_base(vtname, vtype) \
  instantiate_multi_block_sort(vtname, vtype, uint32, uint32_t, true, 512, 8)

instantiate_multi_block_sort_base(uint8, uint8_t)
instantiate_multi_block_sort_base(uint16, uint16_t)
instantiate_multi_block_sort_base(uint32, uint32_t)
instantiate_multi_block_sort_base(int8, int8_t)
instantiate_multi_block_sort_base(int16, int16_t)
instantiate_multi_block_sort_base(int32, int32_t)
instantiate_multi_block_sort_base(float16, half)
instantiate_multi_block_sort_base(float32, float)
instantiate_multi_block_sort_base(bfloat16, bfloat16_t)

#define instantiate_multi_block_sort_long(vtname, vtype) \
  instantiate_multi_block_sort(vtname, vtype, uint32, uint32_t, true, 256, 8)

instantiate_multi_block_sort_long(uint64, uint64_t)
instantiate_multi_block_sort_long(int64, int64_t) // clang-format on
