// Copyright © 2023-2024 Apple Inc.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sort.h"

#define instantiate_block_sort(                                          \
    name, itname, itype, otname, otype, arg_sort, bn, tn)                \
  template [[host_name("c" #name "_" #itname "_" #otname "_bn" #bn           \
                             "_tn" #tn)]] [[kernel]] void                \
  block_sort<itype, otype, arg_sort, bn, tn>(                            \
      const device itype* inp [[buffer(0)]],                             \
      device otype* out [[buffer(1)]],                                   \
      const constant int& size_sorted_axis [[buffer(2)]],                \
      const constant int& stride_sorted_axis [[buffer(3)]],              \
      const constant int& stride_segment_axis [[buffer(4)]],             \
      uint3 tid [[threadgroup_position_in_grid]],                        \
      uint3 lid [[thread_position_in_threadgroup]]);                     \
  template [[host_name("nc" #name "_" #itname "_" #otname "_bn" #bn "_tn" #tn \
                             )]] [[kernel]] void                         \
  block_sort_nc<itype, otype, arg_sort, bn, tn>(                         \
      const device itype* inp [[buffer(0)]],                             \
      device otype* out [[buffer(1)]],                                   \
      const constant int& size_sorted_axis [[buffer(2)]],                \
      const constant int& stride_sorted_axis [[buffer(3)]],              \
      const constant int& nc_dim [[buffer(4)]],                          \
      const device int* nc_shape [[buffer(5)]],                          \
      const device size_t* nc_strides [[buffer(6)]],                     \
      uint3 tid [[threadgroup_position_in_grid]],                        \
      uint3 lid [[thread_position_in_threadgroup]]);

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
  template [[host_name("sort_mbsort_" #vtname "_" #itname "_bn" #bn      \
                       "_tn" #tn)]] [[kernel]] void                        \
  mb_block_sort<vtype, itype, arg_sort, bn, tn>(                           \
      const device vtype* inp [[buffer(0)]],                               \
      device vtype* out_vals [[buffer(1)]],                                \
      device itype* out_idxs [[buffer(2)]],                                \
      const constant int& size_sorted_axis [[buffer(3)]],                  \
      const constant int& stride_sorted_axis [[buffer(4)]],                \
      const constant int& nc_dim [[buffer(5)]],                            \
      const device int* nc_shape [[buffer(6)]],                            \
      const device size_t* nc_strides [[buffer(7)]],                       \
      uint3 tid [[threadgroup_position_in_grid]],                          \
      uint3 lid [[thread_position_in_threadgroup]]);                       \
  template [[host_name("partition_mbsort_" #vtname "_" #itname "_bn" #bn \
                       "_tn" #tn)]] [[kernel]] void                        \
  mb_block_partition<vtype, itype, arg_sort, bn, tn>(                      \
      device itype * block_partitions [[buffer(0)]],                       \
      const device vtype* dev_vals [[buffer(1)]],                          \
      const device itype* dev_idxs [[buffer(2)]],                          \
      const constant int& size_sorted_axis [[buffer(3)]],                  \
      const constant int& merge_tiles [[buffer(4)]],                       \
      uint3 tid [[threadgroup_position_in_grid]],                          \
      uint3 lid [[thread_position_in_threadgroup]],                        \
      uint3 tgp_dims [[threads_per_threadgroup]]);                         \
  template [[host_name("merge_mbsort_" #vtname "_" #itname "_bn" #bn     \
                       "_tn" #tn)]] [[kernel]] void                        \
  mb_block_merge<vtype, itype, arg_sort, bn, tn>(                          \
      const device itype* block_partitions [[buffer(0)]],                  \
      const device vtype* dev_vals_in [[buffer(1)]],                       \
      const device itype* dev_idxs_in [[buffer(2)]],                       \
      device vtype* dev_vals_out [[buffer(3)]],                            \
      device itype* dev_idxs_out [[buffer(4)]],                            \
      const constant int& size_sorted_axis [[buffer(5)]],                  \
      const constant int& merge_tiles [[buffer(6)]],                       \
      const constant int& num_tiles [[buffer(7)]],                         \
      uint3 tid [[threadgroup_position_in_grid]],                          \
      uint3 lid [[thread_position_in_threadgroup]]);

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
