// Copyright © 2023-2024 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>

using namespace metal;

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/softmax.h"

#define instantiate_softmax(name, itype)                          \
  template [[host_name("block_softmax_" #name)]] [[kernel]] void        \
  softmax_single_row<itype>(                                      \
      const device itype* in,                                     \
      device itype* out,                                          \
      constant int& axis_size,                                    \
      uint gid [[thread_position_in_grid]],                       \
      uint _lid [[thread_position_in_threadgroup]],               \
      uint simd_lane_id [[thread_index_in_simdgroup]],            \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);     \
  template [[host_name("looped_softmax_" #name)]] [[kernel]] void \
  softmax_looped<itype>(                                          \
      const device itype* in,                                     \
      device itype* out,                                          \
      constant int& axis_size,                                    \
      uint gid [[threadgroup_position_in_grid]],                  \
      uint lid [[thread_position_in_threadgroup]],                \
      uint lsize [[threads_per_threadgroup]],                     \
      uint simd_lane_id [[thread_index_in_simdgroup]],            \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax_precise(name, itype)                          \
  template [[host_name("block_softmax_precise_" #name)]] [[kernel]] void        \
  softmax_single_row<itype, float>(                                       \
      const device itype* in,                                             \
      device itype* out,                                                  \
      constant int& axis_size,                                            \
      uint gid [[thread_position_in_grid]],                               \
      uint _lid [[thread_position_in_threadgroup]],                       \
      uint simd_lane_id [[thread_index_in_simdgroup]],                    \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);             \
  template [[host_name("looped_softmax_precise_" #name)]] [[kernel]] void \
  softmax_looped<itype, float>(                                           \
      const device itype* in,                                             \
      device itype* out,                                                  \
      constant int& axis_size,                                            \
      uint gid [[threadgroup_position_in_grid]],                          \
      uint lid [[thread_position_in_threadgroup]],                        \
      uint lsize [[threads_per_threadgroup]],                             \
      uint simd_lane_id [[thread_index_in_simdgroup]],                    \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

instantiate_softmax(float32, float)
instantiate_softmax(float16, half)
instantiate_softmax(bfloat16, bfloat16_t)
instantiate_softmax_precise(float16, half)
instantiate_softmax_precise(bfloat16, bfloat16_t) // clang-format on
