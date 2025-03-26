// Copyright © 2023-2024 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>

using namespace metal;

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/logsumexp.h"

#define instantiate_logsumexp(name, itype)                          \
  template [[host_name("block_logsumexp_" #name)]] [[kernel]] void        \
  logsumexp<itype>(                                      \
      const device itype* in,                                     \
      device itype* out,                                          \
      constant int& axis_size,                                    \
      uint gid [[thread_position_in_grid]],                       \
      uint _lid [[thread_position_in_threadgroup]],               \
      uint simd_lane_id [[thread_index_in_simdgroup]],            \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);     \
  template [[host_name("looped_logsumexp_" #name)]] [[kernel]] void \
  logsumexp_looped<itype>(                                          \
      const device itype* in,                                     \
      device itype* out,                                          \
      constant int& axis_size,                                    \
      uint gid [[threadgroup_position_in_grid]],                  \
      uint lid [[thread_position_in_threadgroup]],                \
      uint lsize [[threads_per_threadgroup]],                     \
      uint simd_lane_id [[thread_index_in_simdgroup]],            \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

instantiate_logsumexp(float32, float)
instantiate_logsumexp(float16, half)
instantiate_logsumexp(bfloat16, bfloat16_t) // clang-format on
