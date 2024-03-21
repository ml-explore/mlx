// Copyright © 2024 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void rms_single_row(
    const device T* x,
    const device T* w,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    threadgroup float* local_inv_mean [[threadgroup(0)]],
    threadgroup float* local_sums [[threadgroup(1)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

  float acc = 0;
  x += gid * axis_size + lid * N_READS;
  w += w_stride * lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
       acc += x[i] * x[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        acc += x[i] * x[i];
      }
    }
  }
  acc = simd_sum(acc);
  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sums[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sums[simd_group_id] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    acc = simd_sum(local_sums[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_inv_mean[0] = metal::precise::rsqrt(acc / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write the outputs
  out += gid * axis_size + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i=0; i < N_READS; i++) {
      out[i] = w[w_stride * i] * static_cast<T>(x[i] * local_inv_mean[0]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        out[i] = w[w_stride * i] * static_cast<T>(x[i] * local_inv_mean[0]);
      }
    }
  }
}

#define instantiate_rms_single_row(name, itype)           \
  template [[host_name("rms" #name)]] [[kernel]] void    \
  rms_single_row<itype>(                                  \
      const device itype* x,                                  \
      const device itype* w,                                  \
      device itype* out,                                      \
      constant float& eps,                                \
      constant uint& axis_size,                                \
      constant uint& w_stride,                                \
      threadgroup float* local_inv_mean [[threadgroup(0)]],   \
      threadgroup float* local_sums [[threadgroup(1)]],       \
      uint gid [[thread_position_in_grid]],                   \
      uint lid [[thread_position_in_threadgroup]],            \
      uint simd_lane_id [[thread_index_in_simdgroup]],        \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_rms(name, itype)      \
  instantiate_rms_single_row(name, itype)

instantiate_rms(float32, float)
instantiate_rms(float16, half)
instantiate_rms(bfloat16, bfloat16_t)
