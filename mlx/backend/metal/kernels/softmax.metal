// Copyright © 2023 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename T>
inline T softmax_exp(T x) {
  // Softmax doesn't need high precision exponential cause it is gonna be x
  // will be in (-oo, 0] anyway and subsequently it will be divided by
  // sum(exp(x_i)).
  return fast::exp(x);
}

template <typename T, int N_READS = SOFTMAX_N_READS>
[[kernel]] void softmax_single_row(
    const device T* in,
    device T* out,
    constant int& axis_size,
    threadgroup T* local_max [[threadgroup(0)]],
    threadgroup T* local_normalizer [[threadgroup(1)]],
    uint gid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  int lid = _lid;

  T ld[N_READS];

  in += gid * axis_size + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i=0; i<N_READS; i++) {
        ld[i] = in[i];
    }
  } else {
      for (int i = 0; i < N_READS; i++) {
        ld[i] =
            ((lid * N_READS + i) < axis_size) ? in[i] : T(Limits<T>::finite_min);
      }
  }
  if (simd_group_id == 0) {
    local_max[simd_lane_id] = Limits<T>::finite_min;
    local_normalizer[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Get the max
  T maxval = Limits<T>::finite_min;
  for (int i = 0; i < N_READS; i++) {
    maxval = (maxval < ld[i]) ? ld[i] : maxval;
  }
  maxval = simd_max(maxval);
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    maxval = simd_max(local_max[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_max[0] = maxval;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = local_max[0];

  // Compute exp(x_i - maxval) and store the partial sums in local_normalizer
  T normalizer = 0;
  for (int i = 0; i < N_READS; i++) {
    T exp_x = softmax_exp(ld[i] - maxval);
    ld[i] = exp_x;
    normalizer += exp_x;
  }
  normalizer = simd_sum(normalizer);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    normalizer = simd_sum(local_normalizer[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_normalizer[0] = normalizer;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = 1 / local_normalizer[0];

  // Normalize and write to the output
  out += gid * axis_size + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i=0; i<N_READS; i++) {
        out[i] = ld[i] * normalizer;
    }
  } else {
      for (int i = 0; i < N_READS; i++) {
        if ((lid * N_READS + i) < axis_size) {
          out[i] = ld[i] * normalizer;
        }
      }
  }
}

template <typename T, int N_READS = SOFTMAX_N_READS>
[[kernel]] void softmax_looped(
    const device T* in,
    device T* out,
    constant int& axis_size,
    threadgroup T* local_max [[threadgroup(0)]],
    threadgroup T* local_normalizer [[threadgroup(1)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  in += gid * axis_size;

  // Get the max and the normalizer in one go
  T prevmax;
  T maxval = Limits<T>::finite_min;
  T normalizer = 0;
  for (int r = 0; r < static_cast<int>(ceildiv(axis_size, N_READS * lsize));
       r++) {
    int offset = r * lsize * N_READS + lid * N_READS;
    T vals[N_READS];
    if (offset + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        vals[i] = in[offset + i];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        vals[i] =
            (offset + i < axis_size) ? in[offset + i] : T(Limits<T>::finite_min);
      }
    }
    prevmax = maxval;
    for (int i = 0; i < N_READS; i++) {
      maxval = (maxval < vals[i]) ? vals[i] : maxval;
    }
    normalizer *= softmax_exp(prevmax - maxval);
    for (int i = 0; i < N_READS; i++) {
      normalizer += softmax_exp(vals[i] - maxval);
    }
  }
  // Now we got partial normalizer of N_READS * ceildiv(axis_size, N_READS *
  // lsize) parts. We need to combine them.
  //    1. We start by finding the max across simd groups
  //    2. We then change the partial normalizers to account for a possible
  //       change in max
  //    3. We sum all normalizers
  prevmax = maxval;
  maxval = simd_max(maxval);
  normalizer *= softmax_exp(prevmax - maxval);
  normalizer = simd_sum(normalizer);

  // Now the normalizer and max value is correct for each simdgroup. We write
  // them shared memory and combine them.
  prevmax = maxval;
  if (simd_lane_id == 0) {
    local_max[simd_group_id] = maxval;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  maxval = simd_max(local_max[simd_lane_id]);
  normalizer *= softmax_exp(prevmax - maxval);
  if (simd_lane_id == 0) {
    local_normalizer[simd_group_id] = normalizer;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  normalizer = simd_sum(local_normalizer[simd_lane_id]);
  normalizer = 1 / normalizer;

  // Finally given the normalizer and max value we can directly write the
  // softmax output
  out += gid * axis_size;
  for (int r = 0; r < static_cast<int>(ceildiv(axis_size, N_READS * lsize));
       r++) {
    int offset = r * lsize * N_READS + lid * N_READS;
    if (offset + N_READS <= axis_size) {
      for (int i=0; i<N_READS; i++) {
        out[offset + i] = softmax_exp(in[offset + i] - maxval) * normalizer;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if (offset + i < axis_size) {
          out[offset + i] = softmax_exp(in[offset + i] - maxval) * normalizer;
        }
      }
    }
  }
}

#define instantiate_softmax_single_row(name, itype)           \
  template [[host_name("softmax_" #name)]] [[kernel]] void    \
  softmax_single_row<itype>(                                  \
      const device itype* in,                                 \
      device itype* out,                                      \
      constant int& axis_size,                                \
      threadgroup itype* local_max [[threadgroup(0)]],        \
      threadgroup itype* local_normalizer [[threadgroup(1)]], \
      uint gid [[thread_position_in_grid]],                   \
      uint _lid [[thread_position_in_threadgroup]],           \
      uint simd_lane_id [[thread_index_in_simdgroup]],        \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax_looped(name, itype)                   \
  template [[host_name("softmax_looped_" #name)]] [[kernel]] void \
  softmax_looped<itype>(                                          \
      const device itype* in,                                     \
      device itype* out,                                          \
      constant int& axis_size,                                    \
      threadgroup itype* local_max [[threadgroup(0)]],            \
      threadgroup itype* local_normalizer [[threadgroup(1)]],     \
      uint gid [[threadgroup_position_in_grid]],                  \
      uint lid [[thread_position_in_threadgroup]],                \
      uint lsize [[threads_per_threadgroup]],                     \
      uint simd_lane_id [[thread_index_in_simdgroup]],            \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_softmax(name, itype)      \
  instantiate_softmax_single_row(name, itype) \
      instantiate_softmax_looped(name, itype)

instantiate_softmax(float32, float)
instantiate_softmax(float16, half)
instantiate_softmax(bfloat16, bfloat16_t)
