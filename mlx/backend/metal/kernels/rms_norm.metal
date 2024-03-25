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
      float xi = x[i];
      acc += xi * xi;
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        float xi = x[i];
        acc += xi * xi;
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
    for (int i = 0; i < N_READS; i++) {
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

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void rms_looped(
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
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  float acc = 0;
  x += gid * axis_size + lid * N_READS;
  w += w_stride * lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        acc += xi * xi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          acc += xi * xi;
        }
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
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        out[r + i] = w[w_stride * (i + r)] *
            static_cast<T>(x[r + i] * local_inv_mean[0]);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          out[r + i] = w[w_stride * (i + r)] *
              static_cast<T>(x[r + i] * local_inv_mean[0]);
        }
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void vjp_rms_single_row(
    const device T* x,
    const device T* w,
    const device T* g,
    device T* gx,
    device T* gw,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  // Advance the input pointers
  x += gid * axis_size + lid * N_READS;
  g += gid * axis_size + lid * N_READS;
  w += w_stride * lid * N_READS;

  // Allocate registers for the computation and accumulators
  float thread_x[N_READS];
  float thread_w[N_READS];
  float thread_g[N_READS];
  float sumx2 = 0;
  float sumgwx = 0;

  // Allocate shared memory to implement the reduction
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_sumx2[SIMD_SIZE];
  threadgroup float local_sumgwx[SIMD_SIZE];
  threadgroup float local_normalizer[1];
  threadgroup float local_meangwx[1];

  // Read and accumulate locally
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
      thread_w[i] = w[w_stride * i];
      thread_g[i] = g[i];

      sumx2 += thread_x[i] * thread_x[i];
      sumgwx += thread_x[i] * thread_w[i] * thread_g[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        thread_x[i] = x[i];
        thread_w[i] = w[w_stride * i];
        thread_g[i] = g[i];

        sumx2 += thread_x[i] * thread_x[i];
        sumgwx += thread_x[i] * thread_w[i] * thread_g[i];
      }
    }
  }

  // Accumulate across threads
  sumx2 = simd_sum(sumx2);
  sumgwx = simd_sum(sumgwx);
  if (simd_group_id == 0) {
    local_sumx2[simd_lane_id] = 0;
    local_sumgwx[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sumx2[simd_group_id] = sumx2;
    local_sumgwx[simd_group_id] = sumgwx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    sumx2 = simd_sum(local_sumx2[simd_lane_id]);
    sumgwx = simd_sum(local_sumgwx[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_meangwx[0] = sumgwx / axis_size;
      local_normalizer[0] = metal::precise::rsqrt(sumx2 / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float meangwx = local_meangwx[0];
  float normalizer = local_normalizer[0];
  float normalizer3 = normalizer * normalizer * normalizer;

  // Write the outputs
  gx += gid * axis_size + lid * N_READS;
  gw += gid * axis_size + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      gx[i] = static_cast<T>(thread_g[i] * thread_w[i] * normalizer - thread_x[i] * meangwx * normalizer3);
      gw[i] = static_cast<T>(thread_g[i] * thread_x[i] * normalizer);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        gx[i] = static_cast<T>(thread_g[i] * thread_w[i] * normalizer - thread_x[i] * meangwx * normalizer3);
        gw[i] = static_cast<T>(thread_g[i] * thread_x[i] * normalizer);
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void vjp_rms_looped(
    const device T* x,
    const device T* w,
    const device T* g,
    device T* gx,
    device T* gw,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  // Advance the input pointers
  x += gid * axis_size + lid * N_READS;
  g += gid * axis_size + lid * N_READS;
  w += w_stride * lid * N_READS;

  // Allocate registers for the accumulators
  float sumx2 = 0;
  float sumgwx = 0;

  // Allocate shared memory to implement the reduction
  constexpr int SIMD_SIZE = 32;
  threadgroup float local_sumx2[SIMD_SIZE];
  threadgroup float local_sumgwx[SIMD_SIZE];
  threadgroup float local_normalizer[1];
  threadgroup float local_meangwx[1];

  // Read and accumulate locally
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        float wi = w[w_stride * (i + r)];
        float gi = g[i + r];

        sumx2 += xi * xi;
        sumgwx += xi * wi * gi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          float wi = w[w_stride * (i + r)];
          float gi = g[i + r];

          sumx2 += xi * xi;
          sumgwx += xi * wi * gi;
        }
      }
    }
  }

  // Accumulate across threads
  sumx2 = simd_sum(sumx2);
  sumgwx = simd_sum(sumgwx);
  if (simd_group_id == 0) {
    local_sumx2[simd_lane_id] = 0;
    local_sumgwx[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    local_sumx2[simd_group_id] = sumx2;
    local_sumgwx[simd_group_id] = sumgwx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group_id == 0) {
    sumx2 = simd_sum(local_sumx2[simd_lane_id]);
    sumgwx = simd_sum(local_sumgwx[simd_lane_id]);
    if (simd_lane_id == 0) {
      local_meangwx[0] = sumgwx / axis_size;
      local_normalizer[0] = metal::precise::rsqrt(sumx2 / axis_size + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float meangwx = local_meangwx[0];
  float normalizer = local_normalizer[0];
  float normalizer3 = normalizer * normalizer * normalizer;

  // Write the outputs
  gx += gid * axis_size + lid * N_READS;
  gw += gid * axis_size + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        float wi = w[w_stride * (i + r)];
        float gi = g[i + r];

        gx[i + r] = static_cast<T>(gi * wi * normalizer - xi * meangwx * normalizer3);
        gw[i + r] = static_cast<T>(gi * xi * normalizer);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          float wi = w[w_stride * (i + r)];
          float gi = g[i + r];

          gx[i + r] = static_cast<T>(gi * wi * normalizer - xi * meangwx * normalizer3);
          gw[i + r] = static_cast<T>(gi * xi * normalizer);
        }
      }
    }
  }
}

// clang-format off
#define instantiate_rms_single_row(name, itype)               \
  template [[host_name("rms" #name)]] [[kernel]] void         \
  rms_single_row<itype>(                                      \
      const device itype* x,                                  \
      const device itype* w,                                  \
      device itype* out,                                      \
      constant float& eps,                                    \
      constant uint& axis_size,                               \
      constant uint& w_stride,                                \
      threadgroup float* local_inv_mean [[threadgroup(0)]],   \
      threadgroup float* local_sums [[threadgroup(1)]],       \
      uint gid [[thread_position_in_grid]],                   \
      uint lid [[thread_position_in_threadgroup]],            \
      uint simd_lane_id [[thread_index_in_simdgroup]],        \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]); \
                                                              \
  template [[host_name("vjp_rms" #name)]] [[kernel]] void     \
  vjp_rms_single_row<itype>(                                  \
      const device itype* x,                                  \
      const device itype* w,                                  \
      const device itype* g,                                  \
      device itype* gx,                                       \
      device itype* gw,                                       \
      constant float& eps,                                    \
      constant uint& axis_size,                               \
      constant uint& w_stride,                                \
      uint gid [[thread_position_in_grid]],                   \
      uint lid [[thread_position_in_threadgroup]],            \
      uint simd_lane_id [[thread_index_in_simdgroup]],        \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_rms_looped(name, itype)                      \
  template [[host_name("rms_looped" #name)]] [[kernel]] void     \
  rms_looped<itype>(                                             \
      const device itype* x,                                     \
      const device itype* w,                                     \
      device itype* out,                                         \
      constant float& eps,                                       \
      constant uint& axis_size,                                  \
      constant uint& w_stride,                                   \
      threadgroup float* local_inv_mean [[threadgroup(0)]],      \
      threadgroup float* local_sums [[threadgroup(1)]],          \
      uint gid [[thread_position_in_grid]],                      \
      uint lid [[thread_position_in_threadgroup]],               \
      uint lsize [[threads_per_threadgroup]],                    \
      uint simd_lane_id [[thread_index_in_simdgroup]],           \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);    \
                                                                 \
  template [[host_name("vjp_rms_looped" #name)]] [[kernel]] void \
  vjp_rms_looped<itype>(                                         \
      const device itype* x,                                     \
      const device itype* w,                                     \
      const device itype* g,                                     \
      device itype* gx,                                          \
      device itype* gw,                                          \
      constant float& eps,                                       \
      constant uint& axis_size,                                  \
      constant uint& w_stride,                                   \
      uint gid [[thread_position_in_grid]],                      \
      uint lid [[thread_position_in_threadgroup]],               \
      uint lsize [[threads_per_threadgroup]],                    \
      uint simd_lane_id [[thread_index_in_simdgroup]],           \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_rms(name, itype)      \
  instantiate_rms_single_row(name, itype) \
  instantiate_rms_looped(name, itype)

instantiate_rms(float32, float)
instantiate_rms(float16, half)
instantiate_rms(bfloat16, bfloat16_t)
    // clang-format on
