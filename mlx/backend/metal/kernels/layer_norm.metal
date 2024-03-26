// Copyright © 2024 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void layer_norm_single_row(
    const device T* x,
    const device T* w,
    const device T* b,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    constant uint& b_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  float sumx = 0;
  float sumx2 = 0;
  float thread_x[N_READS];

  constexpr int SIMD_SIZE = 32;

  threadgroup float local_sumx[SIMD_SIZE];
  threadgroup float local_sumx2[SIMD_SIZE];
  threadgroup float local_mean[1];
  threadgroup float local_normalizer[1];

  x += gid * axis_size + lid * N_READS;
  w += w_stride * lid * N_READS;
  b += b_stride * lid * N_READS;

  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
      sumx2 += thread_x[i] * thread_x[i];
      sumx += thread_x[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        thread_x[i] = x[i];
        sumx2 += thread_x[i] * thread_x[i];
        sumx += thread_x[i];
      }
    }
  }

  sumx = simd_sum(sumx);
  sumx2 = simd_sum(sumx2);

  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sumx[simd_lane_id] = 0;
    local_sumx2[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sumx[simd_group_id] = sumx;
    local_sumx2[simd_group_id] = sumx2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    sumx = simd_sum(local_sumx[simd_lane_id]);
    sumx2 = simd_sum(local_sumx2[simd_lane_id]);
    if (simd_lane_id == 0) {
      float mean = sumx / axis_size;
      float variance = sumx2 / axis_size - mean * mean;

      local_mean[0] = mean;
      local_normalizer[0] = metal::precise::rsqrt(variance + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float mean = local_mean[0];
  float normalizer = local_normalizer[0];

  // Write the outputs
  out += gid * axis_size + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = (thread_x[i] - mean) * normalizer;
      out[i] = w[w_stride * i] * static_cast<T>(thread_x[i]) + b[b_stride * i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        thread_x[i] = (thread_x[i] - mean) * normalizer;
        out[i] = w[w_stride * i] * static_cast<T>(thread_x[i]) + b[b_stride * i];
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void layer_norm_looped(
    const device T* x,
    const device T* w,
    const device T* b,
    device T* out,
    constant float& eps,
    constant uint& axis_size,
    constant uint& w_stride,
    constant uint& b_stride,
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  float sumx = 0;
  float sumx2 = 0;

  constexpr int SIMD_SIZE = 32;

  threadgroup float local_sumx[SIMD_SIZE];
  threadgroup float local_sumx2[SIMD_SIZE];
  threadgroup float local_mean[1];
  threadgroup float local_normalizer[1];

  x += gid * axis_size + lid * N_READS;
  w += w_stride * lid * N_READS;
  b += b_stride * lid * N_READS;

  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        sumx2 += xi * xi;
        sumx += xi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          sumx2 += xi * xi;
          sumx += xi;
        }
      }
    }
  }

  sumx = simd_sum(sumx);
  sumx2 = simd_sum(sumx2);

  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sumx[simd_lane_id] = 0;
    local_sumx2[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sumx[simd_group_id] = sumx;
    local_sumx2[simd_group_id] = sumx2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    sumx = simd_sum(local_sumx[simd_lane_id]);
    sumx2 = simd_sum(local_sumx2[simd_lane_id]);
    if (simd_lane_id == 0) {
      float mean = sumx / axis_size;
      float variance = sumx2 / axis_size - mean * mean;

      local_mean[0] = mean;
      local_normalizer[0] = metal::precise::rsqrt(variance + eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float mean = local_mean[0];
  float normalizer = local_normalizer[0];

  // Write the outputs
  out += gid * axis_size + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = (x[r + i] - mean) * normalizer;
        out[r + i] = w[w_stride * (i + r)] * static_cast<T>(xi) + b[b_stride * (i + r)];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = (x[r + i] - mean) * normalizer;
          out[r + i] = w[w_stride * (i + r)] * static_cast<T>(xi) + b[b_stride * (i + r)];
        }
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void vjp_layer_norm_single_row(
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
  float sumx = 0;
  float sumx2 = 0;
  float sumwg = 0;
  float sumwgx = 0;

  constexpr int SIMD_SIZE = 32;

  threadgroup float local_sumx[SIMD_SIZE];
  threadgroup float local_sumx2[SIMD_SIZE];
  threadgroup float local_sumwg[SIMD_SIZE];
  threadgroup float local_sumwgx[SIMD_SIZE];
  threadgroup float local_mean[1];
  threadgroup float local_normalizer[1];
  threadgroup float local_meanwg[1];
  threadgroup float local_meanwgx[1];

  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
      thread_w[i] = w[i * w_stride];
      thread_g[i] = g[i];
      float wg = thread_w[i] * thread_g[i];
      sumx += thread_x[i];
      sumx2 += thread_x[i] * thread_x[i];
      sumwg += wg;
      sumwgx += wg * thread_x[i];
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        thread_x[i] = x[i];
        thread_w[i] = w[i * w_stride];
        thread_g[i] = g[i];
        float wg = thread_w[i] * thread_g[i];
        sumx += thread_x[i];
        sumx2 += thread_x[i] * thread_x[i];
        sumwg += wg;
        sumwgx += wg * thread_x[i];
      }
    }
  }

  sumx = simd_sum(sumx);
  sumx2 = simd_sum(sumx2);
  sumwg = simd_sum(sumwg);
  sumwgx = simd_sum(sumwgx);

  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sumx[simd_lane_id] = 0;
    local_sumx2[simd_lane_id] = 0;
    local_sumwg[simd_lane_id] = 0;
    local_sumwgx[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sumx[simd_group_id] = sumx;
    local_sumx2[simd_group_id] = sumx2;
    local_sumwg[simd_group_id] = sumwg;
    local_sumwgx[simd_group_id] = sumwgx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    sumx = simd_sum(local_sumx[simd_lane_id]);
    sumx2 = simd_sum(local_sumx2[simd_lane_id]);
    sumwg = simd_sum(local_sumwg[simd_lane_id]);
    sumwgx = simd_sum(local_sumwgx[simd_lane_id]);
    if (simd_lane_id == 0) {
      float mean = sumx / axis_size;
      float variance = sumx2 / axis_size - mean * mean;

      local_mean[0] = mean;
      local_normalizer[0] = metal::precise::rsqrt(variance + eps);
      local_meanwg[0] = sumwg / axis_size;
      local_meanwgx[0] = sumwgx / axis_size;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float mean = local_mean[0];
  float normalizer = local_normalizer[0];
  float meanwg = local_meanwg[0];
  float meanwgxc = local_meanwgx[0] - meanwg * mean;
  float normalizer2 = normalizer * normalizer;

  // Write the outputs
  gx += gid * axis_size + lid * N_READS;
  gw += gid * axis_size + lid * N_READS;
  if (lid * N_READS + N_READS <= axis_size) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = (thread_x[i] - mean) * normalizer;
      gx[i] = static_cast<T>(normalizer * (thread_w[i] * thread_g[i] - meanwg) -
                             thread_x[i] * meanwgxc * normalizer2);
      gw[i] = static_cast<T>(thread_g[i] * thread_x[i]);
    }
  } else {
    for (int i = 0; i < N_READS; i++) {
      if ((lid * N_READS + i) < axis_size) {
        thread_x[i] = (thread_x[i] - mean) * normalizer;
        gx[i] = static_cast<T>(normalizer * (thread_w[i] * thread_g[i] - meanwg) -
                               thread_x[i] * meanwgxc * normalizer2);
        gw[i] = static_cast<T>(thread_g[i] * thread_x[i]);
      }
    }
  }
}

template <typename T, int N_READS = RMS_N_READS>
[[kernel]] void vjp_layer_norm_looped(
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
  float sumx = 0;
  float sumx2 = 0;
  float sumwg = 0;
  float sumwgx = 0;

  constexpr int SIMD_SIZE = 32;

  threadgroup float local_sumx[SIMD_SIZE];
  threadgroup float local_sumx2[SIMD_SIZE];
  threadgroup float local_sumwg[SIMD_SIZE];
  threadgroup float local_sumwgx[SIMD_SIZE];
  threadgroup float local_mean[1];
  threadgroup float local_normalizer[1];
  threadgroup float local_meanwg[1];
  threadgroup float local_meanwgx[1];

  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = x[i + r];
        float wi = w[(i + r) * w_stride];
        float gi = g[i + r];
        float wg = wi * gi;
        sumx += xi;
        sumx2 += xi * xi;
        sumwg += wg;
        sumwgx += wg * xi;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = x[i + r];
          float wi = w[(i + r) * w_stride];
          float gi = g[i + r];
          float wg = wi * gi;
          sumx += xi;
          sumx2 += xi * xi;
          sumwg += wg;
          sumwgx += wg * xi;
        }
      }
    }
  }

  sumx = simd_sum(sumx);
  sumx2 = simd_sum(sumx2);
  sumwg = simd_sum(sumwg);
  sumwgx = simd_sum(sumwgx);

  //  Initialize shared memory
  if (simd_group_id == 0) {
    local_sumx[simd_lane_id] = 0;
    local_sumx2[simd_lane_id] = 0;
    local_sumwg[simd_lane_id] = 0;
    local_sumwgx[simd_lane_id] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Write simd accumulations into shared memory
  if (simd_lane_id == 0) {
    local_sumx[simd_group_id] = sumx;
    local_sumx2[simd_group_id] = sumx2;
    local_sumwg[simd_group_id] = sumwg;
    local_sumwgx[simd_group_id] = sumwgx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Accumulate over simd groups
  if (simd_group_id == 0) {
    sumx = simd_sum(local_sumx[simd_lane_id]);
    sumx2 = simd_sum(local_sumx2[simd_lane_id]);
    sumwg = simd_sum(local_sumwg[simd_lane_id]);
    sumwgx = simd_sum(local_sumwgx[simd_lane_id]);
    if (simd_lane_id == 0) {
      float mean = sumx / axis_size;
      float variance = sumx2 / axis_size - mean * mean;

      local_mean[0] = mean;
      local_normalizer[0] = metal::precise::rsqrt(variance + eps);
      local_meanwg[0] = sumwg / axis_size;
      local_meanwgx[0] = sumwgx / axis_size;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float mean = local_mean[0];
  float normalizer = local_normalizer[0];
  float meanwg = local_meanwg[0];
  float meanwgxc = local_meanwgx[0] - meanwg * mean;
  float normalizer2 = normalizer * normalizer;

  // Write the outputs
  gx += gid * axis_size + lid * N_READS;
  gw += gid * axis_size + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = (x[i + r] - mean) * normalizer;
        float wi = w[(i + r) * w_stride];
        float gi = g[i + r];
        gx[i + r] = static_cast<T>(normalizer * (wi * gi - meanwg) -
                                   xi * meanwgxc * normalizer2);
        gw[i + r] = static_cast<T>(gi * xi);
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = (x[i + r] - mean) * normalizer;
          float wi = w[(i + r) * w_stride];
          float gi = g[i + r];
          gx[i + r] = static_cast<T>(normalizer * (wi * gi - meanwg) -
                                     xi * meanwgxc * normalizer2);
          gw[i + r] = static_cast<T>(gi * xi);
        }
      }
    }
  }
}

// clang-format off
#define instantiate_layer_norm_single_row(name, itype)            \
  template [[host_name("layer_norm" #name)]] [[kernel]] void      \
  layer_norm_single_row<itype>(                                   \
      const device itype* x,                                      \
      const device itype* w,                                      \
      const device itype* b,                                      \
      device itype* out,                                          \
      constant float& eps,                                        \
      constant uint& axis_size,                                   \
      constant uint& w_stride,                                    \
      constant uint& b_stride,                                    \
      uint gid [[thread_position_in_grid]],                       \
      uint lid [[thread_position_in_threadgroup]],                \
      uint simd_lane_id [[thread_index_in_simdgroup]],            \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);     \
  template [[host_name("vjp_layer_norm" #name)]] [[kernel]] void  \
  vjp_layer_norm_single_row<itype>(                               \
      const device itype* x,                                      \
      const device itype* w,                                      \
      const device itype* g,                                      \
      device itype* gx,                                           \
      device itype* gw,                                           \
      constant float& eps,                                        \
      constant uint& axis_size,                                   \
      constant uint& w_stride,                                    \
      uint gid [[thread_position_in_grid]],                       \
      uint lid [[thread_position_in_threadgroup]],                \
      uint simd_lane_id [[thread_index_in_simdgroup]],            \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_layer_norm_looped(name, itype)                       \
  template [[host_name("layer_norm_looped" #name)]] [[kernel]] void      \
  layer_norm_looped<itype>(                                              \
      const device itype* x,                                             \
      const device itype* w,                                             \
      const device itype* b,                                             \
      device itype* out,                                                 \
      constant float& eps,                                               \
      constant uint& axis_size,                                          \
      constant uint& w_stride,                                           \
      constant uint& b_stride,                                           \
      uint gid [[thread_position_in_grid]],                              \
      uint lid [[thread_position_in_threadgroup]],                       \
      uint lsize [[threads_per_threadgroup]],                            \
      uint simd_lane_id [[thread_index_in_simdgroup]],                   \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);            \
  template [[host_name("vjp_layer_norm_looped" #name)]] [[kernel]] void  \
  vjp_layer_norm_looped<itype>(                                          \
      const device itype* x,                                             \
      const device itype* w,                                             \
      const device itype* g,                                             \
      device itype* gx,                                                  \
      device itype* gb,                                                  \
      constant float& eps,                                               \
      constant uint& axis_size,                                          \
      constant uint& w_stride,                                           \
      uint gid [[thread_position_in_grid]],                              \
      uint lid [[thread_position_in_threadgroup]],                       \
      uint lsize [[threads_per_threadgroup]],                            \
      uint simd_lane_id [[thread_index_in_simdgroup]],                   \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_layer_norm(name, itype)      \
  instantiate_layer_norm_single_row(name, itype) \
  instantiate_layer_norm_looped(name, itype)

instantiate_layer_norm(float32, float)
instantiate_layer_norm(float16, half)
instantiate_layer_norm(bfloat16, bfloat16_t)
    // clang-format on

