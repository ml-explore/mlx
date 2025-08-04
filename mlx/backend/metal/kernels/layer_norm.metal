// Copyright © 2024 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

constant bool has_w [[function_constant(20)]];

template <int N = 1>
inline void initialize_buffer(
    threadgroup float* xs,
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  if (simd_group_id == 0) {
    for (int i = 0; i < N; i++) {
      xs[N * simd_lane_id + i] = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

template <int N = 1>
inline void threadgroup_sum(
    thread float* x,
    threadgroup float* xs,
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  for (int i = 0; i < N; i++) {
    x[i] = simd_sum(x[i]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane_id == 0) {
    for (int i = 0; i < N; i++) {
      xs[N * simd_group_id + i] = x[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < N; i++) {
    x[i] = xs[N * simd_lane_id + i];
    x[i] = simd_sum(x[i]);
  }
}

template <typename T, int N_READS = 8>
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
  constexpr int SIMD_SIZE = 32;

  // Initialize the registers and threadgroup memory
  float thread_x[N_READS] = {0};
  threadgroup float local_buffer[SIMD_SIZE] = {0};
  initialize_buffer(local_buffer, simd_lane_id, simd_group_id);

  // Advance the pointers
  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  b += b_stride * lid * N_READS;
  out += gid * size_t(axis_size) + lid * N_READS;

  // Compute some variables for reading writing etc
  const bool safe = lid * N_READS + N_READS <= axis_size;
  const int n = axis_size - lid * N_READS;

  // Read the inputs
  if (safe) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      thread_x[i] = x[i];
    }
  }

  // Compute the mean
  float mean = 0;
  for (int i = 0; i < N_READS; i++) {
    mean += thread_x[i];
  }
  threadgroup_sum(&mean, local_buffer, simd_lane_id, simd_group_id);
  mean /= axis_size;

  // Compute the normalizer
  float normalizer = 0;
  if (!safe) {
    for (int i = n; i < N_READS; i++) {
      thread_x[i] = mean;
    }
  }
  for (int i = 0; i < N_READS; i++) {
    thread_x[i] -= mean;
    normalizer += thread_x[i] * thread_x[i];
  }
  threadgroup_sum(&normalizer, local_buffer, simd_lane_id, simd_group_id);
  normalizer = metal::precise::rsqrt(normalizer / axis_size + eps);

  // Write the outputs
  if (safe) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] *= normalizer;
      out[i] = w[w_stride * i] * static_cast<T>(thread_x[i]) + b[b_stride * i];
    }
  } else {
    for (int i = 0; i < n; i++) {
      thread_x[i] *= normalizer;
      out[i] = w[w_stride * i] * static_cast<T>(thread_x[i]) + b[b_stride * i];
    }
  }
}

template <typename T, int N_READS = 4>
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
  constexpr int SIMD_SIZE = 32;

  threadgroup float local_buffer[SIMD_SIZE];
  initialize_buffer(local_buffer, simd_lane_id, simd_group_id);

  x += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;
  b += b_stride * lid * N_READS;

  // Compute the mean
  float mean = 0;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        mean += x[i + r];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          mean += x[i + r];
        }
      }
    }
  }
  threadgroup_sum(&mean, local_buffer, simd_lane_id, simd_group_id);
  mean /= axis_size;

  // Compute the normalizer
  float normalizer = 0;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float t = x[i + r] - mean;
        normalizer += t * t;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float t = x[i + r] - mean;
          normalizer += t * t;
        }
      }
    }
  }
  threadgroup_sum(&normalizer, local_buffer, simd_lane_id, simd_group_id);
  normalizer = metal::precise::rsqrt(normalizer / axis_size + eps);

  // Write the outputs
  out += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = (x[r + i] - mean) * normalizer;
        out[r + i] =
            w[w_stride * (i + r)] * static_cast<T>(xi) + b[b_stride * (i + r)];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = (x[r + i] - mean) * normalizer;
          out[r + i] = w[w_stride * (i + r)] * static_cast<T>(xi) +
              b[b_stride * (i + r)];
        }
      }
    }
  }
}

template <typename T, int N_READS = 8>
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
  constexpr int SIMD_SIZE = 32;

  // Advance the input pointers
  x += gid * size_t(axis_size) + lid * N_READS;
  g += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;

  // Initialize the registers and threadgroup memory
  float thread_x[N_READS] = {0};
  float thread_w[N_READS] = {0};
  float thread_g[N_READS] = {0};
  threadgroup float local_buffer[3 * SIMD_SIZE];
  initialize_buffer<3>(local_buffer, simd_lane_id, simd_group_id);

  // Compute some variables for reading writing etc
  const bool safe = lid * N_READS + N_READS <= axis_size;
  const int n = axis_size - lid * N_READS;

  // Read the inputs
  if (safe) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] = x[i];
      thread_g[i] = g[i];
      thread_w[i] = w[i * w_stride];
    }
  } else {
    for (int i = 0; i < n; i++) {
      thread_x[i] = x[i];
      thread_g[i] = g[i];
      thread_w[i] = w[i * w_stride];
    }
  }

  // Compute the mean
  float mean = 0;
  for (int i = 0; i < N_READS; i++) {
    mean += thread_x[i];
  }
  threadgroup_sum(&mean, local_buffer, simd_lane_id, simd_group_id);
  mean /= axis_size;

  // Compute the neccesary scaling factors using the mean
  if (!safe) {
    for (int i = n; i < N_READS; i++) {
      thread_x[i] = mean;
    }
  }
  float factors[3] = {0};
  constexpr int meanwg = 0;
  constexpr int meanwgxc = 1;
  constexpr int normalizer2 = 2;
  for (int i = 0; i < N_READS; i++) {
    thread_x[i] -= mean;
    factors[meanwg] += thread_w[i] * thread_g[i];
    factors[meanwgxc] += thread_w[i] * thread_g[i] * thread_x[i];
    factors[normalizer2] += thread_x[i] * thread_x[i];
  }
  threadgroup_sum<3>(factors, local_buffer, simd_lane_id, simd_group_id);
  factors[meanwg] /= axis_size;
  factors[meanwgxc] /= axis_size;
  factors[normalizer2] = 1 / (factors[normalizer2] / axis_size + eps);
  float normalizer = metal::precise::sqrt(factors[normalizer2]);

  // Write the outputs
  gx += gid * size_t(axis_size) + lid * N_READS;
  gw += gid * size_t(axis_size) + lid * N_READS;
  if (safe) {
    for (int i = 0; i < N_READS; i++) {
      thread_x[i] *= normalizer;
      gx[i] = static_cast<T>(
          normalizer * (thread_w[i] * thread_g[i] - factors[meanwg]) -
          thread_x[i] * factors[meanwgxc] * factors[normalizer2]);
      if (has_w) {
        gw[i] = static_cast<T>(thread_g[i] * thread_x[i]);
      }
    }
  } else {
    for (int i = 0; i < n; i++) {
      thread_x[i] *= normalizer;
      gx[i] = static_cast<T>(
          normalizer * (thread_w[i] * thread_g[i] - factors[meanwg]) -
          thread_x[i] * factors[meanwgxc] * factors[normalizer2]);
      if (has_w) {
        gw[i] = static_cast<T>(thread_g[i] * thread_x[i]);
      }
    }
  }
}

template <typename T, int N_READS = 4>
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
  constexpr int SIMD_SIZE = 32;

  // Advance the input pointers
  x += gid * size_t(axis_size) + lid * N_READS;
  g += gid * size_t(axis_size) + lid * N_READS;
  w += w_stride * lid * N_READS;

  threadgroup float local_buffer[3 * SIMD_SIZE];
  initialize_buffer<3>(local_buffer, simd_lane_id, simd_group_id);

  // Compute the mean
  float mean = 0;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        mean += x[i + r];
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          mean += x[i + r];
        }
      }
    }
  }
  threadgroup_sum(&mean, local_buffer, simd_lane_id, simd_group_id);
  mean /= axis_size;

  // Compute the neccesary scaling factors using the mean
  float factors[3] = {0};
  constexpr int meanwg = 0;
  constexpr int meanwgxc = 1;
  constexpr int normalizer2 = 2;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float t = x[i + r] - mean;
        float wi = w[(i + r) * w_stride];
        float gi = g[i + r];
        float wg = wi * gi;
        factors[meanwg] += wg;
        factors[meanwgxc] += wg * t;
        factors[normalizer2] += t * t;
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float t = x[i + r] - mean;
          float wi = w[(i + r) * w_stride];
          float gi = g[i + r];
          float wg = wi * gi;
          factors[meanwg] += wg;
          factors[meanwgxc] += wg * t;
          factors[normalizer2] += t * t;
        }
      }
    }
  }
  threadgroup_sum<3>(factors, local_buffer, simd_lane_id, simd_group_id);
  factors[meanwg] /= axis_size;
  factors[meanwgxc] /= axis_size;
  factors[normalizer2] = 1 / (factors[normalizer2] / axis_size + eps);
  float normalizer = metal::precise::sqrt(factors[normalizer2]);

  // Write the outputs
  gx += gid * size_t(axis_size) + lid * N_READS;
  gw += gid * size_t(axis_size) + lid * N_READS;
  for (uint r = 0; r < axis_size; r += lsize * N_READS) {
    if (r + lid * N_READS + N_READS <= axis_size) {
      for (int i = 0; i < N_READS; i++) {
        float xi = (x[i + r] - mean) * normalizer;
        float wi = w[(i + r) * w_stride];
        float gi = g[i + r];
        gx[i + r] = static_cast<T>(
            normalizer * (wi * gi - factors[meanwg]) -
            xi * factors[meanwgxc] * factors[normalizer2]);
        if (has_w) {
          gw[i + r] = static_cast<T>(gi * xi);
        }
      }
    } else {
      for (int i = 0; i < N_READS; i++) {
        if ((r + lid * N_READS + i) < axis_size) {
          float xi = (x[i + r] - mean) * normalizer;
          float wi = w[(i + r) * w_stride];
          float gi = g[i + r];
          gx[i + r] = static_cast<T>(
              normalizer * (wi * gi - factors[meanwg]) -
              xi * factors[meanwgxc] * factors[normalizer2]);
          if (has_w) {
            gw[i + r] = static_cast<T>(gi * xi);
          }
        }
      }
    }
  }
}

// clang-format off
#define instantiate_layer_norm(name, itype)                                       \
  instantiate_kernel("layer_norm" #name, layer_norm_single_row, itype)            \
  instantiate_kernel("vjp_layer_norm" #name, vjp_layer_norm_single_row, itype)    \
  instantiate_kernel("layer_norm_looped" #name, layer_norm_looped, itype)         \
  instantiate_kernel("vjp_layer_norm_looped" #name, vjp_layer_norm_looped, itype)

instantiate_layer_norm(float32, float)
instantiate_layer_norm(float16, half)
instantiate_layer_norm(bfloat16, bfloat16_t) // clang-format on
