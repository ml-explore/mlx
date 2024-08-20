// Copyright © 2023-2024 Apple Inc.

#include <metal_math>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/utils.h"
template <typename T, bool traditional, bool forward>
void rope_single_impl(
    const device T* in,
    device T* out,
    constant const int& offset,
    const float inv_freq,
    constant const float& scale,
    constant const size_t& stride,
    uint2 pos,
    uint2 grid) {
  float L = scale * static_cast<float>(offset);

  // Compute costheta, sintheta
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);

  // Compute the input and output indices
  uint in_index_1, in_index_2;
  uint out_index_1, out_index_2;
  if (traditional) {
    out_index_1 = 2 * pos.x + pos.y * stride;
    out_index_2 = out_index_1 + 1;
    in_index_1 = 2 * pos.x + pos.y * stride;
    in_index_2 = in_index_1 + 1;
  } else {
    out_index_1 = pos.x + pos.y * stride;
    out_index_2 = out_index_1 + grid.x;
    in_index_1 = pos.x + pos.y * stride;
    in_index_2 = in_index_1 + grid.x;
  }

  // Read and write the output
  float x1 = static_cast<float>(in[in_index_1]);
  float x2 = static_cast<float>(in[in_index_2]);
  float rx1;
  float rx2;
  if (forward) {
    rx1 = x1 * costheta - x2 * sintheta;
    rx2 = x1 * sintheta + x2 * costheta;
  } else {
    rx1 = x2 * sintheta + x1 * costheta;
    rx2 = x2 * costheta - x1 * sintheta;
  }
  out[out_index_1] = static_cast<T>(rx1);
  out[out_index_2] = static_cast<T>(rx2);
}

template <typename T, bool traditional, bool forward>
[[kernel]] void rope_single(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset,
    constant const float& scale,
    constant const size_t& stride,
    constant const float& base [[buffer(10)]],
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  float inv_freq = metal::exp2(-d * base);
  rope_single_impl<T, traditional, forward>(
      in, out, offset, inv_freq, scale, stride, pos, grid);
}

template <typename T, bool traditional, bool forward>
[[kernel]] void rope_single_freqs(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset,
    constant const float& scale,
    constant const size_t& stride,
    const device float* freqs [[buffer(10)]],
    constant const size_t& freq_stride [[buffer(11)]],
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  float inv_freq = 1.0 / (freqs[freq_stride * pos.x]);
  rope_single_impl<T, traditional, forward>(
      in, out, offset, inv_freq, scale, stride, pos, grid);
}

template <typename T, bool traditional, bool forward, int N = 4>
void rope_impl(
    const device T* in,
    device T* out,
    constant const int& offset,
    const float inv_freq,
    constant const float& scale,
    constant const size_t strides[3],
    constant const size_t out_strides[3],
    constant const size_t& n_batch,
    uint3 pos,
    uint3 grid) {
  float L = scale * static_cast<float>(pos.y + offset);

  // Compute costheta, sintheta
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);

  // Compute the input and output indices
  size_t in_index_1, in_index_2;
  size_t out_index_1, out_index_2;
  if (traditional) {
    out_index_1 = 2 * pos.x * out_strides[2] + pos.y * out_strides[1] +
        N * pos.z * out_strides[0];
    out_index_2 = out_index_1 + 1;
    in_index_1 =
        2 * pos.x * strides[2] + pos.y * strides[1] + N * pos.z * strides[0];
    in_index_2 = in_index_1 + strides[2];
  } else {
    out_index_1 = pos.x * out_strides[2] + pos.y * out_strides[1] +
        N * pos.z * out_strides[0];
    out_index_2 = out_index_1 + grid.x * out_strides[2];
    in_index_1 =
        pos.x * strides[2] + pos.y * strides[1] + N * pos.z * strides[0];
    in_index_2 = in_index_1 + grid.x * strides[2];
  }
  for (int i = 0; i < N && pos.z * N + i < n_batch; ++i) {
    // Read and write the output
    float x1 = static_cast<float>(in[in_index_1]);
    float x2 = static_cast<float>(in[in_index_2]);
    float rx1;
    float rx2;
    if (forward) {
      rx1 = x1 * costheta - x2 * sintheta;
      rx2 = x1 * sintheta + x2 * costheta;
    } else {
      rx1 = x2 * sintheta + x1 * costheta;
      rx2 = x2 * costheta - x1 * sintheta;
    }
    out[out_index_1] = static_cast<T>(rx1);
    out[out_index_2] = static_cast<T>(rx2);
    in_index_1 += strides[0];
    in_index_2 += strides[0];
    out_index_1 += out_strides[0];
    out_index_2 += out_strides[0];
  }
}

template <typename T, bool traditional, bool forward, int N = 4>
[[kernel]] void rope(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset,
    constant const float& scale,
    constant const size_t strides[3],
    constant const size_t out_strides[3],
    constant const size_t& n_batch,
    constant const float& base [[buffer(10)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  float inv_freq = metal::exp2(-d * base);
  rope_impl<T, traditional, forward, N>(
      in,
      out,
      offset,
      inv_freq,
      scale,
      strides,
      out_strides,
      n_batch,
      pos,
      grid);
}

template <typename T, bool traditional, bool forward, int N = 4>
[[kernel]] void rope_freqs(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset,
    constant const float& scale,
    constant const size_t strides[3],
    constant const size_t out_strides[3],
    constant const size_t& n_batch,
    const device float* freqs [[buffer(10)]],
    constant const size_t& freq_stride [[buffer(11)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  float inv_freq = 1.0 / (freqs[freq_stride * pos.x]);
  rope_impl<T, traditional, forward, N>(
      in,
      out,
      offset,
      inv_freq,
      scale,
      strides,
      out_strides,
      n_batch,
      pos,
      grid);
}

// clang-format off
#define instantiate_rope_g(name, type, traditional, forward) \
  template [[host_name("rope_" #name)]] [[kernel]] void      \
  rope<type, traditional, forward>(                          \
      const device type* in [[buffer(0)]],                   \
      device type* out [[buffer(1)]],                        \
      constant const int& offset,                            \
      constant const float& scale,                           \
      constant const size_t strides[3],                      \
      constant const size_t out_strides[3],                  \
      constant const size_t& n_batch,                        \
      constant const float& base [[buffer(10)]],             \
      uint3 pos [[thread_position_in_grid]],                 \
      uint3 grid [[threads_per_grid]]);                      \
  template [[host_name("rope_freqs_" #name)]]                \
  [[kernel]] void rope_freqs<type, traditional, forward>(    \
      const device type* in [[buffer(0)]],                   \
      device type* out [[buffer(1)]],                        \
      constant const int& offset,                            \
      constant const float& scale,                           \
      constant const size_t strides[3],                      \
      constant const size_t out_strides[3],                  \
      constant const size_t& n_batch,                        \
      const device float* freqs [[buffer(10)]],              \
      constant const size_t& freq_stride [[buffer(11)]],     \
      uint3 pos [[thread_position_in_grid]],                 \
      uint3 grid [[threads_per_grid]]);

#define instantiate_rope_s(name, type, traditional, forward)     \
  template [[host_name("rope_single_" #name)]] [[kernel]] void   \
  rope_single<type, traditional, forward>(                       \
      const device type* in [[buffer(0)]],                       \
      device type* out [[buffer(1)]],                            \
      constant const int& offset,                                \
      constant const float& scale,                               \
      constant const size_t& stride,                             \
      constant const float& base [[buffer(10)]],                 \
      uint2 pos [[thread_position_in_grid]],                     \
      uint2 grid [[threads_per_grid]]);                          \
  template [[host_name("rope_single_freqs_" #name)]]             \
  [[kernel]] void rope_single_freqs<type, traditional, forward>( \
      const device type* in [[buffer(0)]],                       \
      device type* out [[buffer(1)]],                            \
      constant const int& offset,                                \
      constant const float& scale,                               \
      constant const size_t& stride,                             \
      const device float* freqs [[buffer(10)]],                  \
      constant const size_t& freq_stride [[buffer(11)]],         \
      uint2 pos [[thread_position_in_grid]],                     \
      uint2 grid [[threads_per_grid]]);

#define instantiate_rope(name, type, traditional, forward) \
  instantiate_rope_s(name, type, traditional, forward)     \
  instantiate_rope_g(name, type, traditional, forward)

instantiate_rope(traditional_float16, half, true, true)
instantiate_rope(traditional_bfloat16, bfloat16_t, true, true)
instantiate_rope(traditional_float32, float, true, true)
instantiate_rope(float16, half, false, true)
instantiate_rope(bfloat16, bfloat16_t, false, true)
instantiate_rope(float32, float, false, true)
instantiate_rope(vjp_traditional_float16, half, true, false)
instantiate_rope(vjp_traditional_bfloat16, bfloat16_t, true, false)
instantiate_rope(vjp_traditional_float32, float, true, false)
instantiate_rope(vjp_float16, half, false, false)
instantiate_rope(vjp_bfloat16, bfloat16_t, false, false)
instantiate_rope(vjp_float32, float, false, false) // clang-format on
