// Copyright © 2023-2024 Apple Inc.

#include <metal_math>

#include "mlx/backend/metal/kernels/utils.h"

constant bool forward [[function_constant(1)]];
constant bool traditional [[function_constant(2)]];
constant bool hs_transpose [[function_constant(3)]];

template <typename T>
void rope_single_impl(
    const device T* in,
    device T* out,
    constant const int& offset,
    const float inv_freq,
    constant const float& scale,
    constant const int64_t& stride,
    uint2 pos,
    uint2 grid) {
  float L = scale * static_cast<float>(offset);

  // Compute costheta, sintheta
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);

  // Compute the input and output indices
  uint index_1, index_2;
  if (traditional) {
    index_1 = 2 * pos.x + pos.y * stride;
    index_2 = index_1 + 1;
  } else {
    index_1 = pos.x + pos.y * stride;
    index_2 = index_1 + grid.x;
  }

  // Read and write the output
  float x1 = static_cast<float>(in[index_1]);
  float x2 = static_cast<float>(in[index_2]);
  float rx1;
  float rx2;
  if (forward) {
    rx1 = x1 * costheta - x2 * sintheta;
    rx2 = x1 * sintheta + x2 * costheta;
  } else {
    rx1 = x2 * sintheta + x1 * costheta;
    rx2 = x2 * costheta - x1 * sintheta;
  }
  out[index_1] = static_cast<T>(rx1);
  out[index_2] = static_cast<T>(rx2);
}

template <typename T>
[[kernel]] void rope_single(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset,
    constant const float& scale,
    constant const int64_t& stride,
    constant const float& base [[buffer(10)]],
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  float inv_freq = metal::exp2(-d * base);
  rope_single_impl<T>(in, out, offset, inv_freq, scale, stride, pos, grid);
}

template <typename T>
[[kernel]] void rope_single_freqs(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const int& offset,
    constant const float& scale,
    constant const int64_t& stride,
    const device float* freqs [[buffer(10)]],
    constant const int64_t& freq_stride [[buffer(11)]],
    uint2 pos [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  float inv_freq = 1.0 / (freqs[freq_stride * pos.x]);
  rope_single_impl<T>(in, out, offset, inv_freq, scale, stride, pos, grid);
}

template <typename T, typename IdxT, int N = 4>
void rope_impl(
    const device T* in,
    device T* out,
    const device int* offset,
    const float inv_freq,
    constant const float& scale,
    constant const int64_t strides[3],
    constant const int64_t out_strides[3],
    constant const int64_t& offset_stride,
    constant const int& n_head,
    uint3 pos,
    uint3 grid) {
  auto n_head_up = N * ((n_head + N - 1) / N);
  auto head_idx = static_cast<int>((pos.z * N) % n_head_up);
  auto batch_idx = (pos.z * N) / n_head_up;
  auto batch_offset = offset[batch_idx * offset_stride];
  float L = scale * static_cast<float>(pos.y + batch_offset);
  auto mat_idx = batch_idx * n_head + head_idx;

  // Compute costheta, sintheta
  float theta = L * inv_freq;
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);
  // Compute the input and output indices
  IdxT in_index_1;
  if (hs_transpose) {
    IdxT batch_stride = grid.y * IdxT(strides[1]);
    in_index_1 =
        batch_idx * batch_stride + pos.y * strides[1] + head_idx * strides[0];
  } else {
    in_index_1 = pos.y * IdxT(strides[1]) + mat_idx * IdxT(strides[0]);
  }
  IdxT in_index_2;
  IdxT out_index_1 =
      pos.y * IdxT(out_strides[1]) + mat_idx * IdxT(out_strides[0]);
  IdxT out_index_2;
  if (traditional) {
    out_index_1 += 2 * pos.x * IdxT(out_strides[2]);
    out_index_2 = out_index_1 + 1;
    in_index_1 += 2 * pos.x * IdxT(strides[2]);
    in_index_2 = in_index_1 + IdxT(strides[2]);
  } else {
    out_index_1 += pos.x * IdxT(out_strides[2]);
    out_index_2 = out_index_1 + grid.x * IdxT(out_strides[2]);
    in_index_1 += pos.x * IdxT(strides[2]);
    in_index_2 = in_index_1 + grid.x * IdxT(strides[2]);
  }
  for (int i = 0; i < N && head_idx + i < n_head; ++i) {
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
    in_index_1 += IdxT(strides[0]);
    in_index_2 += IdxT(strides[0]);
    out_index_1 += IdxT(out_strides[0]);
    out_index_2 += IdxT(out_strides[0]);
  }
}

template <typename T, typename IdxT, int N = 4>
[[kernel]] void rope(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const device int* offset,
    constant const float& scale,
    constant const int64_t strides[3],
    constant const int64_t out_strides[3],
    constant const int64_t& offset_stride,
    constant const int& n_head,
    constant const float& base [[buffer(10)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);
  float inv_freq = metal::exp2(-d * base);
  rope_impl<T, IdxT, N>(
      in,
      out,
      offset,
      inv_freq,
      scale,
      strides,
      out_strides,
      offset_stride,
      n_head,
      pos,
      grid);
}

template <typename T, typename IdxT, int N = 4>
[[kernel]] void rope_freqs(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const device int* offset,
    constant const float& scale,
    constant const int64_t strides[3],
    constant const int64_t out_strides[3],
    constant const int64_t& offset_stride,
    constant const int& n_head,
    const device float* freqs [[buffer(10)]],
    constant const int64_t& freq_stride [[buffer(11)]],
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  float inv_freq = 1.0 / (freqs[freq_stride * pos.x]);
  rope_impl<T, IdxT, N>(
      in,
      out,
      offset,
      inv_freq,
      scale,
      strides,
      out_strides,
      offset_stride,
      n_head,
      pos,
      grid);
}

// Fused RoPE(K) + key cache append for the single-token decode case.
//
// Rotates the key pair of each frequency index exactly like rope_single and
// writes it directly into the key cache at position `offset`. Thread (x, y)
// handles one rotated pair (x < dims/2) or a pass-through pair of the
// unrotated tail (dims/2 <= x < k_dims/2) for head row y.
template <typename T, typename IdxT>
void rope_append_impl(
    const device T* k_in,
    device T* k_cache,
    const float inv_freq,
    constant const float& scale,
    constant const int& offset,
    constant const int& dims,
    constant const int& k_dims,
    constant const int64_t& k_cache_mat_stride,
    uint2 pos) {
  int dh = dims / 2;
  int kh = k_dims / 2;
  IdxT m = static_cast<IdxT>(pos.y);
  IdxT off = static_cast<IdxT>(offset);

  if (pos.x < static_cast<uint>(dh)) {
    // Same arithmetic as rope_single (forward direction)
    float L = scale * static_cast<float>(offset);
    float theta = L * inv_freq;
    float costheta = metal::fast::cos(theta);
    float sintheta = metal::fast::sin(theta);
    IdxT i1, i2;
    if (traditional) {
      i1 = 2 * pos.x;
      i2 = i1 + 1;
    } else {
      i1 = pos.x;
      i2 = pos.x + dh;
    }
    IdxT kr = m * static_cast<IdxT>(k_dims);
    IdxT cw = m * static_cast<IdxT>(k_cache_mat_stride) +
        off * static_cast<IdxT>(k_dims);
    float x1 = static_cast<float>(k_in[kr + i1]);
    float x2 = static_cast<float>(k_in[kr + i2]);
    k_cache[cw + i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
    k_cache[cw + i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
  } else if (pos.x < static_cast<uint>(kh)) {
    // Unrotated tail [dims, k_dims) is copied unchanged
    int64_t j = dims + 2 * (static_cast<int>(pos.x) - dh);
    IdxT kr = m * static_cast<IdxT>(k_dims);
    IdxT cw = m * static_cast<IdxT>(k_cache_mat_stride) +
        off * static_cast<IdxT>(k_dims);
    if (j < k_dims) {
      k_cache[cw + j] = k_in[kr + j];
      if (j + 1 < k_dims) {
        k_cache[cw + j + 1] = k_in[kr + j + 1];
      }
    }
  }
}

template <typename T, typename IdxT>
[[kernel]] void rope_append(
    const device T* k_in [[buffer(0)]],
    device T* k_cache [[buffer(1)]],
    constant const int& offset [[buffer(2)]],
    constant const float& scale [[buffer(3)]],
    constant const int& dims [[buffer(4)]],
    constant const int& k_dims [[buffer(5)]],
    constant const int64_t& k_cache_mat_stride [[buffer(6)]],
    constant const float& base [[buffer(7)]],
    uint2 pos [[thread_position_in_grid]]) {
  float inv_freq = 0.0f;
  if (pos.x < static_cast<uint>(dims / 2)) {
    float d = static_cast<float>(pos.x) / static_cast<float>(dims / 2);
    inv_freq = metal::exp2(-d * base);
  }
  rope_append_impl<T, IdxT>(
      k_in,
      k_cache,
      inv_freq,
      scale,
      offset,
      dims,
      k_dims,
      k_cache_mat_stride,
      pos);
}

template <typename T, typename IdxT>
[[kernel]] void rope_append_freqs(
    const device T* k_in [[buffer(0)]],
    device T* k_cache [[buffer(1)]],
    constant const int& offset [[buffer(2)]],
    constant const float& scale [[buffer(3)]],
    constant const int& dims [[buffer(4)]],
    constant const int& k_dims [[buffer(5)]],
    constant const int64_t& k_cache_mat_stride [[buffer(6)]],
    const device float* freqs [[buffer(7)]],
    constant const int64_t& freq_stride [[buffer(8)]],
    uint2 pos [[thread_position_in_grid]]) {
  float inv_freq = 0.0f;
  if (pos.x < static_cast<uint>(dims / 2)) {
    inv_freq = 1.0 / (freqs[freq_stride * pos.x]);
  }
  rope_append_impl<T, IdxT>(
      k_in,
      k_cache,
      inv_freq,
      scale,
      offset,
      dims,
      k_dims,
      k_cache_mat_stride,
      pos);
}

// clang-format off
#define instantiate_rope_g(name, type) \
  instantiate_kernel("rope_" #name, rope, type, int32_t) \
  instantiate_kernel("rope_freqs_" #name, rope_freqs, type, int32_t) \
  instantiate_kernel("rope_large_" #name, rope, type, int64_t) \
  instantiate_kernel("rope_freqs_large_" #name, rope_freqs, type, int64_t)

#define instantiate_rope_append(name, type) \
  instantiate_kernel("rope_append_" #name, rope_append, type, int32_t) \
  instantiate_kernel( \
      "rope_append_freqs_" #name, rope_append_freqs, type, int32_t) \
  instantiate_kernel("rope_append_large_" #name, rope_append, type, int64_t) \
  instantiate_kernel( \
      "rope_append_freqs_large_" #name, rope_append_freqs, type, int64_t)

#define instantiate_rope_s(name, type) \
  instantiate_kernel("rope_single_" #name, rope_single, type) \
  instantiate_kernel("rope_single_freqs_" #name, rope_single_freqs, type)

#define instantiate_rope(name, type) \
  instantiate_rope_s(name, type)     \
  instantiate_rope_g(name, type)

instantiate_rope(float16, half)
instantiate_rope(bfloat16, bfloat16_t)
instantiate_rope(float32, float)

instantiate_rope_append(float16, half)
instantiate_rope_append(bfloat16, bfloat16_t)
instantiate_rope_append(float32, float) // clang-format on
