// Copyright © 2026 Apple Inc.

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

template <typename T, const int group_size, const int bits>
[[kernel]] void ternary_quantize(
    const device T* w [[buffer(0)]],
    device uint32_t* out [[buffer(1)]],
    device T* scales [[buffer(2)]],
    uint2 index [[thread_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint2 grid_dim [[threads_per_grid]]) {
  // Quantize {-1,0,1} to 2-bit codes q = round(w/scale)+1 with per-group
  // scale=max|w|.
  static_assert(
      bits == 2, "Ternary quantization Metal kernel only supports 2 bits.");

  constexpr float eps = 1e-7f;
  constexpr int simd_size = 32;
  constexpr int elements_per_uint = 32 / bits; // values packed into one uint32

  constexpr int values_per_reduce = group_size / simd_size;
  constexpr int threads_per_pack = elements_per_uint / values_per_reduce;

  static_assert(threads_per_pack > 0, "Threads per pack must be positive.");
  static_assert(
      (threads_per_pack & (threads_per_pack - 1)) == 0,
      "Threads per pack must be power of 2.");
  static_assert(
      group_size % simd_size == 0,
      "Group size must be divisible by simd size.");
  static_assert(threads_per_pack > 0, "Threads per pack must be positive.");

  const size_t offset = index.x + grid_dim.x * size_t(index.y);
  const size_t in_index = offset * values_per_reduce;

  float w_thread[values_per_reduce];
  float w_max = 0.0f;

#pragma clang loop unroll(full)
  for (int i = 0; i < values_per_reduce; i++) {
    const float val = static_cast<float>(w[in_index + i]);
    w_thread[i] = val;
    w_max = max(w_max, abs(val));
  }

  // Group max and scale
  w_max = simd_max(w_max);
  const float scale = max(w_max, eps);

  if (simd_lane_id == 0) {
    scales[in_index / group_size] = static_cast<T>(scale);
  }

  uint32_t packed = 0;

  const uint32_t start_bit =
      (simd_lane_id % threads_per_pack) * (values_per_reduce * bits);

#pragma clang loop unroll(full)
  for (int i = 0; i < values_per_reduce; i++) {
    uint32_t q = static_cast<uint32_t>(round(w_thread[i] / scale) + 1.0f);
    packed |= q << ((bits * i) + start_bit);
  }

#pragma clang loop unroll(full)
  for (uint32_t stride = 1; stride < threads_per_pack; stride <<= 1) {
    packed |= simd_shuffle_xor(packed, stride);
  }

  if (simd_lane_id % threads_per_pack == 0) {
    const size_t out_index = in_index / elements_per_uint;
    out[out_index] = packed;
  }
}

template <typename T, const int group_size, const int bits>
[[kernel]] void ternary_dequantize(
    const device uint8_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    device T* out [[buffer(3)]],
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  // Dequantize packed 2-bit codes with w = (q - 1) * scale.
  static_assert(
      bits == 2, "Ternary dequantization Metal kernel only supports 2 bits.");

  constexpr int pack_factor = 8 / bits; // values per packed byte

  const size_t offset = index.x + grid_dim.x * size_t(index.y);
  const size_t out_index = offset * pack_factor;

  const T scale = scales[out_index / group_size];
  const uint32_t val = w[offset];

  out += out_index;

#pragma clang loop unroll(full)
  for (int i = 0; i < pack_factor; i++) {
    const uint8_t d = (val >> (bits * i)) & 0x03;
    out[i] = static_cast<T>(scale * (int(d) - 1));
  }
}
