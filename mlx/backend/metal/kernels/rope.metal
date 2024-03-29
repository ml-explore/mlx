// Copyright © 2023-2024 Apple Inc.

#include <metal_math>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/utils.h"

template <typename T, bool traditional, bool forward>
[[kernel]] void rope(
    const device T *in [[buffer(0)]],
    device T * out [[buffer(1)]],
    constant const size_t strides[3],
    constant const size_t out_strides[3],
    constant const int& offset,
    constant const float& base,
    constant const float& scale,
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  // Compute the input and output indices
  uint in_index_1, in_index_2;
  uint out_index_1, out_index_2;
  if (traditional) {
    out_index_1 = 2 * pos.x * out_strides[2] + pos.y * out_strides[1] + pos.z * out_strides[0];
    out_index_2 = out_index_1 + 1;
    in_index_1 = 2 * pos.x * strides[2] + pos.y * strides[1] + pos.z * strides[0];
    in_index_2 = in_index_1 + strides[2];
  } else {
    out_index_1 = pos.x * out_strides[2] + pos.y * out_strides[1] + pos.z * out_strides[0];
    out_index_2 = out_index_1 + grid.x * out_strides[2];
    in_index_1 = pos.x * strides[2] + pos.y * strides[1] + pos.z * strides[0];
    in_index_2 = in_index_1 + grid.x * strides[2];
  }

  // Figure out L and d.
  float L = scale * static_cast<float>(pos.y + offset);
  float d = static_cast<float>(pos.x) / static_cast<float>(grid.x);

  // Compute costheta, sintheta
  float theta = L * metal::exp2(-d * base);
  float costheta = metal::fast::cos(theta);
  float sintheta = metal::fast::sin(theta);

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

#define instantiate_rope(name, type, traditional, forward) \
  template [[host_name("rope_" #name)]] \
  [[kernel]] void rope<type, traditional, forward>( \
      const device type* in [[buffer(0)]], \
      device type* out [[buffer(1)]], \
    constant const size_t strides[3], \
    constant const size_t out_strides[3], \
    constant const int& offset, \
    constant const float& base, \
    constant const float& scale, \
    uint3 pos [[thread_position_in_grid]], \
    uint3 grid [[threads_per_grid]]);

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
instantiate_rope(vjp_float32, float, false, false)
