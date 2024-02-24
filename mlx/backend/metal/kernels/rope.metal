// Copyright © 2023-2024 Apple Inc.

#include <metal_math>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/utils.h"

template <typename T, bool traditional>
[[kernel]] void rope(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant const size_t strides[3],
    constant const int& offset,
    constant const float& base,
    constant const float& scale,
    uint3 pos [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  // Compute the input and output indices
  uint in_index_1, in_index_2;
  uint out_index_1, out_index_2;
  if (traditional) {
    out_index_1 = 2 * (pos.x + grid.x * (pos.y + grid.y * pos.z));
    out_index_2 = out_index_1 + 1;
    in_index_1 =
        2 * pos.x * strides[2] + pos.y * strides[1] + pos.z * strides[0];
    in_index_2 = in_index_1 + strides[2];
  } else {
    out_index_1 = pos.x + 2 * (grid.x * (pos.y + grid.y * pos.z));
    out_index_2 = out_index_1 + grid.x;
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
  float rx1 = x1 * costheta - x2 * sintheta;
  float rx2 = x1 * sintheta + x2 * costheta;
  out[out_index_1] = static_cast<T>(rx1);
  out[out_index_2] = static_cast<T>(rx2);
}

#define instantiate_rope(name, type, traditional)       \
  template [[host_name("rope_" #name)]] [[kernel]] void \
  rope<type, traditional>(                              \
      const device type* in [[buffer(0)]],              \
      device type* out [[buffer(1)]],                   \
      constant const size_t strides[3],                 \
      constant const int& offset,                       \
      constant const float& base,                       \
      constant const float& scale,                      \
      uint3 pos [[thread_position_in_grid]],            \
      uint3 grid [[threads_per_grid]]);

instantiate_rope(traditional_float16, half, true)
    instantiate_rope(traditional_bfloat16, bfloat16_t, true)
        instantiate_rope(traditional_float32, float, true)
            instantiate_rope(float16, half, false)
                instantiate_rope(bfloat16, bfloat16_t, false)
                    instantiate_rope(float32, float, false)
