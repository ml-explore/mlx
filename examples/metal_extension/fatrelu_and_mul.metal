// SPDX-License-Identifier: Apache-2.0
// Adapted from huggingface/kernels-community activation_metal at commit
// 47a3168d0808921eef2f7daca794a4fccae13078.
#include <metal_stdlib>
using namespace metal;

inline float fatrelu(float x, float threshold) {
  return (x > threshold) ? x : 0.0f;
}

inline float4 fatrelu(float4 x, float threshold) {
  return select(float4(0.0f), x, x > threshold);
}

kernel void fatrelu_and_mul_f16(
    device half* out [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& d [[buffer(2)]],
    constant float& threshold [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint base_idx = gid.x * 8;
  if (base_idx >= d) {
    return;
  }

  uint in_base = gid.y * 2u * d;
  uint out_base = gid.y * d;

  if (base_idx + 8 <= d) {
    float4 x0 = float4(
        *reinterpret_cast<device const half4*>(&input[in_base + base_idx]));
    float4 x1 = float4(
        *reinterpret_cast<device const half4*>(&input[in_base + base_idx + 4]));
    float4 y0 = float4(
        *reinterpret_cast<device const half4*>(&input[in_base + d + base_idx]));
    float4 y1 = float4(*reinterpret_cast<device const half4*>(
        &input[in_base + d + base_idx + 4]));

    *reinterpret_cast<device half4*>(&out[out_base + base_idx]) =
        half4(fatrelu(x0, threshold) * y0);
    *reinterpret_cast<device half4*>(&out[out_base + base_idx + 4]) =
        half4(fatrelu(x1, threshold) * y1);
  } else if (base_idx + 4 <= d) {
    float4 x = float4(
        *reinterpret_cast<device const half4*>(&input[in_base + base_idx]));
    float4 y = float4(
        *reinterpret_cast<device const half4*>(&input[in_base + d + base_idx]));
    *reinterpret_cast<device half4*>(&out[out_base + base_idx]) =
        half4(fatrelu(x, threshold) * y);

    for (uint i = base_idx + 4; i < d; i++) {
      float x = float(input[in_base + i]);
      float y = float(input[in_base + d + i]);
      out[out_base + i] = half(fatrelu(x, threshold) * y);
    }
  } else {
    for (uint i = base_idx; i < d; i++) {
      float x = float(input[in_base + i]);
      float y = float(input[in_base + d + i]);
      out[out_base + i] = half(fatrelu(x, threshold) * y);
    }
  }
}
