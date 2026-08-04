// SPDX-License-Identifier: Apache-2.0
// Adapted from huggingface/kernels-community activation_metal at commit
// 47a3168d0808921eef2f7daca794a4fccae13078.
#include <metal_stdlib>
using namespace metal;

inline float silu(float x) {
  return x / (1.0f + metal::exp(-x));
}

inline float4 silu(float4 x) {
  return x / (1.0f + metal::exp(-x));
}

kernel void silu_and_mul_f16(
    device half* out [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& d [[buffer(2)]],
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
        half4(silu(x0) * y0);
    *reinterpret_cast<device half4*>(&out[out_base + base_idx + 4]) =
        half4(silu(x1) * y1);
  } else if (base_idx + 4 <= d) {
    float4 x = float4(
        *reinterpret_cast<device const half4*>(&input[in_base + base_idx]));
    float4 y = float4(
        *reinterpret_cast<device const half4*>(&input[in_base + d + base_idx]));
    *reinterpret_cast<device half4*>(&out[out_base + base_idx]) =
        half4(silu(x) * y);

    for (uint i = base_idx + 4; i < d; i++) {
      float x = float(input[in_base + i]);
      float y = float(input[in_base + d + i]);
      out[out_base + i] = half(silu(x) * y);
    }
  } else {
    for (uint i = base_idx; i < d; i++) {
      float x = float(input[in_base + i]);
      float y = float(input[in_base + d + i]);
      out[out_base + i] = half(silu(x) * y);
    }
  }
}
