// SPDX-License-Identifier: Apache-2.0
// Adapted from huggingface/kernels-community activation_metal at commit
// 47a3168d0808921eef2f7daca794a4fccae13078.
#include <metal_stdlib>
using namespace metal;

inline float erf_approx(float x) {
  float sign = (x >= 0.0f) ? 1.0f : -1.0f;
  x = metal::abs(x);

  constexpr float a1 = 0.254829592f;
  constexpr float a2 = -0.284496736f;
  constexpr float a3 = 1.421413741f;
  constexpr float a4 = -1.453152027f;
  constexpr float a5 = 1.061405429f;
  constexpr float p = 0.3275911f;

  float t = 1.0f / metal::fma(p, x, 1.0f);
  float polynomial = metal::fma(a5, t, a4);
  polynomial = metal::fma(polynomial, t, a3);
  polynomial = metal::fma(polynomial, t, a2);
  polynomial = metal::fma(polynomial, t, a1) * t;

  float y = metal::fma(-polynomial, metal::exp(-x * x), 1.0f);
  return sign * y;
}

inline float4 erf_approx(float4 x) {
  float4 sign = select(float4(-1.0f), float4(1.0f), x >= 0.0f);
  x = metal::abs(x);

  constexpr float a1 = 0.254829592f;
  constexpr float a2 = -0.284496736f;
  constexpr float a3 = 1.421413741f;
  constexpr float a4 = -1.453152027f;
  constexpr float a5 = 1.061405429f;
  constexpr float p = 0.3275911f;

  float4 t = 1.0f / metal::fma(float4(p), x, float4(1.0f));
  float4 polynomial = metal::fma(float4(a5), t, float4(a4));
  polynomial = metal::fma(polynomial, t, float4(a3));
  polynomial = metal::fma(polynomial, t, float4(a2));
  polynomial = metal::fma(polynomial, t, float4(a1)) * t;

  float4 y = metal::fma(-polynomial, metal::exp(-x * x), float4(1.0f));
  return sign * y;
}

inline float gelu(float x) {
  constexpr float inv_sqrt2 = 0.7071067811865475f;
  return 0.5f * x * metal::fma(erf_approx(x * inv_sqrt2), 1.0f, 1.0f);
}

inline float4 gelu(float4 x) {
  constexpr float inv_sqrt2 = 0.7071067811865475f;
  return 0.5f * x *
      metal::fma(erf_approx(x * inv_sqrt2), float4(1.0f), float4(1.0f));
}

kernel void gelu_f16(
    device half* out [[buffer(0)]],
    device const half* input [[buffer(1)]],
    constant uint& d [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  uint base_idx = gid.x * 8;
  if (base_idx >= d) {
    return;
  }

  uint base = gid.y * d;

  if (base_idx + 8 <= d) {
    float4 x0 =
        float4(*reinterpret_cast<device const half4*>(&input[base + base_idx]));
    float4 x1 = float4(
        *reinterpret_cast<device const half4*>(&input[base + base_idx + 4]));
    *reinterpret_cast<device half4*>(&out[base + base_idx]) = half4(gelu(x0));
    *reinterpret_cast<device half4*>(&out[base + base_idx + 4]) =
        half4(gelu(x1));
  } else if (base_idx + 4 <= d) {
    float4 x =
        float4(*reinterpret_cast<device const half4*>(&input[base + base_idx]));
    *reinterpret_cast<device half4*>(&out[base + base_idx]) = half4(gelu(x));
    for (uint i = base_idx + 4; i < d; i++) {
      out[base + i] = half(gelu(float(input[base + i])));
    }
  } else {
    for (uint i = base_idx; i < d; i++) {
      out[base + i] = half(gelu(float(input[base + i])));
    }
  }
}
