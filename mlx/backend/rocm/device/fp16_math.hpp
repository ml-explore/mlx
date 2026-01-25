// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mlx::core::rocm {

// Half-precision math functions for HIP
// Note: bfloat16 operations are computed in float since HIP doesn't have native bfloat16 math

// Helper to convert bfloat16 to float and back
__device__ inline float bf16_to_float(hip_bfloat16 x) {
  return static_cast<float>(x);
}

__device__ inline hip_bfloat16 float_to_bf16(float x) {
  return hip_bfloat16(x);
}

// Abs for half types
__device__ inline __half abs(__half x) {
  return __habs(x);
}

__device__ inline hip_bfloat16 abs(hip_bfloat16 x) {
  return float_to_bf16(fabsf(bf16_to_float(x)));
}

// Sqrt for half types
__device__ inline __half sqrt(__half x) {
  return hsqrt(x);
}

__device__ inline hip_bfloat16 sqrt(hip_bfloat16 x) {
  return float_to_bf16(sqrtf(bf16_to_float(x)));
}

// Rsqrt for half types
__device__ inline __half rsqrt(__half x) {
  return hrsqrt(x);
}

__device__ inline hip_bfloat16 rsqrt(hip_bfloat16 x) {
  return float_to_bf16(rsqrtf(bf16_to_float(x)));
}

// Exp for half types
__device__ inline __half exp(__half x) {
  return hexp(x);
}

__device__ inline hip_bfloat16 exp(hip_bfloat16 x) {
  return float_to_bf16(expf(bf16_to_float(x)));
}

// Log for half types
__device__ inline __half log(__half x) {
  return hlog(x);
}

__device__ inline hip_bfloat16 log(hip_bfloat16 x) {
  return float_to_bf16(logf(bf16_to_float(x)));
}

// Log2 for half types
__device__ inline __half log2(__half x) {
  return hlog2(x);
}

__device__ inline hip_bfloat16 log2(hip_bfloat16 x) {
  return float_to_bf16(log2f(bf16_to_float(x)));
}

// Log10 for half types
__device__ inline __half log10(__half x) {
  return hlog10(x);
}

__device__ inline hip_bfloat16 log10(hip_bfloat16 x) {
  return float_to_bf16(log10f(bf16_to_float(x)));
}

// Sin for half types
__device__ inline __half sin(__half x) {
  return hsin(x);
}

__device__ inline hip_bfloat16 sin(hip_bfloat16 x) {
  return float_to_bf16(sinf(bf16_to_float(x)));
}

// Cos for half types
__device__ inline __half cos(__half x) {
  return hcos(x);
}

__device__ inline hip_bfloat16 cos(hip_bfloat16 x) {
  return float_to_bf16(cosf(bf16_to_float(x)));
}

// Ceil for half types
__device__ inline __half ceil(__half x) {
  return hceil(x);
}

__device__ inline hip_bfloat16 ceil(hip_bfloat16 x) {
  return float_to_bf16(ceilf(bf16_to_float(x)));
}

// Floor for half types
__device__ inline __half floor(__half x) {
  return hfloor(x);
}

__device__ inline hip_bfloat16 floor(hip_bfloat16 x) {
  return float_to_bf16(floorf(bf16_to_float(x)));
}

// Rint (round to nearest integer) for half types
__device__ inline __half rint(__half x) {
  return hrint(x);
}

__device__ inline hip_bfloat16 rint(hip_bfloat16 x) {
  return float_to_bf16(rintf(bf16_to_float(x)));
}

// Trunc for half types
__device__ inline __half trunc(__half x) {
  return htrunc(x);
}

__device__ inline hip_bfloat16 trunc(hip_bfloat16 x) {
  return float_to_bf16(truncf(bf16_to_float(x)));
}

// Conversion helpers
__device__ inline float half2float(__half x) {
  return __half2float(x);
}

__device__ inline __half float2half(float x) {
  return __float2half(x);
}

__device__ inline float bfloat162float(hip_bfloat16 x) {
  return bf16_to_float(x);
}

__device__ inline hip_bfloat16 float2bfloat16(float x) {
  return float_to_bf16(x);
}

// Erf for half types (compute in float)
__device__ inline __half erf(__half x) {
  return __float2half(erff(__half2float(x)));
}

__device__ inline hip_bfloat16 erf(hip_bfloat16 x) {
  return float_to_bf16(erff(bf16_to_float(x)));
}

// Erfinv for half types (compute in float)
__device__ inline __half erfinv(__half x) {
  return __float2half(erfinvf(__half2float(x)));
}

__device__ inline hip_bfloat16 erfinv(hip_bfloat16 x) {
  return float_to_bf16(erfinvf(bf16_to_float(x)));
}

// Expm1 for half types (compute in float)
__device__ inline __half expm1(__half x) {
  return __float2half(expm1f(__half2float(x)));
}

__device__ inline hip_bfloat16 expm1(hip_bfloat16 x) {
  return float_to_bf16(expm1f(bf16_to_float(x)));
}

// Log1p for half types (compute in float)
__device__ inline __half log1p(__half x) {
  return __float2half(log1pf(__half2float(x)));
}

__device__ inline hip_bfloat16 log1p(hip_bfloat16 x) {
  return float_to_bf16(log1pf(bf16_to_float(x)));
}

// Tanh for half types
__device__ inline __half tanh(__half x) {
  // HIP may not have htanh, compute in float
  return __float2half(tanhf(__half2float(x)));
}

__device__ inline hip_bfloat16 tanh(hip_bfloat16 x) {
  return float_to_bf16(tanhf(bf16_to_float(x)));
}

// Sinh for half types
__device__ inline __half sinh(__half x) {
  return __float2half(sinhf(__half2float(x)));
}

__device__ inline hip_bfloat16 sinh(hip_bfloat16 x) {
  return float_to_bf16(sinhf(bf16_to_float(x)));
}

// Cosh for half types
__device__ inline __half cosh(__half x) {
  return __float2half(coshf(__half2float(x)));
}

__device__ inline hip_bfloat16 cosh(hip_bfloat16 x) {
  return float_to_bf16(coshf(bf16_to_float(x)));
}

// Asin for half types
__device__ inline __half asin(__half x) {
  return __float2half(asinf(__half2float(x)));
}

__device__ inline hip_bfloat16 asin(hip_bfloat16 x) {
  return float_to_bf16(asinf(bf16_to_float(x)));
}

// Acos for half types
__device__ inline __half acos(__half x) {
  return __float2half(acosf(__half2float(x)));
}

__device__ inline hip_bfloat16 acos(hip_bfloat16 x) {
  return float_to_bf16(acosf(bf16_to_float(x)));
}

// Atan for half types
__device__ inline __half atan(__half x) {
  return __float2half(atanf(__half2float(x)));
}

__device__ inline hip_bfloat16 atan(hip_bfloat16 x) {
  return float_to_bf16(atanf(bf16_to_float(x)));
}

// Asinh for half types
__device__ inline __half asinh(__half x) {
  return __float2half(asinhf(__half2float(x)));
}

__device__ inline hip_bfloat16 asinh(hip_bfloat16 x) {
  return float_to_bf16(asinhf(bf16_to_float(x)));
}

// Acosh for half types
__device__ inline __half acosh(__half x) {
  return __float2half(acoshf(__half2float(x)));
}

__device__ inline hip_bfloat16 acosh(hip_bfloat16 x) {
  return float_to_bf16(acoshf(bf16_to_float(x)));
}

// Atanh for half types
__device__ inline __half atanh(__half x) {
  return __float2half(atanhf(__half2float(x)));
}

__device__ inline hip_bfloat16 atanh(hip_bfloat16 x) {
  return float_to_bf16(atanhf(bf16_to_float(x)));
}

// Tan for half types
__device__ inline __half tan(__half x) {
  return __float2half(tanf(__half2float(x)));
}

__device__ inline hip_bfloat16 tan(hip_bfloat16 x) {
  return float_to_bf16(tanf(bf16_to_float(x)));
}

} // namespace mlx::core::rocm
