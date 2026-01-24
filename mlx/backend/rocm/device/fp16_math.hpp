// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

namespace mlx::core::rocm {

// Half-precision math functions for HIP

// Abs for half types
__device__ inline __half abs(__half x) {
  return __habs(x);
}

__device__ inline __hip_bfloat16 abs(__hip_bfloat16 x) {
  return __habs(x);
}

// Sqrt for half types
__device__ inline __half sqrt(__half x) {
  return hsqrt(x);
}

__device__ inline __hip_bfloat16 sqrt(__hip_bfloat16 x) {
  return hsqrt(x);
}

// Rsqrt for half types
__device__ inline __half rsqrt(__half x) {
  return hrsqrt(x);
}

__device__ inline __hip_bfloat16 rsqrt(__hip_bfloat16 x) {
  return hrsqrt(x);
}

// Exp for half types
__device__ inline __half exp(__half x) {
  return hexp(x);
}

__device__ inline __hip_bfloat16 exp(__hip_bfloat16 x) {
  return hexp(x);
}

// Log for half types
__device__ inline __half log(__half x) {
  return hlog(x);
}

__device__ inline __hip_bfloat16 log(__hip_bfloat16 x) {
  return hlog(x);
}

// Log2 for half types
__device__ inline __half log2(__half x) {
  return hlog2(x);
}

__device__ inline __hip_bfloat16 log2(__hip_bfloat16 x) {
  return hlog2(x);
}

// Log10 for half types
__device__ inline __half log10(__half x) {
  return hlog10(x);
}

__device__ inline __hip_bfloat16 log10(__hip_bfloat16 x) {
  return hlog10(x);
}

// Sin for half types
__device__ inline __half sin(__half x) {
  return hsin(x);
}

__device__ inline __hip_bfloat16 sin(__hip_bfloat16 x) {
  return hsin(x);
}

// Cos for half types
__device__ inline __half cos(__half x) {
  return hcos(x);
}

__device__ inline __hip_bfloat16 cos(__hip_bfloat16 x) {
  return hcos(x);
}

// Ceil for half types
__device__ inline __half ceil(__half x) {
  return hceil(x);
}

__device__ inline __hip_bfloat16 ceil(__hip_bfloat16 x) {
  return hceil(x);
}

// Floor for half types
__device__ inline __half floor(__half x) {
  return hfloor(x);
}

__device__ inline __hip_bfloat16 floor(__hip_bfloat16 x) {
  return hfloor(x);
}

// Rint (round to nearest integer) for half types
__device__ inline __half rint(__half x) {
  return hrint(x);
}

__device__ inline __hip_bfloat16 rint(__hip_bfloat16 x) {
  return hrint(x);
}

// Trunc for half types
__device__ inline __half trunc(__half x) {
  return htrunc(x);
}

__device__ inline __hip_bfloat16 trunc(__hip_bfloat16 x) {
  return htrunc(x);
}

// Conversion helpers
__device__ inline float half2float(__half x) {
  return __half2float(x);
}

__device__ inline __half float2half(float x) {
  return __float2half(x);
}

__device__ inline float bfloat162float(__hip_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ inline __hip_bfloat16 float2bfloat16(float x) {
  return __float2bfloat16(x);
}

// Erf for half types (compute in float)
__device__ inline __half erf(__half x) {
  return __float2half(erff(__half2float(x)));
}

__device__ inline __hip_bfloat16 erf(__hip_bfloat16 x) {
  return __float2bfloat16(erff(__bfloat162float(x)));
}

// Erfinv for half types (compute in float)
__device__ inline __half erfinv(__half x) {
  return __float2half(erfinvf(__half2float(x)));
}

__device__ inline __hip_bfloat16 erfinv(__hip_bfloat16 x) {
  return __float2bfloat16(erfinvf(__bfloat162float(x)));
}

// Expm1 for half types (compute in float)
__device__ inline __half expm1(__half x) {
  return __float2half(expm1f(__half2float(x)));
}

__device__ inline __hip_bfloat16 expm1(__hip_bfloat16 x) {
  return __float2bfloat16(expm1f(__bfloat162float(x)));
}

// Log1p for half types (compute in float)
__device__ inline __half log1p(__half x) {
  return __float2half(log1pf(__half2float(x)));
}

__device__ inline __hip_bfloat16 log1p(__hip_bfloat16 x) {
  return __float2bfloat16(log1pf(__bfloat162float(x)));
}

// Tanh for half types
__device__ inline __half tanh(__half x) {
  // HIP may not have htanh, compute in float
  return __float2half(tanhf(__half2float(x)));
}

__device__ inline __hip_bfloat16 tanh(__hip_bfloat16 x) {
  return __float2bfloat16(tanhf(__bfloat162float(x)));
}

// Sinh for half types
__device__ inline __half sinh(__half x) {
  return __float2half(sinhf(__half2float(x)));
}

__device__ inline __hip_bfloat16 sinh(__hip_bfloat16 x) {
  return __float2bfloat16(sinhf(__bfloat162float(x)));
}

// Cosh for half types
__device__ inline __half cosh(__half x) {
  return __float2half(coshf(__half2float(x)));
}

__device__ inline __hip_bfloat16 cosh(__hip_bfloat16 x) {
  return __float2bfloat16(coshf(__bfloat162float(x)));
}

// Asin for half types
__device__ inline __half asin(__half x) {
  return __float2half(asinf(__half2float(x)));
}

__device__ inline __hip_bfloat16 asin(__hip_bfloat16 x) {
  return __float2bfloat16(asinf(__bfloat162float(x)));
}

// Acos for half types
__device__ inline __half acos(__half x) {
  return __float2half(acosf(__half2float(x)));
}

__device__ inline __hip_bfloat16 acos(__hip_bfloat16 x) {
  return __float2bfloat16(acosf(__bfloat162float(x)));
}

// Atan for half types
__device__ inline __half atan(__half x) {
  return __float2half(atanf(__half2float(x)));
}

__device__ inline __hip_bfloat16 atan(__hip_bfloat16 x) {
  return __float2bfloat16(atanf(__bfloat162float(x)));
}

// Asinh for half types
__device__ inline __half asinh(__half x) {
  return __float2half(asinhf(__half2float(x)));
}

__device__ inline __hip_bfloat16 asinh(__hip_bfloat16 x) {
  return __float2bfloat16(asinhf(__bfloat162float(x)));
}

// Acosh for half types
__device__ inline __half acosh(__half x) {
  return __float2half(acoshf(__half2float(x)));
}

__device__ inline __hip_bfloat16 acosh(__hip_bfloat16 x) {
  return __float2bfloat16(acoshf(__bfloat162float(x)));
}

// Atanh for half types
__device__ inline __half atanh(__half x) {
  return __float2half(atanhf(__half2float(x)));
}

__device__ inline __hip_bfloat16 atanh(__hip_bfloat16 x) {
  return __float2bfloat16(atanhf(__bfloat162float(x)));
}

// Tan for half types
__device__ inline __half tan(__half x) {
  return __float2half(tanf(__half2float(x)));
}

__device__ inline __hip_bfloat16 tan(__hip_bfloat16 x) {
  return __float2bfloat16(tanf(__bfloat162float(x)));
}

} // namespace mlx::core::rocm
