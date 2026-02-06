// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// Half-precision math functions for HIP
// Note: bfloat16 operations are computed in float since HIP doesn't have native
// bfloat16 math

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

// Complex math functions
// exp(z) = exp(x) * (cos(y) + i*sin(y))
__device__ inline hipFloatComplex exp(hipFloatComplex z) {
  float ex = expf(z.x);
  // Handle special case: if real part is -inf, result is 0
  if (isinf(z.x) && z.x < 0) {
    return make_hipFloatComplex(0.0f, 0.0f);
  }
  float s, c;
  sincosf(z.y, &s, &c);
  return make_hipFloatComplex(ex * c, ex * s);
}

// log(z) = log(|z|) + i*arg(z)
__device__ inline hipFloatComplex log(hipFloatComplex z) {
  float r = hypotf(z.x, z.y);
  float theta = atan2f(z.y, z.x);
  return make_hipFloatComplex(logf(r), theta);
}

// log10(z) = log(z) / log(10)
__device__ inline hipFloatComplex log10(hipFloatComplex z) {
  hipFloatComplex lz = log(z);
  constexpr float ln10 = 2.302585092994045684017991454684364208f;
  return make_hipFloatComplex(lz.x / ln10, lz.y / ln10);
}

// sin(z) = sin(x)*cosh(y) + i*cos(x)*sinh(y)
__device__ inline hipFloatComplex sin(hipFloatComplex z) {
  float sx, cx;
  sincosf(z.x, &sx, &cx);
  return make_hipFloatComplex(sx * coshf(z.y), cx * sinhf(z.y));
}

// cos(z) = cos(x)*cosh(y) - i*sin(x)*sinh(y)
__device__ inline hipFloatComplex cos(hipFloatComplex z) {
  float sx, cx;
  sincosf(z.x, &sx, &cx);
  return make_hipFloatComplex(cx * coshf(z.y), -sx * sinhf(z.y));
}

// tan(z) = sin(z) / cos(z)
__device__ inline hipFloatComplex tan(hipFloatComplex z) {
  return hipCdivf(sin(z), cos(z));
}

// sinh(z) = sinh(x)*cos(y) + i*cosh(x)*sin(y)
__device__ inline hipFloatComplex sinh(hipFloatComplex z) {
  float sy, cy;
  sincosf(z.y, &sy, &cy);
  return make_hipFloatComplex(sinhf(z.x) * cy, coshf(z.x) * sy);
}

// cosh(z) = cosh(x)*cos(y) + i*sinh(x)*sin(y)
__device__ inline hipFloatComplex cosh(hipFloatComplex z) {
  float sy, cy;
  sincosf(z.y, &sy, &cy);
  return make_hipFloatComplex(coshf(z.x) * cy, sinhf(z.x) * sy);
}

// tanh(z) = sinh(z) / cosh(z)
__device__ inline hipFloatComplex tanh(hipFloatComplex z) {
  return hipCdivf(sinh(z), cosh(z));
}

// sqrt(z) = sqrt(|z|) * (cos(arg(z)/2) + i*sin(arg(z)/2))
__device__ inline hipFloatComplex sqrt(hipFloatComplex z) {
  float r = hypotf(z.x, z.y);
  float theta = atan2f(z.y, z.x);
  float sr = sqrtf(r);
  float half_theta = theta * 0.5f;
  float s, c;
  sincosf(half_theta, &s, &c);
  return make_hipFloatComplex(sr * c, sr * s);
}

// abs(z) = |z| (returns complex with real part = magnitude, imag = 0)
__device__ inline hipFloatComplex abs(hipFloatComplex z) {
  return make_hipFloatComplex(hypotf(z.x, z.y), 0.0f);
}

// asin(z) = -i * log(i*z + sqrt(1 - z^2))
__device__ inline hipFloatComplex asin(hipFloatComplex z) {
  // i*z
  hipFloatComplex iz = make_hipFloatComplex(-z.y, z.x);
  // z^2
  hipFloatComplex z2 = hipCmulf(z, z);
  // 1 - z^2
  hipFloatComplex one_minus_z2 = make_hipFloatComplex(1.0f - z2.x, -z2.y);
  // sqrt(1 - z^2)
  hipFloatComplex sqrt_term = sqrt(one_minus_z2);
  // i*z + sqrt(1 - z^2)
  hipFloatComplex sum =
      make_hipFloatComplex(iz.x + sqrt_term.x, iz.y + sqrt_term.y);
  // log(...)
  hipFloatComplex log_term = log(sum);
  // -i * log(...) = (log.y, -log.x)
  return make_hipFloatComplex(log_term.y, -log_term.x);
}

// acos(z) = pi/2 - asin(z)
__device__ inline hipFloatComplex acos(hipFloatComplex z) {
  hipFloatComplex asin_z = asin(z);
  constexpr float pi_2 = 1.5707963267948966192313216916397514f;
  return make_hipFloatComplex(pi_2 - asin_z.x, -asin_z.y);
}

// atan(z) = (i/2) * log((i+z)/(i-z))
__device__ inline hipFloatComplex atan(hipFloatComplex z) {
  // i + z
  hipFloatComplex i_plus_z = make_hipFloatComplex(z.x, 1.0f + z.y);
  // i - z
  hipFloatComplex i_minus_z = make_hipFloatComplex(-z.x, 1.0f - z.y);
  // (i+z)/(i-z)
  hipFloatComplex ratio = hipCdivf(i_plus_z, i_minus_z);
  // log(...)
  hipFloatComplex log_term = log(ratio);
  // (i/2) * log(...) = (-log.y/2, log.x/2)
  return make_hipFloatComplex(-log_term.y * 0.5f, log_term.x * 0.5f);
}

// asinh(z) = log(z + sqrt(z^2 + 1))
__device__ inline hipFloatComplex asinh(hipFloatComplex z) {
  hipFloatComplex z2 = hipCmulf(z, z);
  hipFloatComplex z2_plus_1 = make_hipFloatComplex(z2.x + 1.0f, z2.y);
  hipFloatComplex sqrt_term = sqrt(z2_plus_1);
  hipFloatComplex sum =
      make_hipFloatComplex(z.x + sqrt_term.x, z.y + sqrt_term.y);
  return log(sum);
}

// acosh(z) = log(z + sqrt(z^2 - 1))
__device__ inline hipFloatComplex acosh(hipFloatComplex z) {
  hipFloatComplex z2 = hipCmulf(z, z);
  hipFloatComplex z2_minus_1 = make_hipFloatComplex(z2.x - 1.0f, z2.y);
  hipFloatComplex sqrt_term = sqrt(z2_minus_1);
  hipFloatComplex sum =
      make_hipFloatComplex(z.x + sqrt_term.x, z.y + sqrt_term.y);
  return log(sum);
}

// atanh(z) = (1/2) * log((1+z)/(1-z))
__device__ inline hipFloatComplex atanh(hipFloatComplex z) {
  hipFloatComplex one_plus_z = make_hipFloatComplex(1.0f + z.x, z.y);
  hipFloatComplex one_minus_z = make_hipFloatComplex(1.0f - z.x, -z.y);
  hipFloatComplex ratio = hipCdivf(one_plus_z, one_minus_z);
  hipFloatComplex log_term = log(ratio);
  return make_hipFloatComplex(log_term.x * 0.5f, log_term.y * 0.5f);
}

} // namespace mlx::core::rocm
