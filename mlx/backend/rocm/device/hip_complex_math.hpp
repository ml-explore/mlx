// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// Complex number type alias
using complex64_t = hipFloatComplex;

// Make complex from real and imaginary parts
__device__ inline hipFloatComplex make_complex(float real, float imag) {
  return make_hipFloatComplex(real, imag);
}

// Get real part
__device__ inline float real(hipFloatComplex z) {
  return hipCrealf(z);
}

// Get imaginary part
__device__ inline float imag(hipFloatComplex z) {
  return hipCimagf(z);
}

// Complex conjugate
__device__ inline hipFloatComplex conj(hipFloatComplex z) {
  return hipConjf(z);
}

// Complex absolute value (magnitude)
__device__ inline float abs(hipFloatComplex z) {
  return hipCabsf(z);
}

// Complex addition
__device__ inline hipFloatComplex operator+(
    hipFloatComplex a,
    hipFloatComplex b) {
  return hipCaddf(a, b);
}

// Complex subtraction
__device__ inline hipFloatComplex operator-(
    hipFloatComplex a,
    hipFloatComplex b) {
  return hipCsubf(a, b);
}

// Complex multiplication
__device__ inline hipFloatComplex operator*(
    hipFloatComplex a,
    hipFloatComplex b) {
  return hipCmulf(a, b);
}

// Complex division
__device__ inline hipFloatComplex operator/(
    hipFloatComplex a,
    hipFloatComplex b) {
  return hipCdivf(a, b);
}

// Complex negation
__device__ inline hipFloatComplex operator-(hipFloatComplex z) {
  return make_hipFloatComplex(-hipCrealf(z), -hipCimagf(z));
}

// Complex comparison (by magnitude, for sorting)
__device__ inline bool operator<(hipFloatComplex a, hipFloatComplex b) {
  float mag_a = hipCabsf(a);
  float mag_b = hipCabsf(b);
  return mag_a < mag_b;
}

__device__ inline bool operator>(hipFloatComplex a, hipFloatComplex b) {
  float mag_a = hipCabsf(a);
  float mag_b = hipCabsf(b);
  return mag_a > mag_b;
}

__device__ inline bool operator<=(hipFloatComplex a, hipFloatComplex b) {
  return !(a > b);
}

__device__ inline bool operator>=(hipFloatComplex a, hipFloatComplex b) {
  return !(a < b);
}

__device__ inline bool operator==(hipFloatComplex a, hipFloatComplex b) {
  return hipCrealf(a) == hipCrealf(b) && hipCimagf(a) == hipCimagf(b);
}

__device__ inline bool operator!=(hipFloatComplex a, hipFloatComplex b) {
  return !(a == b);
}

// Complex exponential
__device__ inline hipFloatComplex exp(hipFloatComplex z) {
  float r = expf(hipCrealf(z));
  float i = hipCimagf(z);
  return make_hipFloatComplex(r * cosf(i), r * sinf(i));
}

// Complex logarithm
__device__ inline hipFloatComplex log(hipFloatComplex z) {
  return make_hipFloatComplex(
      logf(hipCabsf(z)), atan2f(hipCimagf(z), hipCrealf(z)));
}

// Complex square root
__device__ inline hipFloatComplex sqrt(hipFloatComplex z) {
  float r = hipCabsf(z);
  float x = hipCrealf(z);
  float y = hipCimagf(z);
  float t = sqrtf((r + fabsf(x)) / 2.0f);
  if (x >= 0) {
    return make_hipFloatComplex(t, y / (2.0f * t));
  } else {
    return make_hipFloatComplex(fabsf(y) / (2.0f * t), copysignf(t, y));
  }
}

// Complex sine
__device__ inline hipFloatComplex sin(hipFloatComplex z) {
  float x = hipCrealf(z);
  float y = hipCimagf(z);
  return make_hipFloatComplex(sinf(x) * coshf(y), cosf(x) * sinhf(y));
}

// Complex cosine
__device__ inline hipFloatComplex cos(hipFloatComplex z) {
  float x = hipCrealf(z);
  float y = hipCimagf(z);
  return make_hipFloatComplex(cosf(x) * coshf(y), -sinf(x) * sinhf(y));
}

// Complex tangent
__device__ inline hipFloatComplex tan(hipFloatComplex z) {
  return hipCdivf(sin(z), cos(z));
}

// Complex hyperbolic sine
__device__ inline hipFloatComplex sinh(hipFloatComplex z) {
  float x = hipCrealf(z);
  float y = hipCimagf(z);
  return make_hipFloatComplex(sinhf(x) * cosf(y), coshf(x) * sinf(y));
}

// Complex hyperbolic cosine
__device__ inline hipFloatComplex cosh(hipFloatComplex z) {
  float x = hipCrealf(z);
  float y = hipCimagf(z);
  return make_hipFloatComplex(coshf(x) * cosf(y), sinhf(x) * sinf(y));
}

// Complex hyperbolic tangent
__device__ inline hipFloatComplex tanh(hipFloatComplex z) {
  return hipCdivf(sinh(z), cosh(z));
}

// Complex power
__device__ inline hipFloatComplex pow(
    hipFloatComplex base,
    hipFloatComplex exp) {
  // base^exp = exp(exp * log(base))
  return rocm::exp(hipCmulf(exp, rocm::log(base)));
}

} // namespace mlx::core::rocm
