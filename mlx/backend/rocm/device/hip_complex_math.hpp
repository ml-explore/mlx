// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// HIP complex math functions
__device__ inline hipFloatComplex hip_complex_add(
    hipFloatComplex a,
    hipFloatComplex b) {
  return make_hipFloatComplex(
      hipCrealf(a) + hipCrealf(b), hipCimagf(a) + hipCimagf(b));
}

__device__ inline hipFloatComplex hip_complex_sub(
    hipFloatComplex a,
    hipFloatComplex b) {
  return make_hipFloatComplex(
      hipCrealf(a) - hipCrealf(b), hipCimagf(a) - hipCimagf(b));
}

__device__ inline hipFloatComplex hip_complex_mul(
    hipFloatComplex a,
    hipFloatComplex b) {
  float real = hipCrealf(a) * hipCrealf(b) - hipCimagf(a) * hipCimagf(b);
  float imag = hipCrealf(a) * hipCimagf(b) + hipCimagf(a) * hipCrealf(b);
  return make_hipFloatComplex(real, imag);
}

__device__ inline hipFloatComplex hip_complex_div(
    hipFloatComplex a,
    hipFloatComplex b) {
  float denom = hipCrealf(b) * hipCrealf(b) + hipCimagf(b) * hipCimagf(b);
  float real =
      (hipCrealf(a) * hipCrealf(b) + hipCimagf(a) * hipCimagf(b)) / denom;
  float imag =
      (hipCimagf(a) * hipCrealf(b) - hipCrealf(a) * hipCimagf(b)) / denom;
  return make_hipFloatComplex(real, imag);
}

__device__ inline float hip_complex_abs(hipFloatComplex z) {
  return sqrtf(hipCrealf(z) * hipCrealf(z) + hipCimagf(z) * hipCimagf(z));
}

__device__ inline hipFloatComplex hip_complex_conj(hipFloatComplex z) {
  return make_hipFloatComplex(hipCrealf(z), -hipCimagf(z));
}

} // namespace mlx::core::rocm