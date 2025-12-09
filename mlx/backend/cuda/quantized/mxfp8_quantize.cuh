#pragma once

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include "mlx/backend/cuda/vector_types.cuh"

namespace mlx::core::cu {

// TODO implement fast path
template <typename T>
__device__ __forceinline__ uint32_t
scale_cvt_Tx4_to_fp8x4_fallback(const Vector4_t<T> input, const float scale) {
  uint32_t out_fp8x4 = 0;
  Vector4_t<float> scaled;
  scaled.x = static_cast<float>(input.x) * scale;
  scaled.y = static_cast<float>(input.y) * scale;
  scaled.z = static_cast<float>(input.z) * scale;
  scaled.w = static_cast<float>(input.w) * scale;
  uint8_t q0 = __nv_fp8_e4m3(scaled.x).__x;
  uint8_t q1 = __nv_fp8_e4m3(scaled.y).__x;
  uint8_t q2 = __nv_fp8_e4m3(scaled.z).__x;
  uint8_t q3 = __nv_fp8_e4m3(scaled.w).__x;
  out_fp8x4 = (static_cast<uint32_t>(q3) << 24) |
      (static_cast<uint32_t>(q2) << 16) | (static_cast<uint32_t>(q1) << 8) |
      static_cast<uint32_t>(q0);
  return out_fp8x4;
}

// Place holder for future fast path implementation
template <typename T, bool USE_SR>
__device__ __forceinline__ uint32_t scale_cvt_Tx4_to_fp8x4(
    const Vector4_t<T> input,
    const float scale,
    uint32_t rbits) {
  return scale_cvt_Tx4_to_fp8x4_fallback(input, scale);
}
} // namespace mlx::core::cu