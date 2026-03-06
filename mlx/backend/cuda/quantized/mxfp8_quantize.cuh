#pragma once

#include "mlx/backend/cuda/vector_types.cuh"

#include <cutlass/numeric_conversion.h>

namespace mlx::core::cu {

// Place holder for future fast path implementation
template <typename T, bool USE_SR>
__device__ __forceinline__ uint32_t scale_cvt_Tx4_to_fp8x4(
    const Vector4_t<T>& input,
    const float scale,
    uint32_t rbits) {
  cutlass::NumericArrayConverter<float, T, 4> fp32_t;
  auto scaled =
      fp32_t(*reinterpret_cast<const cutlass::Array<T, 4>*>(&input)) * scale;
  cutlass::NumericArrayConverter<cutlass::float_e4m3_t, float, 4> fp8_fp32;
  auto quant = fp8_fp32(scaled);
  return *reinterpret_cast<uint32_t*>(&quant);
}

} // namespace mlx::core::cu
