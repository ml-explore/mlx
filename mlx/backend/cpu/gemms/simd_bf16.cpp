// Copyright © 2025-2026 Apple Inc.

#include "mlx/backend/cpu/gemm.h"
#include "mlx/backend/cpu/gemms/simd_low_precision_gemm.h"

namespace mlx::core {

template <>
void matmul<bfloat16_t>(
    const bfloat16_t* a,
    const bfloat16_t* b,
    bfloat16_t* out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    size_t ldc,
    float alpha,
    float beta,
    size_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides) {
  detail::matmul_lowp(
      a,
      b,
      out,
      a_transposed,
      b_transposed,
      lda,
      ldb,
      ldc,
      alpha,
      beta,
      batch_size,
      a_shape,
      a_strides,
      b_shape,
      b_strides);
}

} // namespace mlx::core
