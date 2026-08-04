// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/quantized/qqmm_impl.h"
#include "mlx/backend/cuda/quantized/cublas_qqmm.h"

namespace mlx::core {

void qqmm_impl(
    cu::CommandEncoder& encoder,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    array& out,
    const array& a,
    const array& b,
    const array& a_scale,
    const array& b_scale,
    QuantizationMode mode,
    const GemmScalars& scalars,
    const std::optional<array>& bias) {
  std::string qmode = quantization_mode_to_string(mode);

  CublasQQMM qqmm(
      encoder.device(),
      a_transposed,
      M,
      K,
      lda,
      b_transposed,
      K,
      N,
      ldb,
      1, // batch_count
      0, // a_batch_stride
      0, // b_batch_stride
      out.dtype(),
      qmode);

  // Note: Unlike regular GEMM, no complex64 check is needed here because
  // quantized matmul only supports real floating types (float16, bfloat16,
  // float32). The type constraint is enforced in validate_qqmm_inputs() in
  // ops.cpp.
  if (bias) {
    qqmm.set_bias(encoder, *bias);
  }

  if (scalars.has_values()) {
    qqmm.run(
        encoder,
        out,
        a,
        b,
        a_scale,
        b_scale,
        *scalars.alpha_device,
        *scalars.beta_device);
  } else {
    qqmm.run(encoder, out, a, b, a_scale, b_scale);
  }
}

} // namespace mlx::core
