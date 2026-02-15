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
    const GemmScalars& scalars) {
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

  if (scalars.uses_device_pointers()) {
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
