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
    Dtype out_dtype,
    QuantizationMode mode,
    float alpha) {
  // Invoke CublasQQMM
  std::string qmode = quantization_mode_to_string(mode);

  // Currently only supports non-batched QQMM operations
  // that covers all use cases for training, we will just collapse (batch,
  // seq_len) into (tokens)
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
      out_dtype,
      qmode);

  qqmm.run(encoder, out, a, b, a_scale, b_scale, alpha);
}

} // namespace mlx::core
