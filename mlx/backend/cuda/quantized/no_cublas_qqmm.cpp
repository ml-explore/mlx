// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/quantized/cublas_qqmm.h"

#include <fmt/format.h>
#include "mlx/backend/cuda/cublas_utils.h"

#include "mlx/backend/cuda/device.h"
#include "mlx/dtype_utils.h"
#include "mlx/utils.h"

namespace mlx::core {} // namespace

CublasQQMM::CublasQQMM(
    cu::Device& device,
    bool a_transposed,
    uint64_t a_rows,
    uint64_t a_cols,
    int64_t lda,
    bool b_transposed,
    uint64_t b_rows,
    uint64_t b_cols,
    int64_t ldb,
    int32_t batch_count,
    int64_t a_batch_stride,
    int64_t b_batch_stride,
    Dtype out_dtype,
    std::string qmode) {
  throw std::runtime_error(
      "[QQMatmul::eval_gpu] QQMM is only supported with CUDA 12.8 or higher.");
}

CublasQQMM::CublasQQMM(
    cu::Device& device,
    bool a_transposed,
    uint64_t a_rows,
    uint64_t a_cols,
    int64_t lda,
    bool b_transposed,
    uint64_t b_rows,
    uint64_t b_cols,
    int64_t ldb,
    int64_t ldc,
    int32_t batch_count,
    int64_t a_batch_stride,
    int64_t b_batch_stride,
    int64_t c_batch_stride,
    Dtype out_dtype,
    std::string qmode)
    : CublasQQMM(
          device,
          a_transposed,
          a_rows,
          a_cols,
          lda,
          b_transposed,
          b_rows,
          b_cols,
          ldb,
          batch_count,
          a_batch_stride,
          b_batch_stride,
          out_dtype,
          qmode) {}

void CublasQQMM::run(
    cu::CommandEncoder& encoder,
    array& out,
    const array& a,
    const array& b,
    const array& a_scale,
    const array& b_scale,
    float alpha) {}

} // namespace mlx::core
