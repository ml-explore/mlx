// Copyright © 2025 Apple Inc.
#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/cublas_utils.h"
#include "mlx/backend/cuda/device.h"

#include <cublasLt.h>

namespace mlx::core {

class CublasQQMM : public CublasMatmulBase {
 public:
  CublasQQMM(
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
      std::string quantization_mode);

  CublasQQMM(
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
      std::string quantization_mode);

  void run(
      cu::CommandEncoder& encoder,
      array& out,
      const array& a,
      const array& b,
      const array& a_scale,
      const array& b_scale,
      float alpha = 1.0f);

 private:
  void run_batched(
      cu::CommandEncoder& encoder,
      array& out,
      const array& a,
      const array& b,
      const array& a_scale,
      const array& b_scale,
      const Shape& batch_shape,
      const Strides& a_batch_strides,
      const Strides& b_batch_strides,
      float alpha);

  void execute(
      cu::CommandEncoder& encoder,
      void* out,
      const void* a,
      const void* b,
      const void* a_scale,
      const void* b_scale,
      const void* c,
      float alpha = 1,
      float beta = 0);

  std::string quantization_mode_;
  cublasLtMatmulMatrixScale_t a_scale_mode_;
  cublasLtMatmulMatrixScale_t b_scale_mode_;
  cublasLtMatmulMatrixScale_t c_scale_mode_;
  cublasLtMatmulMatrixScale_t out_scale_mode_;
};

} // namespace mlx::core
