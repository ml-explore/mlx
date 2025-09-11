// Copyright © 2025 Apple Inc.
#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/device.h"

#include <cublasLt.h>

namespace mlx::core {

class CublasGemm {
 public:
  CublasGemm(
      cu::Device& device,
      Dtype dtype,
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
      int64_t b_batch_stride);

  CublasGemm(
      cu::Device& device,
      Dtype dtype,
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
      int64_t c_batch_stride);

  ~CublasGemm();

  // The output's descriptor is inferred from inputs by default, use this method
  // for unusual output.
  void set_out(
      Dtype dtype,
      bool transposed,
      uint64_t rows,
      uint64_t cols,
      int64_t ld,
      int32_t batch_count,
      int64_t batch_stride);

  void set_bias(cu::CommandEncoder& encoder, const array& bias);

  void run(
      cu::CommandEncoder& encoder,
      array& out,
      const array& a,
      const array& b,
      const Shape& batch_shape,
      const Strides& a_batch_strides,
      const Strides& b_batch_strides,
      float alpha = 1.0f);

  void run(
      cu::CommandEncoder& encoder,
      array& out,
      const array& a,
      const array& b,
      const array& c,
      const Shape& batch_shape,
      const Strides& a_batch_strides,
      const Strides& b_batch_strides,
      const Strides& c_batch_strides,
      float alpha,
      float beta);

 private:
  void run_batched(
      cu::CommandEncoder& encoder,
      array& out,
      const array& a,
      const array& b,
      const Shape& batch_shape,
      const Strides& a_batch_strides,
      const Strides& b_batch_strides,
      float alpha);

  void run_batched(
      cu::CommandEncoder& encoder,
      array& out,
      const array& a,
      const array& b,
      const array& c,
      const Shape& batch_shape,
      const Strides& a_batch_strides,
      const Strides& b_batch_strides,
      const Strides& c_batch_strides,
      float alpha,
      float beta);

  void execute(
      cu::CommandEncoder& encoder,
      void* out,
      const void* a,
      const void* b,
      const void* c,
      float alpha = 1,
      float beta = 0);

  uint64_t M_;
  uint64_t N_;
  cublasLtMatmulPreference_t pref_{nullptr};
  cublasLtHandle_t handle_{nullptr};
  cublasLtMatmulDesc_t matmul_desc_{nullptr};
  cublasLtMatrixLayout_t a_desc_{nullptr};
  cublasLtMatrixLayout_t b_desc_{nullptr};
  cublasLtMatrixLayout_t c_desc_{nullptr};
  cublasLtMatrixLayout_t out_desc_{nullptr};
  cublasLtMatmulHeuristicResult_t heuristic_;
};

} // namespace mlx::core
