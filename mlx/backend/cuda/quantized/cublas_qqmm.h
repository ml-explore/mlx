// Copyright © 2025 Apple Inc.
#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/device.h"

#include <cublasLt.h>

namespace mlx::core {

class CublasQQMM {
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
      std::string_view quantization_mode,
      int32_t batch_count,
      int64_t a_batch_stride,
      int64_t b_batch_stride);

  ~CublasQQMM();

  void run(
      cu::CommandEncoder& encoder,
      array& out,
      const array& a,
      const array& b,
      const array& a_scale,
      const array& b_scale,
      const Shape& batch_shape,
      const Strides& a_batch_strides,
      const Strides& b_batch_strides,
      float alpha = 1.0f);

  //   void run(
  //       cu::CommandEncoder& encoder,
  //       array& out,
  //       const array& a,
  //       const array& b,
  //       const array& c,
  //       const Shape& batch_shape,
  //       const Strides& a_batch_strides,
  //       const Strides& b_batch_strides,
  //       const Strides& c_batch_strides,
  //       float alpha,
  //       float beta);

  //  private:
  //   void run_batched(
  //       cu::CommandEncoder& encoder,
  //       array& out,
  //       const array& a,
  //       const array& b,
  //       const Shape& batch_shape,
  //       const Strides& a_batch_strides,
  //       const Strides& b_batch_strides,
  //       float alpha);

  //   void run_batched(
  //       cu::CommandEncoder& encoder,
  //       array& out,
  //       const array& a,
  //       const array& b,
  //       const array& c,
  //       const Shape& batch_shape,
  //       const Strides& a_batch_strides,
  //       const Strides& b_batch_strides,
  //       const Strides& c_batch_strides,
  //       float alpha,
  //       float beta);

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

  uint64_t M_;
  uint64_t N_;
  std::string quantization_mode_;
  cudaDataType_t scale_type_;
  cublasLtMatmulPreference_t pref_{nullptr};
  cublasLtHandle_t handle_{nullptr};
  cublasLtMatmulDesc_t matmul_desc_{nullptr};
  cublasLtMatrixLayout_t a_desc_{nullptr};
  cublasLtMatrixLayout_t b_desc_{nullptr};
  cublasLtMatrixLayout_t c_desc_{nullptr};
  cublasLtMatrixLayout_t out_desc_{nullptr};
  cublasLtMatmulMatrixScale_t a_scale_mode_;
  cublasLtMatmulMatrixScale_t b_scale_mode_;
  cublasLtMatmulMatrixScale_t c_scale_mode_;
  cublasLtMatmulMatrixScale_t out_scale_mode_;
  cublasLtMatmulHeuristicResult_t heuristic_;
};

} // namespace mlx::core
