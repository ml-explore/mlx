// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cpu/gemm.h"
#include "mlx/backend/cpu/lapack.h"

namespace mlx::core {

template <>
void matmul<float>(
    const float* a,
    const float* b,
    float* out,
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
  auto ndim = a_shape.size();
  size_t M = a_shape[ndim - 2];
  size_t N = b_shape[ndim - 1];
  size_t K = a_shape[ndim - 1];

  for (int i = 0; i < batch_size; ++i) {
    cblas_sgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans, // transA
        b_transposed ? CblasTrans : CblasNoTrans, // transB
        M,
        N,
        K,
        alpha,
        a + elem_to_loc(M * K * i, a_shape, a_strides),
        lda,
        b + elem_to_loc(K * N * i, b_shape, b_strides),
        ldb,
        beta,
        out + M * N * i,
        ldc);
  }
}

template <>
void matmul<double>(
    const double* a,
    const double* b,
    double* out,
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
  auto ndim = a_shape.size();
  size_t M = a_shape[ndim - 2];
  size_t N = b_shape[ndim - 1];
  size_t K = a_shape[ndim - 1];

  for (int i = 0; i < batch_size; ++i) {
    cblas_dgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans, // transA
        b_transposed ? CblasTrans : CblasNoTrans, // transB
        M,
        N,
        K,
        alpha,
        a + elem_to_loc(M * K * i, a_shape, a_strides),
        lda,
        b + elem_to_loc(K * N * i, b_shape, b_strides),
        ldb,
        beta,
        out + M * N * i,
        ldc);
  }
}

} // namespace mlx::core
