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
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    float alpha,
    float beta,
    int64_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides) {
  auto ndim = a_shape.size();
  int64_t M = a_shape[ndim - 2];
  int64_t N = b_shape[ndim - 1];
  int64_t K = a_shape[ndim - 1];

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
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    float alpha,
    float beta,
    int64_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides) {
  auto ndim = a_shape.size();
  int64_t M = a_shape[ndim - 2];
  int64_t N = b_shape[ndim - 1];
  int64_t K = a_shape[ndim - 1];

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

template <>
void matmul<complex64_t>(
    const complex64_t* a,
    const complex64_t* b,
    complex64_t* out,
    bool a_transposed,
    bool b_transposed,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    float alpha,
    float beta,
    int64_t batch_size,
    const Shape& a_shape,
    const Strides& a_strides,
    const Shape& b_shape,
    const Strides& b_strides) {
  auto ndim = a_shape.size();
  int64_t M = a_shape[ndim - 2];
  int64_t N = b_shape[ndim - 1];
  int64_t K = a_shape[ndim - 1];
  auto calpha = static_cast<complex64_t>(alpha);
  auto cbeta = static_cast<complex64_t>(beta);

  for (int i = 0; i < batch_size; ++i) {
    cblas_cgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans, // transA
        b_transposed ? CblasTrans : CblasNoTrans, // transB
        M,
        N,
        K,
        &calpha,
        a + elem_to_loc(M * K * i, a_shape, a_strides),
        lda,
        b + elem_to_loc(K * N * i, b_shape, b_strides),
        ldb,
        &cbeta,
        out + M * N * i,
        ldc);
  }
}

} // namespace mlx::core
