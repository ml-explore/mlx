// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/device.h"

#include <rocblas/rocblas.h>

namespace mlx::core::rocm {

// rocBLAS GEMM wrapper functions

void rocblas_gemm(
    CommandEncoder& encoder,
    bool transpose_a,
    bool transpose_b,
    int M,
    int N,
    int K,
    float alpha,
    const array& a,
    int lda,
    const array& b,
    int ldb,
    float beta,
    array& c,
    int ldc,
    Dtype dtype);

void rocblas_gemm_batched(
    CommandEncoder& encoder,
    bool transpose_a,
    bool transpose_b,
    int M,
    int N,
    int K,
    float alpha,
    const array& a,
    int lda,
    int64_t stride_a,
    const array& b,
    int ldb,
    int64_t stride_b,
    float beta,
    array& c,
    int ldc,
    int64_t stride_c,
    int batch_count,
    Dtype dtype);

} // namespace mlx::core::rocm
