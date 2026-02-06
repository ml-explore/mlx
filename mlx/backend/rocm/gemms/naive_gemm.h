// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/device.h"

namespace mlx::core::rocm {

// Naive GEMM implementation for when rocBLAS is not available
// C = alpha * op(A) * op(B) + beta * C
// where op(X) = X if not transposed, X^T if transposed
void naive_gemm(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    float alpha = 1.0f,
    float beta = 0.0f);

// Batched naive GEMM
void naive_gemm_batched(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    int64_t stride_a,
    bool b_transposed,
    int64_t ldb,
    int64_t stride_b,
    int64_t stride_c,
    int batch_count,
    float alpha = 1.0f,
    float beta = 0.0f);

// Naive GEMM with explicit offsets (for non-uniform batch strides)
void naive_gemm_with_offset(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    int64_t a_offset,
    bool b_transposed,
    int64_t ldb,
    int64_t b_offset,
    int64_t out_offset,
    float alpha = 1.0f,
    float beta = 0.0f);

// Naive GEMM with explicit offsets and custom ldc (for grouped conv)
void naive_gemm_with_offset_ldc(
    CommandEncoder& encoder,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    int64_t a_offset,
    bool b_transposed,
    int64_t ldb,
    int64_t b_offset,
    int64_t ldc,
    int64_t out_offset,
    float alpha = 1.0f,
    float beta = 0.0f);

} // namespace mlx::core::rocm
