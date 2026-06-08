// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/device.h"

namespace mlx::core::rocm {

// hipBLASLt GEMM wrapper functions
// hipBLASLt provides optimized GEMM kernels that can outperform rocBLAS
// for half-precision (fp16/bf16) matrix multiplications by using hardware
// matrix cores more efficiently and selecting algorithms via heuristics.

// Returns true if hipBLASLt is available and usable on the current device.
bool is_hipblaslt_available();

void hipblaslt_gemm(
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

void hipblaslt_gemm_batched(
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

// Raw hipBLASLt GEMM — parameters already in column-major convention
// (A/B swapped, M/N swapped). Call directly from inside kernel lambdas.
void hipblaslt_gemm_raw(
    hipStream_t stream,
    int op_a, // rocblas_operation / hipblasOperation_t value
    int op_b,
    int M,
    int N,
    int K,
    const float* alpha,
    const void* a_ptr,
    int lda,
    const void* b_ptr,
    int ldb,
    const float* beta,
    void* c_ptr,
    int ldc,
    int data_type, // hipDataType value (HIP_R_16BF, HIP_R_16F, HIP_R_32F)
    int compute_type); // hipblasComputeType_t value

} // namespace mlx::core::rocm
