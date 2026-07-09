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
// Same as hipblaslt_gemm but with raw device pointers (segmented / offset GEMMs).
void hipblaslt_gemm_ptrs(
    CommandEncoder& encoder,
    bool transpose_a,
    bool transpose_b,
    int M,
    int N,
    int K,
    float alpha,
    const void* a,
    int lda,
    const void* b,
    int ldb,
    float beta,
    void* c,
    int ldc,
    Dtype dtype);

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

// True iff this device has e4m3 fp8 GEMM kernels (probed once, cached).
bool device_has_fp8_gemm(int device_id);

// Raw fp8 (e4m3) GEMM: A/B are e4m3 buffers in column-major convention,
// a_scale/b_scale are device float scalars applied as descale factors
// (out = a_scale*b_scale * (A@B)), output written as bf16. Picks the fastest
// available algorithm for the shape (heuristic top-pick is poor for fp8).
void hipblaslt_gemm_fp8_raw(
    hipStream_t stream,
    int op_a,
    int op_b,
    int M,
    int N,
    int K,
    const void* a_ptr,
    int lda,
    const void* b_ptr,
    int ldb,
    void* c_ptr,
    int ldc,
    const float* a_scale,
    const float* b_scale);

} // namespace mlx::core::rocm
