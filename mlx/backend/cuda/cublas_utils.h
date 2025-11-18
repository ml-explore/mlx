// Copyright © 2025 Apple Inc.
#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/device.h"

#include <cublasLt.h>

namespace mlx::core {
namespace cublas_utils {

// Get the shared cublas preference for a device
cublasLtMatmulPreference_t get_preference(cu::Device& device);

// Allocate workspace for matmul if needed and return pointer
// The workspace array is added to the encoder's temporaries
void* allocate_workspace(cu::CommandEncoder& encoder, size_t workspace_size);

// Create matrix layout
cublasLtMatrixLayout_t create_matrix_layout(
    cudaDataType_t type,
    uint64_t rows,
    uint64_t cols,
    bool transposed,
    int64_t ld,
    int32_t batch_count,
    int64_t batch_stride);

// Execute matmul with pre-configured descriptors
void execute_matmul(
    cu::CommandEncoder& encoder,
    cublasLtHandle_t handle,
    cublasLtMatmulDesc_t matmul_desc,
    cublasLtMatrixLayout_t a_desc,
    cublasLtMatrixLayout_t b_desc,
    cublasLtMatrixLayout_t c_desc,
    cublasLtMatrixLayout_t out_desc,
    cublasLtMatmulHeuristicResult_t& heuristic,
    cublasLtMatmulPreference_t pref,
    void* out,
    const void* a,
    const void* b,
    const void* c,
    const void* alpha_ptr,
    const void* beta_ptr);

// Set bias for matmul epilogue
void set_bias(
    cu::CommandEncoder& encoder,
    cublasLtMatmulDesc_t matmul_desc,
    const array& bias);

} // namespace cublas_utils

} // namespace mlx::core
