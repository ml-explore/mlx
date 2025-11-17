// Copyright © 2025 Apple Inc.

#include <metal_integer>
#include <metal_math>

// clang-format off
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/sparse.h"

// Instantiate sparse matrix operations for common types
#define instantiate_sparse_ops(tname, type)                              \
  instantiate_kernel("sparse_mm_csr_" #tname, sparse_mm_csr, type)       \
  instantiate_kernel("sparse_mv_csr_" #tname, sparse_mv_csr, type)       \

// Instantiate for floating point types
instantiate_sparse_ops(float32, float)
instantiate_sparse_ops(float16, half)
instantiate_sparse_ops(bfloat16, bfloat16_t)
// clang-format on
