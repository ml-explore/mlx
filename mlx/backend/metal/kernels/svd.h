// Copyright © 2024 Apple Inc.

#pragma once

// Complete Metal SVD implementation using one-sided Jacobi algorithm
//
// IMPLEMENTED FEATURES:
// - Full Jacobi iteration with rotation matrices
// - Convergence monitoring and control
// - Singular value and vector computation
// - Batched operations support
// - Optimized Metal compute kernels
//
// Note: These structs are defined outside namespace for Metal kernel
// compatibility - Metal kernels cannot access namespaced types directly

/**
 * Parameters for SVD Metal kernels
 */
struct SVDParams {
  const int M; // Matrix rows
  const int N; // Matrix columns
  const int K; // min(M, N) - number of singular values
  const int max_iterations; // Maximum Jacobi iterations
  const float tolerance; // Convergence threshold
  const int batch_size; // Number of matrices in batch
  const long matrix_stride; // Stride between matrices in batch
  const bool compute_uv; // Whether to compute U and V matrices
};

/**
 * Jacobi rotation parameters for SVD computation
 */
struct JacobiRotation {
  float cos_theta; // Cosine of rotation angle
  float sin_theta; // Sine of rotation angle
  int p, q; // Column indices for rotation (p < q)
};

/**
 * Convergence tracking for iterative SVD algorithms
 */
struct SVDConvergenceInfo {
  float off_diagonal_norm; // Norm of off-diagonal elements
  int iteration_count; // Current iteration number
  bool converged; // Whether algorithm has converged
};

namespace mlx::core {
// Namespace aliases for C++ code
using ::JacobiRotation;
using ::SVDConvergenceInfo;
using ::SVDParams;
} // namespace mlx::core
