// Copyright © 2024 Apple Inc.

#pragma once

namespace mlx::core {

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
  const int64_t matrix_stride; // Stride between matrices in batch
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

} // namespace mlx::core
