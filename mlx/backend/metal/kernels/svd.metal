// Copyright © 2024 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/svd.h"

using namespace metal;

// Forward declarations for SVD kernels
// These will be implemented in subsequent PRs

/**
 * Preprocess matrix for SVD computation
 * Computes A^T * A for one-sided Jacobi algorithm
 */
template <typename T>
[[kernel]] void svd_preprocess(
    const device T* A [[buffer(0)]],
    device T* AtA [[buffer(1)]],
    const constant SVDParams& params [[buffer(2)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

  const int M = params.M;
  const int N = params.N;
  const int batch_idx = tid.z;

  // Each thread computes one element of A^T * A
  const int i = tid.y; // Row in A^T * A
  const int j = tid.x; // Column in A^T * A

  if (i >= N || j >= N) {
    return;
  }

  // Compute A^T * A[i,j] = sum_k A[k,i] * A[k,j]
  T sum = T(0);
  const device T* A_batch = A + batch_idx * params.matrix_stride;

  for (int k = 0; k < M; k++) {
    sum += A_batch[k * N + i] * A_batch[k * N + j];
  }

  device T* AtA_batch = AtA + batch_idx * (N * N);
  AtA_batch[i * N + j] = sum;
}

/**
 * Perform one iteration of Jacobi rotations
 * Updates A^T * A matrix and tracks convergence
 */
template <typename T>
[[kernel]] void svd_jacobi_iteration(
    device T* AtA [[buffer(0)]],
    device JacobiRotation* rotations [[buffer(1)]],
    device SVDConvergenceInfo* convergence [[buffer(2)]],
    const constant SVDParams& params [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

  const int N = params.N;
  const int batch_idx = tid.z;
  const int pair_idx = tid.x; // Index of (p,q) pair to process

  // Calculate total number of pairs: N*(N-1)/2
  const int total_pairs = (N * (N - 1)) / 2;

  if (pair_idx >= total_pairs) {
    return;
  }

  // Convert linear pair index to (p,q) coordinates where p < q
  int p, q;
  int idx = pair_idx;
  for (p = 0; p < N - 1; p++) {
    int pairs_in_row = N - 1 - p;
    if (idx < pairs_in_row) {
      q = p + 1 + idx;
      break;
    }
    idx -= pairs_in_row;
  }

  device T* AtA_batch = AtA + batch_idx * (N * N);

  // Get matrix elements
  T app = AtA_batch[p * N + p];
  T aqq = AtA_batch[q * N + q];
  T apq = AtA_batch[p * N + q];

  // Check if rotation is needed
  if (abs(apq) < params.tolerance) {
    return;
  }

  // Compute Jacobi rotation angle
  T tau = (aqq - app) / (2 * apq);
  T t = (tau >= 0) ? 1 / (tau + sqrt(1 + tau * tau)) : 1 / (tau - sqrt(1 + tau * tau));
  T c = 1 / sqrt(1 + t * t);
  T s = t * c;

  // Store rotation for later use in computing singular vectors
  device JacobiRotation* rot_batch = rotations + batch_idx * total_pairs;
  rot_batch[pair_idx].cos_theta = c;
  rot_batch[pair_idx].sin_theta = s;
  rot_batch[pair_idx].p = p;
  rot_batch[pair_idx].q = q;

  // Apply rotation to A^T * A
  // Update diagonal elements
  AtA_batch[p * N + p] = c * c * app + s * s * aqq - 2 * s * c * apq;
  AtA_batch[q * N + q] = s * s * app + c * c * aqq + 2 * s * c * apq;
  AtA_batch[p * N + q] = 0; // Should be zero after rotation
  AtA_batch[q * N + p] = 0;

  // Update other elements in rows/columns p and q
  for (int i = 0; i < N; i++) {
    if (i != p && i != q) {
      T aip = AtA_batch[i * N + p];
      T aiq = AtA_batch[i * N + q];
      AtA_batch[i * N + p] = c * aip - s * aiq;
      AtA_batch[i * N + q] = s * aip + c * aiq;
      AtA_batch[p * N + i] = AtA_batch[i * N + p]; // Maintain symmetry
      AtA_batch[q * N + i] = AtA_batch[i * N + q];
    }
  }
}

/**
 * Extract singular values from diagonalized matrix
 */
template <typename T>
[[kernel]] void svd_extract_singular_values(
    const device T* AtA [[buffer(0)]],
    device T* S [[buffer(1)]],
    const constant SVDParams& params [[buffer(2)]],
    uint3 tid [[threadgroup_position_in_grid]]) {

  const int N = params.N;
  const int K = params.K;
  const int batch_idx = tid.z;
  const int i = tid.x;

  if (i >= K) {
    return;
  }

  const device T* AtA_batch = AtA + batch_idx * (N * N);
  device T* S_batch = S + batch_idx * K;

  // Singular values are square roots of diagonal elements of A^T * A
  T diagonal_element = AtA_batch[i * N + i];
  S_batch[i] = sqrt(max(diagonal_element, T(0))); // Ensure non-negative
}

/**
 * Check convergence of Jacobi iterations
 * Computes the Frobenius norm of off-diagonal elements
 */
template <typename T>
[[kernel]] void svd_check_convergence(
    const device T* AtA [[buffer(0)]],
    device SVDConvergenceInfo* convergence [[buffer(1)]],
    const constant SVDParams& params [[buffer(2)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

  const int N = params.N;
  const int batch_idx = tid.z;
  const int thread_id = lid.x;
  const int threads_per_group = 256; // Assuming 256 threads per group

  // Shared memory for reduction
  threadgroup float shared_sum[256];

  const device T* AtA_batch = AtA + batch_idx * (N * N);
  device SVDConvergenceInfo* conv_batch = convergence + batch_idx;

  // Each thread computes sum of squares of some off-diagonal elements
  float local_sum = 0.0f;

  for (int idx = thread_id; idx < N * N; idx += threads_per_group) {
    int i = idx / N;
    int j = idx % N;

    // Only consider off-diagonal elements
    if (i != j) {
      float val = static_cast<float>(AtA_batch[i * N + j]);
      local_sum += val * val;
    }
  }

  // Store in shared memory
  shared_sum[thread_id] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduction to compute total off-diagonal norm
  for (int stride = threads_per_group / 2; stride > 0; stride /= 2) {
    if (thread_id < stride) {
      shared_sum[thread_id] += shared_sum[thread_id + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Thread 0 writes the result
  if (thread_id == 0) {
    float off_diagonal_norm = sqrt(shared_sum[0]);
    conv_batch->off_diagonal_norm = off_diagonal_norm;
    conv_batch->converged = (off_diagonal_norm < params.tolerance);
  }
}

/**
 * Compute singular vectors U and V
 */
template <typename T>
[[kernel]] void svd_compute_vectors(
    const device T* A [[buffer(0)]],
    const device JacobiRotation* rotations [[buffer(1)]],
    device T* U [[buffer(2)]],
    device T* V [[buffer(3)]],
    const constant SVDParams& params [[buffer(4)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {

  const int M = params.M;
  const int N = params.N;
  const int batch_idx = tid.z;
  const int i = tid.y; // Row index
  const int j = tid.x; // Column index

  if (!params.compute_uv) {
    return; // Skip if not computing singular vectors
  }

  const int total_pairs = (N * (N - 1)) / 2;
  const device JacobiRotation* rot_batch = rotations + batch_idx * total_pairs;

  // Initialize V as identity matrix (right singular vectors)
  if (i < N && j < N) {
    device T* V_batch = V + batch_idx * (N * N);
    V_batch[i * N + j] = (i == j) ? T(1) : T(0);

    // Apply accumulated Jacobi rotations to build V
    // This gives us the right singular vectors
    for (int rot_idx = 0; rot_idx < total_pairs; rot_idx++) {
      int p = rot_batch[rot_idx].p;
      int q = rot_batch[rot_idx].q;
      T c = static_cast<T>(rot_batch[rot_idx].cos_theta);
      T s = static_cast<T>(rot_batch[rot_idx].sin_theta);

      // Apply rotation to columns p and q of V
      if (j == p || j == q) {
        T vip = V_batch[i * N + p];
        T viq = V_batch[i * N + q];
        V_batch[i * N + p] = c * vip - s * viq;
        V_batch[i * N + q] = s * vip + c * viq;
      }
    }
  }

  // Compute U = A * V * S^(-1) for left singular vectors
  if (i < M && j < N) {
    device T* U_batch = U + batch_idx * (M * M);
    const device T* A_batch = A + batch_idx * params.matrix_stride;
    const device T* V_batch = V + batch_idx * (N * N);

    // U[:, j] = A * V[:, j] / S[j]
    // This is a simplified computation - in practice we'd need the singular values
    T sum = T(0);
    for (int k = 0; k < N; k++) {
      sum += A_batch[i * N + k] * V_batch[k * N + j];
    }

    // For now, store the result without normalization
    // Proper normalization would require the computed singular values
    if (j < M) {
      U_batch[i * M + j] = sum;
    }
  }
}

// Template instantiations for float
template [[host_name("svd_preprocess_float")]] [[kernel]]
decltype(svd_preprocess<float>) svd_preprocess<float>;

template [[host_name("svd_jacobi_iteration_float")]] [[kernel]]
decltype(svd_jacobi_iteration<float>) svd_jacobi_iteration<float>;

template [[host_name("svd_extract_singular_values_float")]] [[kernel]]
decltype(svd_extract_singular_values<float>) svd_extract_singular_values<float>;

template [[host_name("svd_check_convergence_float")]] [[kernel]]
decltype(svd_check_convergence<float>) svd_check_convergence<float>;

template [[host_name("svd_compute_vectors_float")]] [[kernel]]
decltype(svd_compute_vectors<float>) svd_compute_vectors<float>;

// Template instantiations for double
template [[host_name("svd_preprocess_double")]] [[kernel]]
decltype(svd_preprocess<double>) svd_preprocess<double>;

template [[host_name("svd_jacobi_iteration_double")]] [[kernel]]
decltype(svd_jacobi_iteration<double>) svd_jacobi_iteration<double>;

template [[host_name("svd_extract_singular_values_double")]] [[kernel]]
decltype(svd_extract_singular_values<double>) svd_extract_singular_values<double>;

template [[host_name("svd_check_convergence_double")]] [[kernel]]
decltype(svd_check_convergence<double>) svd_check_convergence<double>;

template [[host_name("svd_compute_vectors_double")]] [[kernel]]
decltype(svd_compute_vectors<double>) svd_compute_vectors<double>;
