// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/kernels/svd.h"
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

/**
 * Select appropriate SVD algorithm based on matrix properties
 */
enum class SVDAlgorithm {
  JACOBI_ONE_SIDED, // Default for most cases
  JACOBI_TWO_SIDED, // Better numerical stability (future)
  BIDIAGONAL_QR // For very large matrices (future)
};

SVDAlgorithm select_svd_algorithm(int M, int N, Dtype dtype) {
  // Algorithm selection based on matrix properties

  // For very large matrices, we might want different algorithms in the future
  if (std::max(M, N) > 2048) {
    // For now, still use Jacobi but with different parameters
    return SVDAlgorithm::JACOBI_ONE_SIDED;
  }

  // For very rectangular matrices, one-sided Jacobi is efficient
  double aspect_ratio = static_cast<double>(std::max(M, N)) / std::min(M, N);
  if (aspect_ratio > 3.0) {
    return SVDAlgorithm::JACOBI_ONE_SIDED;
  }

  // Default to one-sided Jacobi for most cases
  return SVDAlgorithm::JACOBI_ONE_SIDED;
}

/**
 * Compute SVD parameters based on matrix size and algorithm
 */
SVDParams compute_svd_params(
    int M,
    int N,
    size_t num_matrices,
    bool compute_uv,
    SVDAlgorithm algorithm) {
  const int K = std::min(M, N);

  // Adjust parameters based on matrix size and algorithm
  int max_iterations = 100;
  float tolerance = 1e-6f;

  // For larger matrices, we might need more iterations
  if (std::max(M, N) > 512) {
    max_iterations = 200;
    tolerance = 1e-5f; // Slightly relaxed tolerance for large matrices
  }

  // For very small matrices, we can use tighter tolerance
  if (std::max(M, N) < 64) {
    tolerance = 1e-7f;
  }

  return SVDParams{
      M, // M
      N, // N
      K, // K
      max_iterations, // max_iterations
      tolerance, // tolerance
      static_cast<int>(num_matrices), // batch_size
      M * N, // matrix_stride
      compute_uv // compute_uv
  };
}

/**
 * Validate SVD input parameters
 */
void validate_svd_inputs(const array& a) {
  if (a.ndim() < 2) {
    throw std::invalid_argument(
        "[SVD::eval_gpu] Input must have >= 2 dimensions");
  }

  if (a.dtype() != float32 && a.dtype() != float64) {
    throw std::invalid_argument(
        "[SVD::eval_gpu] Only float32 and float64 supported");
  }

  // Check for reasonable matrix size
  int M = a.shape(-2);
  int N = a.shape(-1);
  if (M > 4096 || N > 4096) {
    throw std::invalid_argument(
        "[SVD::eval_gpu] Matrix too large for current implementation. "
        "Maximum supported size is 4096x4096");
  }

  if (M == 0 || N == 0) {
    throw std::invalid_argument(
        "[SVD::eval_gpu] Matrix dimensions must be positive");
  }
}

} // anonymous namespace

/**
 * Metal implementation of SVD using one-sided Jacobi algorithm
 * This is a placeholder implementation that will be completed in subsequent PRs
 */
template <typename T>
void svd_metal_impl(
    const array& a,
    std::vector<array>& outputs,
    bool compute_uv,
    metal::Device& d,
    const Stream& s) {
  // Validate inputs
  validate_svd_inputs(a);

  // Extract matrix dimensions
  const int M = a.shape(-2);
  const int N = a.shape(-1);
  const int K = std::min(M, N);
  const size_t num_matrices = a.size() / (M * N);

  // Select algorithm and compute parameters
  SVDAlgorithm algorithm = select_svd_algorithm(M, N, a.dtype());
  SVDParams params =
      compute_svd_params(M, N, num_matrices, compute_uv, algorithm);

  // Allocate workspace arrays
  array AtA({static_cast<int>(num_matrices), N, N}, a.dtype(), nullptr, {});
  AtA.set_data(allocator::malloc(AtA.nbytes()));

  // Allocate rotation storage for Jacobi algorithm
  const int total_pairs = (N * (N - 1)) / 2;
  array rotations(
      {static_cast<int>(num_matrices), total_pairs, 4},
      float32,
      nullptr,
      {}); // JacobiRotation struct storage
  rotations.set_data(allocator::malloc(rotations.nbytes()));

  // Allocate convergence tracking
  array convergence_info(
      {static_cast<int>(num_matrices), 3},
      float32,
      nullptr,
      {}); // SVDConvergenceInfo struct storage
  convergence_info.set_data(allocator::malloc(convergence_info.nbytes()));

  // Get command encoder
  auto& compute_encoder = d.get_command_encoder(s.index);

  // Step 1: Preprocess - compute A^T * A
  {
    auto kernel = d.get_kernel("svd_preprocess_" + get_type_string(a.dtype()));
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(a, 0);
    compute_encoder.set_output_array(AtA, 1);
    compute_encoder.set_bytes(params, 2);

    MTL::Size grid_dims = MTL::Size(N, N, num_matrices);
    MTL::Size group_dims = MTL::Size(std::min(32, N), std::min(32, N), 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  // Step 2: Jacobi iterations with convergence checking
  bool converged = false;
  for (int iter = 0; iter < params.max_iterations && !converged; iter++) {
    // Perform Jacobi iteration
    {
      auto kernel =
          d.get_kernel("svd_jacobi_iteration_" + get_type_string(a.dtype()));
      compute_encoder.set_compute_pipeline_state(kernel);
      compute_encoder.set_input_array(AtA, 0);
      compute_encoder.set_input_array(rotations, 1);
      compute_encoder.set_input_array(convergence_info, 2);
      compute_encoder.set_bytes(params, 3);

      MTL::Size grid_dims = MTL::Size(total_pairs, 1, num_matrices);
      MTL::Size group_dims = MTL::Size(std::min(256, total_pairs), 1, 1);
      compute_encoder.dispatch_threads(grid_dims, group_dims);
    }

    // Check convergence every few iterations to avoid overhead
    if (iter % 5 == 4 || iter == params.max_iterations - 1) {
      auto kernel =
          d.get_kernel("svd_check_convergence_" + get_type_string(a.dtype()));
      compute_encoder.set_compute_pipeline_state(kernel);
      compute_encoder.set_input_array(AtA, 0);
      compute_encoder.set_input_array(convergence_info, 1);
      compute_encoder.set_bytes(params, 2);

      MTL::Size grid_dims = MTL::Size(1, 1, num_matrices);
      MTL::Size group_dims = MTL::Size(256, 1, 1);
      compute_encoder.dispatch_threads(grid_dims, group_dims);

      // Note: In a complete implementation, we would read back convergence
      // status from GPU and break early. For now, we run all iterations.
    }
  }

  // Step 3: Extract singular values
  {
    auto kernel = d.get_kernel(
        "svd_extract_singular_values_" + get_type_string(a.dtype()));
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(AtA, 0);

    if (compute_uv) {
      compute_encoder.set_output_array(outputs[1], 1); // S
    } else {
      compute_encoder.set_output_array(outputs[0], 1); // S
    }
    compute_encoder.set_bytes(params, 2);

    MTL::Size grid_dims = MTL::Size(K, 1, num_matrices);
    MTL::Size group_dims = MTL::Size(std::min(256, K), 1, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  // Step 4: Compute singular vectors (if requested)
  if (compute_uv) {
    auto kernel =
        d.get_kernel("svd_compute_vectors_" + get_type_string(a.dtype()));
    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(a, 0);
    compute_encoder.set_input_array(rotations, 1);
    compute_encoder.set_output_array(outputs[0], 2); // U
    compute_encoder.set_output_array(outputs[2], 3); // V
    compute_encoder.set_bytes(params, 4);

    MTL::Size grid_dims =
        MTL::Size(std::max(M, N), std::max(M, N), num_matrices);
    MTL::Size group_dims = MTL::Size(16, 16, 1);
    compute_encoder.dispatch_threads(grid_dims, group_dims);
  }

  // Add temporary arrays for cleanup
  d.add_temporaries({AtA, rotations, convergence_info}, s.index);
}

// Explicit template instantiations
template void svd_metal_impl<float>(
    const array& a,
    std::vector<array>& outputs,
    bool compute_uv,
    metal::Device& d,
    const Stream& s);

template void svd_metal_impl<double>(
    const array& a,
    std::vector<array>& outputs,
    bool compute_uv,
    metal::Device& d,
    const Stream& s);

} // namespace mlx::core
