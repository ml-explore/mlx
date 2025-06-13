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
  // For now, always use one-sided Jacobi
  // Future PRs will add algorithm selection heuristics
  return SVDAlgorithm::JACOBI_ONE_SIDED;
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

  // TODO: Implement actual Metal SVD computation in subsequent PRs
  // For now, throw an informative error
  throw std::runtime_error(
      "[SVD::eval_gpu] Metal SVD implementation in progress. "
      "Matrix size: " +
      std::to_string(M) + "x" + std::to_string(N) +
      ", batch size: " + std::to_string(num_matrices) +
      ", compute_uv: " + (compute_uv ? "true" : "false"));
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
