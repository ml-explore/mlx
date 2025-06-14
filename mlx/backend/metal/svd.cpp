#include "mlx/backend/metal/kernels/svd.h"
#include <iostream>
#include "mlx/allocator.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

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
        "[SVD::eval_gpu] Input must have >= 2 dimensions, got " +
        std::to_string(a.ndim()) + "D array");
  }

  if (a.dtype() != float32 && a.dtype() != float64) {
    throw std::invalid_argument(
        "[SVD::eval_gpu] Only float32 and float64 supported, got " +
        type_to_name(a.dtype()));
  }

  // Note: Metal does not support double precision, will fall back to CPU
  if (a.dtype() == float64) {
    throw std::runtime_error(
        "[SVD::eval_gpu] Double precision not supported on Metal GPU. "
        "Use mx.set_default_device(mx.cpu) for float64 SVD operations.");
  }

  // Check for reasonable matrix size
  int M = a.shape(-2);
  int N = a.shape(-1);
  if (M > 4096 || N > 4096) {
    throw std::invalid_argument(
        "[SVD::eval_gpu] Matrix too large for current implementation. "
        "Got " +
        std::to_string(M) + "x" + std::to_string(N) +
        ", maximum supported size is 4096x4096");
  }

  if (M == 0 || N == 0) {
    throw std::invalid_argument(
        "[SVD::eval_gpu] Matrix dimensions must be positive, got " +
        std::to_string(M) + "x" + std::to_string(N));
  }

  // Check for empty arrays
  if (a.size() == 0) {
    throw std::invalid_argument("[SVD::eval_gpu] Input matrix is empty");
  }

  // Check for NaN or Inf values
  if (!all(isfinite(a)).item<bool>()) {
    throw std::invalid_argument(
        "[SVD::eval_gpu] Input matrix contains NaN or Inf values");
  }
}

} // anonymous namespace

/**
 * Metal implementation of SVD using one-sided Jacobi algorithm
 * This is a placeholder implementation that will be completed in subsequent PRs
 * For now, it validates GPU path and falls back to CPU computation
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

  // For now, fall back to CPU implementation but validate we're on GPU path
  // This allows testing the infrastructure while Metal kernels are being
  // developed

  // Get CPU stream for fallback computation
  auto cpu_stream = default_stream(Device::cpu);

  // Call CPU SVD implementation directly
  SVD cpu_svd(cpu_stream, compute_uv);
  cpu_svd.eval_cpu({a}, outputs);

  // Note: For now, outputs are computed on CPU. In a full implementation,
  // we would copy them to GPU memory here.
}

// Explicit template instantiation for float32 only
// Note: Metal does not support double precision
template void svd_metal_impl<float>(
    const array& a,
    std::vector<array>& outputs,
    bool compute_uv,
    metal::Device& d,
    const Stream& s);

} // namespace mlx::core
