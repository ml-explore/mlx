// Copyright © 2023-2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/lapack.h"
#include "mlx/linalg.h"
#include "mlx/primitives.h"

namespace mlx::core {

void cholesky_impl(const array& a, array& factor, bool upper) {
  // Lapack uses the column-major convention. We take advantage of the fact that
  // the matrix should be symmetric:
  //   (A)ᵀ = A
  // and that a column-major lower triangular matrix is a row-major upper
  // triangular matrix, so uplo is the opposite of what we would expect from
  // upper

  char uplo = (upper) ? 'L' : 'U';

  // The decomposition is computed in place, so just copy the input to the
  // output.
  copy(
      a,
      factor,
      a.flags().row_contiguous ? CopyType::Vector : CopyType::General);

  const int N = a.shape(-1);
  const size_t num_matrices = a.size() / (N * N);

  float* matrix = factor.data<float>();

  for (int i = 0; i < num_matrices; i++) {
    // Compute Cholesky factorization.
    int info;
    MLX_LAPACK_FUNC(spotrf)
    (
        /* uplo = */ &uplo,
        /* n = */ &N,
        /* a = */ matrix,
        /* lda = */ &N,
        /* info = */ &info);

    // TODO: We do nothing when the matrix is not positive semi-definite
    // because throwing an error would result in a crash. If we figure out how
    // to catch errors from the implementation we should throw.
    if (info < 0) {
      std::stringstream msg;
      msg << "[cholesky] Cholesky decomposition failed with error code "
          << info;
      throw std::runtime_error(msg.str());
    }

    // Zero out the upper/lower triangle while advancing the pointer to the
    // next matrix at the same time.
    for (int row = 0; row < N; row++) {
      if (upper) {
        std::fill(matrix, matrix + row, 0);
      } else {
        std::fill(matrix + row + 1, matrix + N, 0);
      }
      matrix += N;
    }
  }
}

void Cholesky::eval(const std::vector<array>& inputs, array& output) {
  if (inputs[0].dtype() != float32) {
    throw std::runtime_error("[Cholesky::eval] only supports float32.");
  }
  cholesky_impl(inputs[0], output, upper_);
}

} // namespace mlx::core
