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

    if (info != 0) {
      std::stringstream msg;
      msg << "[cholesky] ";
      // https://www.netlib.org/lapack/explore-html/d0/d18/group__ppsv_gab87078282c6c31853cfed4829976c0d9.html
      if (info > 0) {
        msg << "The leading principal minor of order " << info
            << " of the matrix is not positive, so the factorization could not be completed.";
      } else {
        msg << "The " << -info
            << " slot argument to the LAPACK Cholesky decomposition is invalid.";
      }
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
