// Copyright © 2023-2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/common/copy.h"
#include "mlx/linalg.h"
#include "mlx/primitives.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <lapack.h>
#endif

namespace mlx::core {

void cholesky_impl(const array& a, array& T, bool upper) {
  // Lapack uses the column-major convention. We take advantage of the fact that
  // the matrix should be symmetric:
  //   (A)ᵀ = A
  // and that a column-major lower triangular matrix is a row-major upper
  // triangular matrix, so uplo is the opposite of what we would expect from
  // upper

  char uplo;
  if (upper) {
    uplo = 'L';
  } else {
    uplo = 'U';
  }

  // The decomposition is computed in place, so just copy the input to the
  // output.
  copy(a, T, a.flags().row_contiguous ? CopyType::Vector : CopyType::General);

  const int N = a.shape(-1);
  const size_t num_matrices = a.size() / (N * N);

  int info;

  for (int i = 0; i < num_matrices; i++) {
    // Compute Cholesky factorization.
    spotrf_(
        /* uplo = */ &uplo,
        /* n = */ &N,
        /* a = */ T.data<float>() + N * N * i,
        /* lda = */ &N,
        /* info = */ &info);

    if (info != 0) {
      std::stringstream ss;
      if (info < 0)
        ss << "cholesky_impl: failed with error code " << info;
      else {
        ss << "cholesky_impl: matrix is not positive definite.";
      }
      throw std::runtime_error(ss.str());
    }

    // Zero out the upper/lower triangle.
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < j; k++) {
        if (upper)
          T.data<float>()[N * N * i + j * N + k] = 0.;
        else
          T.data<float>()[N * N * i + k * N + j] = 0.;
      }
    }
  }
}

void Cholesky::eval(const std::vector<array>& inputs, array& output) {
  if (inputs[0].dtype() != float32) {
    throw std::runtime_error("[Cholesky::eval] only supports float32.");
  }
  cholesky_impl(inputs[0], output, upper_);
}

std::pair<std::vector<array>, std::vector<int>> Cholesky::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto ax = axes[0] >= 0 ? 0 : -1;
  auto a = axes[0] > 0 ? moveaxis(inputs[0], axes[0], 0, stream()) : inputs[0];
  return {{linalg::cholesky(a, upper_, stream())}, {ax}};
}

} // namespace mlx::core
