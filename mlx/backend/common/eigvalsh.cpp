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

namespace {

// Delegate to the eigenvalue decomposition taking into account differences in
// LAPACK implementations (basically how to pass the 'jobz' and 'uplo' strings
// to fortran).
int ssyevd_wrapper(char jobz, char uplo, float* matrix, float* w, int N) {
  int info;
  int lwork = -1;
  int liwork = -1;
  float work_query;
  int iwork_query;

  // Query for optimal work array sizes
#ifdef LAPACK_FORTRAN_STRLEN_END
  ssyevd_(
      /* jobz = */ &jobz,
      /* uplo = */ &uplo,
      /* n = */ &N,
      /* a = */ matrix,
      /* lda = */ &N,
      /* w = */ w,
      /* work = */ &work_query,
      /* lwork = */ &lwork,
      /* iwork = */ &iwork_query,
      /* liwork = */ &liwork,
      /* info = */ &info,
      /* jobz_len = */ static_cast<size_t>(1),
      /* uplo_len = */ static_cast<size_t>(1));
#else
  ssyevd_(
      /* jobz = */ &jobz,
      /* uplo = */ &uplo,
      /* n = */ &N,
      /* a = */ matrix,
      /* lda = */ &N,
      /* w = */ w,
      /* work = */ &work_query,
      /* lwork = */ &lwork,
      /* iwork = */ &iwork_query,
      /* liwork = */ &liwork,
      /* info = */ &info);
#endif

  lwork = static_cast<int>(work_query);
  liwork = iwork_query;

  std::vector<float> work(lwork);
  std::vector<int> iwork(liwork);

  // Compute eigenvalues (and optionally eigenvectors)
#ifdef LAPACK_FORTRAN_STRLEN_END
  ssyevd_(
      /* jobz = */ &jobz,
      /* uplo = */ &uplo,
      /* n = */ &N,
      /* a = */ matrix,
      /* lda = */ &N,
      /* w = */ w,
      /* work = */ work.data(),
      /* lwork = */ &lwork,
      /* iwork = */ iwork.data(),
      /* liwork = */ &liwork,
      /* info = */ &info,
      /* jobz_len = */ static_cast<size_t>(1),
      /* uplo_len = */ static_cast<size_t>(1));
#else
  ssyevd_(
      /* jobz = */ &jobz,
      /* uplo = */ &uplo,
      /* n = */ &N,
      /* a = */ matrix,
      /* lda = */ &N,
      /* w = */ w,
      /* work = */ work.data(),
      /* lwork = */ &lwork,
      /* iwork = */ iwork.data(),
      /* liwork = */ &liwork,
      /* info = */ &info);
#endif

  return info;
}

} // namespace

void eigvalsh_impl(
    const array& a,
    array& values,
    array& vectors,
    bool upper,
    bool compute_vectors) {
  char jobz = compute_vectors ? 'V' : 'N';
  char uplo = (upper) ? 'U' : 'L'; // Use upper triangle of the matrix

  // Create a copy of the input array for in-place computation
  array buffer = copy(a);

  const int N = static_cast<int>(a.shape(-1));
  const int num_matrices = static_cast<int>(a.size() / (N * N));

  // Allocate output arrays
  std::vector<int> values_shape = {num_matrices, N};
  values = array({}, values_shape, a.dtype());

  if (compute_vectors) {
    vectors = array({}, a.shape(), a.dtype());
  }

  float* matrix = buffer.data<float>();
  float* w = values.data<float>();
  float* vecs = compute_vectors ? vectors.data<float>() : nullptr;

  for (int i = 0; i < num_matrices; i++) {
    // Compute eigenvalue decomposition
    int info = ssyevd_wrapper(jobz, uplo, matrix, w, N);

    if (info != 0) {
      std::stringstream msg;
      msg << "[eigvalsh] Eigenvalue decomposition failed with error code "
          << info;
      throw std::runtime_error(msg.str());
    }

    // Copy eigenvectors if computed
    if (compute_vectors) {
      std::copy(matrix, matrix + N * N, vecs);
      vecs += N * N;
    }

    // Move to next matrix
    matrix += N * N;
    w += N;
  }
}

void Eigvalsh::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (inputs[0].dtype() != float32) {
    throw std::runtime_error("[Eigvalsh::eval] only supports float32.");
  }
  eigvalsh_impl(inputs[0], output[0], output[1], upper_, compute_vectors_);
}

} // namespace mlx::core