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
    bool upper) {
  char jobz = 'N'; // Only compute eigenvalues
  char uplo = (upper) ? 'U' : 'L';

  array buffer = copy(a);

  const int N = static_cast<int>(a.shape(-1));
  const int num_matrices = static_cast<int>(a.size() / (N * N));

  std::vector<int> values_shape = {num_matrices, N};
  values = array(allocator::malloc(num_matrices * N * size_of(a.dtype())), values_shape, a.dtype());

  float* matrix = buffer.data<float>();
  float* w = values.data<float>();

  for (int i = 0; i < num_matrices; i++) {
    int info = ssyevd_wrapper(jobz, uplo, matrix, w, N);

    if (info != 0) {
      std::stringstream msg;
      msg << "[eigvalsh] Eigenvalue decomposition failed with error code " << info;
      throw std::runtime_error(msg.str());
    }

    matrix += N * N;
    w += N;
  }
}

void eigh_impl(
    const array& a,
    array& vectors,
    bool upper) {
  char jobz = 'V'; // Compute both eigenvalues and eigenvectors
  char uplo = (upper) ? 'U' : 'L';

  array buffer = copy(a);

  const int N = static_cast<int>(a.shape(-1));
  const int num_matrices = static_cast<int>(a.size() / (N * N));

  std::vector<int> vectors_shape = a.shape();
  vectors = array(allocator::malloc(a.size() * size_of(a.dtype())), vectors_shape, a.dtype());

  float* matrix = buffer.data<float>();
  float* vecs = vectors.data<float>();

  // Temporary buffer for eigenvalues (we don't return these)
  std::vector<float> w(N);

  for (int i = 0; i < num_matrices; i++) {
    int info = ssyevd_wrapper(jobz, uplo, matrix, w.data(), N);

    if (info != 0) {
      std::stringstream msg;
      msg << "[eigh] Eigenvalue decomposition failed with error code " << info;
      throw std::runtime_error(msg.str());
    }

    // Copy eigenvectors to the output array
    std::copy(matrix, matrix + N * N, vecs);

    matrix += N * N;
    vecs += N * N;
  }
}

void Eigvalsh::eval(
    const std::vector<array>& inputs, array& output) {
  if (inputs[0].dtype() != float32) {
    throw std::runtime_error("[Eigvalsh::eval] only supports float32.");
  }
  eigvalsh_impl(inputs[0], output, upper_);
}

void Eigh::eval(
    const std::vector<array>& inputs, array& output) {
  if (inputs[0].dtype() != float32) {
    throw std::runtime_error("[Eigh::eval] only supports float32.");
  }
  eigh_impl(inputs[0], output, upper_);
}

} // namespace mlx::core