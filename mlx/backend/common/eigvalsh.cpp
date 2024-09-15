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

void eigh_impl(
    const array& a,
    array& values,
    array& vectors,
    bool upper,
    bool compute_eigenvectors) {
  char jobz = compute_eigenvectors ? 'V' : 'N';
  char uplo = upper ? 'U' : 'L';

  array buffer = copy(a);

  const int N = static_cast<int>(a.shape(-1));
  const int num_matrices = static_cast<int>(a.size() / (N * N));

  std::vector<int> values_shape = {num_matrices, N};
  values = array(allocator::malloc(num_matrices * N * size_of(a.dtype())), values_shape, a.dtype());

  float* matrix = buffer.data<float>();
  float* w = values.data<float>();

  if (compute_eigenvectors) {
    std::vector<int> vectors_shape = a.shape();
    vectors = array(allocator::malloc(a.size() * size_of(a.dtype())), vectors_shape, a.dtype());
  }

  float* vecs = compute_eigenvectors ? vectors.data<float>() : nullptr;

  for (int i = 0; i < num_matrices; i++) {
    int info = ssyevd_wrapper(jobz, uplo, matrix, w, N);

    if (info != 0) {
      std::stringstream msg;
      msg << "[eigh] Eigenvalue decomposition failed with error code " << info;
      throw std::runtime_error(msg.str());
    }

    if (compute_eigenvectors) {
      // Copy eigenvectors to the output array
      std::copy(matrix, matrix + N * N, vecs);
      vecs += N * N;
    }

    matrix += N * N;
    w += N;
  }
}

void EighPrimitive::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  if (inputs[0].dtype() != float32) {
    throw std::runtime_error("[EighPrimitive::eval] only supports float32.");
  }

  array values, vectors;
  eigh_impl(inputs[0], values, vectors, upper_, compute_eigenvectors_);

  if (compute_eigenvectors_) {
    outputs = {values, vectors};
  } else {
    outputs = {values};
  }
}

} // namespace mlx::core