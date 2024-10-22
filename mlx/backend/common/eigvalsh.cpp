// Copyright © 2023-2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/array.h"
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
  values = array(
      allocator::malloc(num_matrices * N * size_of(a.dtype())),
      values_shape,
      a.dtype());

  float* matrix = buffer.data<float>();
  float* w = values.data<float>();

  if (compute_eigenvectors) {
    std::vector<int> vectors_shape = a.shape();
    vectors = array(
        allocator::malloc(a.size() * size_of(a.dtype())),
        vectors_shape,
        a.dtype());
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

void Eigh::eval(const std::vector<array>& inputs, std::vector<array>& outputs) {
  // Validate the number of inputs
  if (inputs.size() != 1) {
    throw std::invalid_argument(
        "[Eigh::eval] Expected exactly one input array.");
  }

  const array& input = inputs[0];

  // Ensure the input array is evaluated before accessing its data
  const_cast<array&>(input).eval();

  // Validate the data type
  Dtype input_dtype = input.dtype(); // Changed from 'dtype_t' to 'Dtype'

  // Validate the number of dimensions (expecting at least 2D)
  if (input.ndim() < 2) {
    throw std::invalid_argument(
        "[Eigh::eval] Input array must be at least 2-dimensional.");
  }

  array values{};
  array vectors{};
  eigh_impl(input, values, vectors, upper_, compute_eigenvectors_);

  // Ensure the output arrays are evaluated
  values.eval();
  if (compute_eigenvectors_) {
    vectors.eval();
    outputs = {values, vectors};
  } else {
    outputs = {values};
  }
}

} // namespace mlx::core