// Copyright © 2023-2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/lapack.h"
#include "mlx/linalg.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

void ssyevd(
    char jobz,
    char uplo,
    float* a,
    int N,
    float* w,
    float* work,
    int lwork,
    int* iwork,
    int liwork) {
  int info;
  MLX_LAPACK_FUNC(ssyevd)
  (
      /* jobz = */ &jobz,
      /* uplo = */ &uplo,
      /* n = */ &N,
      /* a = */ a,
      /* lda = */ &N,
      /* w = */ w,
      /* work = */ work,
      /* lwork = */ &lwork,
      /* iwork = */ iwork,
      /* liwork = */ &liwork,
      /* info = */ &info);
  if (info != 0) {
    std::stringstream msg;
    msg << "[Eigh::eval_cpu] Eigenvalue decomposition failed with error code "
        << info;
    throw std::runtime_error(msg.str());
  }
}

} // namespace

void Eigh::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  const auto& a = inputs[0];
  auto& values = outputs[0];

  auto vectors = compute_eigenvectors_
      ? outputs[1]
      : array(a.shape(), a.dtype(), nullptr, {});

  values.set_data(allocator::malloc_or_wait(values.nbytes()));

  copy(
      a,
      vectors,
      a.flags().row_contiguous ? CopyType::Vector : CopyType::General);

  if (compute_eigenvectors_) {
    // Set the strides and flags so the eigenvectors
    // are in the columns of the output
    auto flags = vectors.flags();
    auto strides = vectors.strides();
    auto ndim = a.ndim();
    std::swap(strides[ndim - 1], strides[ndim - 2]);

    if (a.size() > 1) {
      flags.row_contiguous = false;
      if (ndim > 2) {
        flags.col_contiguous = false;
      } else {
        flags.col_contiguous = true;
      }
    }
    vectors.move_shared_buffer(vectors, strides, flags, vectors.data_size());
  }

  auto vec_ptr = vectors.data<float>();
  auto eig_ptr = values.data<float>();

  char jobz = compute_eigenvectors_ ? 'V' : 'N';
  auto N = a.shape(-1);

  // Work query
  int lwork;
  int liwork;
  {
    float work;
    int iwork;
    ssyevd(jobz, uplo_[0], nullptr, N, nullptr, &work, -1, &iwork, -1);
    lwork = static_cast<int>(work);
    liwork = iwork;
  }

  auto work_buf = array::Data{allocator::malloc_or_wait(sizeof(float) * lwork)};
  auto iwork_buf = array::Data{allocator::malloc_or_wait(sizeof(int) * liwork)};
  for (size_t i = 0; i < a.size() / (N * N); ++i) {
    ssyevd(
        jobz,
        uplo_[0],
        vec_ptr,
        N,
        eig_ptr,
        static_cast<float*>(work_buf.buffer.raw_ptr()),
        lwork,
        static_cast<int*>(iwork_buf.buffer.raw_ptr()),
        liwork);
    vec_ptr += N * N;
    eig_ptr += N;
  }
}

} // namespace mlx::core
