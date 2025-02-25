// Copyright © 2023-2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/linalg.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

template <typename T>
void eigh_impl(
    array& vectors,
    array& values,
    const std::string& uplo,
    bool compute_eigenvectors) {
  auto vec_ptr = vectors.data<T>();
  auto eig_ptr = values.data<T>();

  char jobz = compute_eigenvectors ? 'V' : 'N';
  auto N = vectors.shape(-1);

  // Work query
  int lwork = -1;
  int liwork = -1;
  int info;
  {
    T work;
    int iwork;
    syevd<T>(
        &jobz,
        uplo.c_str(),
        &N,
        nullptr,
        &N,
        nullptr,
        &work,
        &lwork,
        &iwork,
        &liwork,
        &info);
    lwork = static_cast<int>(work);
    liwork = iwork;
  }

  auto work_buf = array::Data{allocator::malloc_or_wait(sizeof(T) * lwork)};
  auto iwork_buf = array::Data{allocator::malloc_or_wait(sizeof(int) * liwork)};
  for (size_t i = 0; i < vectors.size() / (N * N); ++i) {
    syevd<T>(
        &jobz,
        uplo.c_str(),
        &N,
        vec_ptr,
        &N,
        eig_ptr,
        static_cast<T*>(work_buf.buffer.raw_ptr()),
        &lwork,
        static_cast<int*>(iwork_buf.buffer.raw_ptr()),
        &liwork,
        &info);
    vec_ptr += N * N;
    eig_ptr += N;
    if (info != 0) {
      std::stringstream msg;
      msg << "[Eigh::eval_cpu] Eigenvalue decomposition failed with error code "
          << info;
      throw std::runtime_error(msg.str());
    }
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
  switch (a.dtype()) {
    case float32:
      eigh_impl<float>(vectors, values, uplo_, compute_eigenvectors_);
      break;
    case float64:
      eigh_impl<double>(vectors, values, uplo_, compute_eigenvectors_);
      break;
    default:
      throw std::runtime_error(
          "[Eigh::eval_cpu] only supports float32 or float64.");
  }
}

} // namespace mlx::core
