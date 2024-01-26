// Copyright © 2023-2024 Apple Inc.

#include <cassert>

#include <iostream>
#include "mlx/allocator.h"
#include "mlx/backend/common/copy.h"
#include "mlx/device.h"
#include "mlx/dtype.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/lapack.h>
#else
#include <lapack.h>
#endif

namespace mlx::core {
using allocator::Buffer;

template <typename T>
struct lpack;

template <>
struct lpack<float> {
  static void xgeqrf(
      const int* m,
      const int* n,
      float* a,
      const int* lda,
      float* tau,
      float* work,
      const int* lwork,
      int* info) {
    sgeqrf_(m, n, a, lda, tau, work, lwork, info);
  }
  static void xorgqr(
      const int* m,
      const int* n,
      const int* k,
      float* a,
      const int* lda,
      const float* tau,
      float* work,
      const int* lwork,
      int* info) {
    sorgqr_(m, n, k, a, lda, tau, work, lwork, info);
  }
};

template <typename T>
void qrf_impl(array& A, array& Q, array& R) {
  const int M = A.shape(0);
  const int N = A.shape(1);
  const int lda = std::max(M, N);

  // No. of elementary reflectors
  const int tau_size = std::min(M, N);
  // Holds scalar factors of the elementary reflectors
  Buffer tau = allocator::malloc_or_wait(sizeof(T) * tau_size);

  // Copy A to inplace input and make it col-contiguous
  array in(A.shape(), float32, nullptr, {});
  auto flags = in.flags();
  flags.col_contiguous = true;
  flags.row_contiguous = false;
  in.set_data(
      allocator::malloc_or_wait(in.nbytes()),
      in.nbytes(),
      {1, static_cast<size_t>(M)}, // col contiguous
      flags);
  copy_inplace(A, in, CopyType::GeneralGeneral);

  T optimal_work;
  int lwork = -1;
  int info;

  // Compute workspace size
  lpack<T>::xgeqrf(
      &M,
      &N,
      in.data<T>(),
      &lda,
      static_cast<T*>(tau.ptr()),
      &optimal_work,
      &lwork,
      &info);

  // Update workspace size
  lwork = optimal_work;
  Buffer work = allocator::malloc_or_wait(sizeof(T) * lwork);

  // Solve
  lpack<T>::xgeqrf(
      &M,
      &N,
      in.data<T>(),
      &lda,
      static_cast<T*>(tau.ptr()),
      static_cast<T*>(work.ptr()),
      &lwork,
      &info);

  R.set_data(allocator::malloc_or_wait(R.nbytes()));
  copy_inplace(in, R, CopyType::General);

  // Zero lower triangle
  for (int i = 0; i < R.shape(0); ++i) {
    for (int j = 0; j < i; ++j) {
      R.data<T>()[i * N + j] = 0;
    }
  }

  // Get work size
  int lwork2 = -1;
  lpack<T>::xorgqr(
      &M,
      &N,
      &tau_size,
      in.data<T>(),
      &lda,
      static_cast<T*>(tau.ptr()),
      &optimal_work,
      &lwork2,
      &info);

  if (optimal_work != lwork) {
    throw std::runtime_error("[QR::eval] Mismatch work size");
  }
  lwork2 = optimal_work;

  // Compute Q
  lpack<T>::xorgqr(
      &M,
      &N,
      &tau_size,
      in.data<T>(),
      &lda,
      static_cast<T*>(tau.ptr()),
      static_cast<T*>(work.ptr()),
      &lwork2,
      &info);

  Q.set_data(allocator::malloc_or_wait(Q.nbytes()));
  copy_inplace(in, Q, CopyType::General);
}

void QRF::eval(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 1);

  array A = inputs[0];

  if (!(A.dtype() == float32)) {
    throw std::runtime_error("[QRF::eval] only supports float32.");
  }

  array Q = outputs[0];
  array R = outputs[1];
  qrf_impl<float>(A, Q, R);
}

} // namespace mlx::core
