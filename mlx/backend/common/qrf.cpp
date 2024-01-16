// Copyright © 2023 Apple Inc.

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
#include <vecLib/lapack_types.h>
#endif

namespace mlx::core {

template <typename T>
struct lpack;

template <>
struct lpack<float> {
  static void xgeqrf(
      const __LAPACK_int* _Nonnull m,
      const __LAPACK_int* _Nonnull n,
      float* _Nullable a,
      const __LAPACK_int* _Nonnull lda,
      float* _Nullable tau,
      float* _Nonnull work,
      const __LAPACK_int* _Nonnull lwork,
      __LAPACK_int* _Nonnull info) {
    sgeqrf_(m, n, a, lda, tau, work, lwork, info);
  }
  static void xorgqr(
      const __LAPACK_int * _Nonnull m,
  const __LAPACK_int * _Nonnull n,
  const __LAPACK_int * _Nonnull k,
  float * _Nullable a,
  const __LAPACK_int * _Nonnull lda,
  const float * _Nullable tau,
  float * _Nonnull work,
  const __LAPACK_int * _Nonnull lwork,
  __LAPACK_int * _Nonnull info) {
    sorgqr_(m, n, k, a, lda, tau, work, lwork, info);
  }
};

template <typename T>
void qrf_impl(array& A, array& Q, array& R) {
  const int M = A.shape(0);
  const int N = A.shape(1);
  const int lda = std::max(M, N);
  // Holds scalar factors of the elementary reflectors
  array tau = zeros({std::min(M, N)}, default_device());
  tau.eval();
  array work = array(1, A.dtype());
  int lwork = -1;
  int info;

  // Compute workspace size
  lpack<T>::xgeqrf(
      &M, &N, A.data<T>(), &lda, tau.data<T>(), work.data<T>(), &lwork, &info);

  // Update workspace size
  lwork = work.item<T>();
  work = array(std::max(1, lwork), A.dtype());

  // Solve
  lpack<T>::xgeqrf(
      &M, &N, A.data<T>(), &lda, tau.data<T>(), work.data<T>(), &lwork, &info);

  // For m ≥ n, R is an upper triangular matrix.
  // For m < n, R is an upper trapezoidal matrix.
  array R_ = triu(A, 0);
  R_.eval();

  R.set_data(
    allocator::malloc_or_wait(R_.nbytes()),
    R_.data_size(),
    R_.strides(),
    R_.flags());

  copy_inplace(R_, R, CopyType::Vector);

  // No. of elementary reflectors
  const int K = tau.size();

  // retrieve Q from the elementary reflectors
  // uses the same worksize as before
  lpack<T>::xorgqr(
      &M,
      &N,
      &K,
      A.data<T>(),
      &lda,
      tau.data<T>(),
      work.data<T>(),
      &lwork,
      &info);

  Q.set_data(
      allocator::malloc_or_wait(A.nbytes()),
      A.data_size(),
      A.strides(),
      A.flags());

  copy_inplace(A, Q, CopyType::Vector);
}

void QRF::eval(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 1);

  array A = inputs[0];

  if (!(A.dtype() == Dtype::Val::float32)) {
    throw std::runtime_error(
        "QR factorization is only supported for floating point 32bit type.");
  }

  array Q = outputs[0];
  array R = outputs[1];
  qrf_impl<float>(A, Q, R);

  // std::cout << "Q: " <<  Q << "\n";
  // std::cout << "R: " <<  R << "\n";
}

} // namespace mlx::core
