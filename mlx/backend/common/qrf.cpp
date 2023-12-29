// Copyright © 2023 Apple Inc.

#include <cassert>

#include <iostream>
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
  static void xormqr(
      const char* _Nonnull side,
      const char* _Nonnull trans,
      const __LAPACK_int* _Nonnull m,
      const __LAPACK_int* _Nonnull n,
      const __LAPACK_int* _Nonnull k,
      float* _Nullable a,
      const __LAPACK_int* _Nonnull lda,
      const float* _Nullable tau,
      float* _Nullable c,
      const __LAPACK_int* _Nonnull ldc,
      float* _Nonnull work,
      const __LAPACK_int* _Nonnull lwork,
      __LAPACK_int* _Nonnull info) {
    sormqr_(side, trans, m, n, k, a, lda, tau, c, ldc, work, lwork, info);
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
  R = triu(A, 0);

  int ldc = std::max(1, M);
  array C = identity(ldc, default_device());
  C.eval();
  const char side = 'L';
  const char trans = 'N';
  const int K = tau.size(); // no. of elementary reflectors

  assert(side == 'L' && M >= K && K >= 0);

  // retrieve Q from the elementary reflectors
  // uses the same worksize as before
  lpack<T>::xormqr(
      &side,
      &trans,
      &M,
      &N,
      &K,
      A.data<T>(),
      &lda,
      tau.data<T>(),
      C.data<float>(),
      &ldc,
      work.data<T>(),
      &lwork,
      &info);

  Q.set_data(
      allocator::malloc_or_wait(C.nbytes()),
      C.data_size(),
      C.strides(),
      C.flags());

  copy_inplace(C, Q, CopyType::Vector);
}

void QRF::eval(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  array A = inputs[0];

  if (!is_floating_point(A.dtype())) {
    throw std::runtime_error(
        "QR factorization is only supported for floating point types.");
  }

  std::cout << "A:" << A << "\n";
  auto fnormA = sqrt(sum(square(A)));
  std::cout << "fnorm A: " << fnormA << "\n";

  array Q = out;
  array R = out;
  qrf_impl<float>(A, Q, R);

  std::cout << "Q:" << Q << "\n";
  std::cout << "R:" << R << "\n";
  auto qr = matmul(Q, R);
  std::cout << "QR: " << qr << "\n";
  auto diff = subtract(A, qr);
  std::cout << "A - QR: " << diff << "\n";
  auto fnorm = sqrt(sum(square(diff)));
  std::cout << "fnorm: " << fnorm << "\n";
}

} // namespace mlx::core
