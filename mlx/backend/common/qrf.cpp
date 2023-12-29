// Copyright © 2023 Apple Inc.

#include <cassert>

#include "mlx/backend/common/copy.h"
#include "mlx/device.h"
#include "mlx/dtype.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include <iostream>
#include "mlx/utils.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/lapack.h>
#include <vecLib/lapack_types.h>
#endif

namespace mlx::core {

void QRF::eval(const std::vector<array>& inputs, array& out) {
  auto& in = inputs[0];

  array A = in;
  const int M = in.shape(0);
  const int N = in.shape(1);
  const int lda =  std::max(M, N);
  array tau = zeros({std::min(M, N)}, default_device()); // scalar factors of the elementary reflectors
  tau.eval();
  array work = array(1, in.dtype());
  int lwork = -1;
  int info;

  std::cout << "A:" << A << "\n";
  auto fnormA = sqrt(sum(square(A)));
  std::cout << "fnorm A: " << fnormA << "\n";

  // Compute workspace size
  sgeqrf_(
    &M,
    &N,
    A.data<float>(),
    &lda,
    tau.data<float>(),
    work.data<float>(),
    &lwork,
    &info
  );

  // Update workspace size
  lwork = work.item<float>();
  work = array(std::max(1, lwork), in.dtype());

  // Solve
  sgeqrf_(
    &M,
    &N,
    A.data<float>(),
    &lda,
    tau.data<float>(),
    work.data<float>(),
    &lwork,
    &info
  );

  // For m ≥ n, R is an upper triangular matrix.
  // For m < n, R is an upper trapezoidal matrix.
  array R = triu(A, 0);

  int ldc = std::max(1, M);
  array C = identity(ldc, default_device());
  C.eval();
  const char side = 'L';
  const char should_trans = 'N';
  const int K = tau.size(); // no. of elementary reflectors

  assert(side == 'L' && M >= K && K >= 0);

  sormqr_(
    &side,
    &should_trans,
    &M,
    &N,
    &K,
    A.data<float>(),
    &lda,
    tau.data<float>(),
    C.data<float>(),
    &ldc,
    work.data<float>(),
    &lwork,
    &info);

  std::cout << "Q:" << C << "\n";
  std::cout << "R:" << R << "\n";
  auto qr = matmul(C, R);
  std::cout << "QR: " << qr << "\n";
  auto diff = subtract(A, qr);
  std::cout << "A - QR: " << diff << "\n";
  auto fnorm = sqrt(sum(square(diff)));
  std::cout << "fnorm: " << fnorm << "\n";

  out.set_data(
    allocator::malloc_or_wait(C.nbytes()),
    C.data_size(),
    C.strides(),
    C.flags()
  );

  // TODO(): copies only 1 array, we need two Q and R
  // find a way to return two arrays
  copy_inplace(C, out, CopyType::Vector);
}

} // namespace mlx::core
