// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/svd.h"
#ifdef ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <lapack.h>
#endif

namespace mlx::core {

void pseudoinverse_impl(const array& a, array& pinv) {
  const auto m = a.shape(-2);
  const auto n = a.shape(-1);
  const auto k = std::min(m, n);
  const auto rank = a.ndim();

  std::vector<int> u_shape = a.shape();
  u_shape[rank - 2] = m;
  u_shape[rank - 1] = m;

  std::vector<int> s_shape = a.shape();
  s_shape.pop_back();
  s_shape[rank - 2] = std::min(m, n);

  std::vector<int> vt_shape = a.shape();
  vt_shape[rank - 2] = n;
  vt_shape[rank - 1] = n;

  // Compute SVD
  array u(u_shape, float32, nullptr, {});
  array s(s_shape, float32, nullptr, {});
  array vt(vt_shape, float32, nullptr, {});
  svd_impl(a, u, s, vt);

  // Invert the singular values
  float* s_data = s.data<float>();
  // Create a diagonal matrix for the inverted singular values
  float* sigma_inv = (float*)calloc(k * k, sizeof(float));
  for (int i = 0; i < k; i++) {
    if (s_data[i] > 1e-6) { // Adjust the threshold for single precision
      sigma_inv[i * k + i] = 1.0f / s_data[i];
    }
  }

  if (m <= n) {
    // Compute Sigma^+ * U.T
    array u_sigma_inv({m, k}, float32, nullptr, {});
    u_sigma_inv.set_data(allocator::malloc_or_wait(u_sigma_inv.nbytes()));
    cblas_sgemm(
        CblasRowMajor, // layout
        CblasNoTrans, // TransA,
        CblasTrans, // TransB,
        m, // M,
        k, // N,
        m, // K,
        1.0f, // alpha,
        sigma_inv, // A,
        k, // lda, lda must be >= MAX(M,1)
        u.data<float>(), // B,
        m, // ldb, ldb must be >= MAX(N,1)
        0.0f, // beta,
        u_sigma_inv.data<float>(), // C,
        k // ldc  ldc must be >= MAX(N,1)
    );

    // Compute A^+ = V * (Sigma^+ * U.T)
    pinv.set_data(allocator::malloc_or_wait(pinv.nbytes()));
    cblas_sgemm(
        CblasRowMajor, // layout
        CblasTrans, // TransA,
        CblasNoTrans, // TransB,
        n, // M,
        k, // N,
        m, // K,
        1.0f, // alpha,
        vt.data<float>(), // A,
        n, // lda, lda must be >= MAX(M,1)
        u_sigma_inv.data<float>(), // B,
        m, // ldb, ldb must be >= MAX(N,1)
        0.0f, // beta,
        pinv.data<float>(), // C,
        k // ldc  ldc must be >= MAX(N,1)
    );
  } else {
    // Compute Vt.T * Sigma^+
    array v_sigma_inv({k, n}, float32, nullptr, {});
    v_sigma_inv.set_data(allocator::malloc_or_wait(v_sigma_inv.nbytes()));
    cblas_sgemm(
        CblasRowMajor, // layout
        CblasTrans, // TransA,
        CblasNoTrans, // TransB,
        k, // M,
        n, // N,
        k, // K,
        1.0f, // alpha,
        vt.data<float>(), // A,
        k, // lda, lda must be >= MAX(M,1)
        sigma_inv, // B,
        n, // ldb, ldb must be >= MAX(N,1)
        0.0f, // beta,
        v_sigma_inv.data<float>(), // C,
        n // ldc  ldc must be >= MAX(N,1)
    );
    // // Compute A^+ = (V * Sigma^+) * U.T
    pinv.set_data(allocator::malloc_or_wait(pinv.nbytes()));
    cblas_sgemm(
        CblasRowMajor, // layout
        CblasNoTrans, // TransA,
        CblasTrans, // TransB,
        k, // M,
        m, // N,
        n, // K,
        1.0f, // alpha,
        v_sigma_inv.data<float>(), // A,
        k, // lda, lda must be >= MAX(M,1):
        u.data<float>(), // B,
        m, // ldb, ldb must be >= MAX(N,1):
        0.0f, // beta,
        pinv.data<float>(), // C,
        m // ldc  ldc must be >= MAX(N,1)
    );
  }
}

void PseudoInverse::eval(const std::vector<array>& inputs, array& output) {
  pseudoinverse_impl(inputs[0], output);
}

} // namespace mlx::core