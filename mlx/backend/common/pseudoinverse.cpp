// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/svd.h"
#include "mlx/backend/common/utils.h"
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
        CblasRowMajor,              // layout
        CblasNoTrans,               // TransA,
        CblasTrans,                 // TransB,
        m,                          // M,
        k,                          // N,
        m,                          // K,
        1.0f,                       // alpha,
        sigma_inv,                  // A,
        k,                          // lda,
        u.data<float>(),            // B,
        m,                          // ldb,
        0.0f,                       // beta,
        u_sigma_inv.data<float>(),  // C,
        k                           // ldc
        );

    // Compute A^+ = V * (Sigma^+ * U.T)
    pinv.set_data(allocator::malloc_or_wait(pinv.nbytes()));
    cblas_sgemm(
        CblasRowMajor, 	           // layout
        CblasTrans, 	             // TransA,
        CblasNoTrans, 	           // TransB,
        n, 	                       // M,
        k, 	                       // N,
        m, 	                       // K,
        1.0f, 	                   // alpha,
        vt.data<float>(), 	       // A,
        n, 	                       // lda,
        u_sigma_inv.data<float>(), // B,
        m, 	                       // ldb,
        0.0f, 	                   // beta,
        pinv.data<float>(),        // C,
        k                          // ldc
      );
  } else {
    // Compute Sigma^+ * U.T
    array u_sigma_inv({k, m}, float32, nullptr, {});
    u_sigma_inv.set_data(allocator::malloc_or_wait(u_sigma_inv.nbytes()));
    cblas_sgemm(
      CblasRowMajor,	            // layout
      CblasNoTrans,	              // TransA,
      CblasTrans,	                // TransB,
      k,	                        // M,
      m,	                        // N,
      k,	                        // K,
      1.0f,	                      // alpha,
      sigma_inv,	                // A,
      m,	                        // lda,
      u.data<float>(),	          // B,
      m,	                        // ldb,
      0.0f,	                      // beta,
      u_sigma_inv.data<float>(),	// C,
      m	                          // ldc
        );
    // // Compute A^+ = V * (Sigma^+ * U.T)
    pinv.set_data(allocator::malloc_or_wait(pinv.nbytes()));
    // cblas_sgemm(
    //     CblasRowMajor,            // layout
    //     CblasTrans,               // TransA,
    //     CblasNoTrans,             // TransB,
    //     k,                        // M,
    //     m,                        // N,
    //     n,                        // K,
    //     1.0f,                     // alpha,
    //     vt.data<float>(),         // A,
    //     n,                        // lda,
    //     u_sigma_inv.data<float>(),// B,
    //     n,                        // ldb,
    //     0.0f,                     // beta,
    //     pinv.data<float>(),       // C,
    //     m                         // ldc
    // );
    copy(u_sigma_inv, pinv, u_sigma_inv.flags().row_contiguous ? CopyType::Vector : CopyType::General);
  }
}

void PseudoInverse::eval(const std::vector<array>& inputs, array& output) {
  pseudoinverse_impl(inputs[0], output);
}

} // namespace mlx::core