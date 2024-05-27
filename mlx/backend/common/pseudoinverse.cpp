// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/svd.h"
#include "mlx/linalg.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/random.h"
#include "mlx/utils.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <lapack.h>
#endif

// #include "mlx/primitives.h"
// #include "mlx/fast_primitives.h"

using namespace mlx::core::linalg;
using namespace mlx::core::random;

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

  const int M = a.shape(-2);
  const int N = a.shape(-1);
  const int K = std::min(M, N);

  array u(u_shape, float32, nullptr, {});
  array s(s_shape, float32, nullptr, {});
  array vt(vt_shape, float32, nullptr, {});
  array result({N, K}, float32, nullptr, {});
  result.set_data(allocator::malloc_or_wait(result.nbytes()));

  // Compute SVD
  svd_impl(a, u, s, vt);

  // // Invert the singular values
  float* s_data = s.data<float>();
  // Compute U * Sigma^+
  // // A of shape M x N. The leading dimension is N since lapack receives Aᵀ.
  // const int lda = n;
  // U of shape M x M. (N x N in lapack).
  const int ldu = m;
  // Vᵀ of shape N x N. (M x M in lapack).
  const int ldvt = m;

  float* u_data = u.data<float>();
  // Compute A^+ = V^T * (U * Sigma^+)
  // Create a diagonal matrix for the inverted singular values
  float* sigma_inv = (float*)calloc(k * k, sizeof(float));
  for (int i = 0; i < k; i++) {
    if (s_data[i] > 1e-6) { // Adjust the threshold for single precision
      sigma_inv[i * k + i] = 1.0f / s_data[i];
    }
  }

  // Compute Sigma^+ @ U.T
  array u_sigma_inv({M, K}, float32, nullptr, {});
  u_sigma_inv.set_data(allocator::malloc_or_wait(u_sigma_inv.nbytes()));
  float* u_sigma_inv_data = u_sigma_inv.data<float>();
  const int ld_sigma_inv = m;
  cblas_sgemm(
      CblasRowMajor,
      CblasNoTrans,
      CblasTrans,
      m,
      k,
      m,
      1.0f,
      sigma_inv,
      ld_sigma_inv,
      u_data,
      ldu,
      0.0f,
      u_sigma_inv_data,
      k);

  // Compute A^+ = V^T * (Sigma^+ * U.T)
  pinv.set_data(allocator::malloc_or_wait(pinv.nbytes()));
  float* pinv_data = pinv.data<float>();
  float* vt_data = vt.data<float>();

  const array& _a = vt;
  array b = u_sigma_inv;

  size_t _M = _a.shape(-2);
  size_t _N = b.shape(-1);
  size_t _K = _a.shape(-1);

  if (_M == 0 || _N == 0) {
    return;
  }
  if (_K == 0) {
    std::memset(static_cast<void*>(pinv.data<float>()), 0, pinv.nbytes());
    return;
  }

  for (int i = 0; i < (_a.size() / (_M * _K)); ++i) {
    cblas_sgemm(
        CblasRowMajor,
        true ? CblasTrans : CblasNoTrans, // transA
        false ? CblasTrans : CblasNoTrans, // transB
        _M,
        _N,
        _K,
        1.0f, // alpha
        _a.data<float>() + elem_to_loc(_M * _K * i, _a.shape(), _a.strides()),
        n, // lda,
        b.data<float>() + elem_to_loc(_K * _N * i, b.shape(), b.strides()),
        m, // ldb,
        0.0f, // beta
        pinv.data<float>() + _M * _N * i,
        pinv.shape(-1) // ldc
    );
  }
}

void PseudoInverse::eval(const std::vector<array>& inputs, array& output) {
  pseudoinverse_impl(inputs[0], output);
}

} // namespace mlx::core