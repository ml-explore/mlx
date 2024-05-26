// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/svd.h"
#include "mlx/primitives.h"
#include "mlx/ops.h"
#include "mlx/linalg.h"
#include "mlx/random.h"

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
  // If  A = U Σ V ∗ {\displaystyle A=U\Sigma V^{*}} is the singular value
  // decomposition of  A
  //  {\displaystyle A}, then  A + = V Σ + U ∗ {\displaystyle A^{+}=V\Sigma
  //  ^{+}U^{*}}.
  // if SVD(A) = UΣV^*  then A^+ = VΣ^+U^*. For a rectangular diagonal matrix
  // such as Σ {\displaystyle \Sigma }, we get the pseudoinverse by taking the
  // reciprocal of each non-zero element on the diagonal, leaving the zeros in
  // place, and then transposing the matrix

  // Is zero: // const auto outs = mlx::core::linalg::svd(a);
  // const auto& U = outs[0];
  // const auto& S = outs[1];
  // const auto& Vt_full = outs[2];

  // // lapack clobbers the input, so we have to make a copy.
  // array in(a.shape(), float32, nullptr, {});
  // copy(a, in, a.flags().row_contiguous ? CopyType::Vector : CopyType::General);

  // svd_impl(in, U, S, Vt_full);

  // auto Vt = slice(Vt_full, {0, 0}, {K, N});

  // auto S_pinv = diag(1.0 / S);
  // auto Ut = transpose(U);
  // auto V = transpose(Vt);
  // auto a_pinv = matmul(V, matmul(S_pinv, Ut));
  // // auto A_again = matmul(matmul(a, a_pinv), a);
  // // auto isClose = allclose(a, A_again).item<bool>();

  // pinv.set_data(allocator::malloc_or_wait(pinv.nbytes()));
  // pinv = matmul(V, matmul(S_pinv, Ut));
  // pinv = a_pinv;

  // const auto m = a.shape(-2);
  // const auto n = a.shape(-1);
  // const auto k = std::min(m, n);
  // const auto rank = a.ndim();

  // std::vector<int> u_shape = a.shape();
  // u_shape[rank - 2] = m;
  // u_shape[rank - 1] = m;

  
  // std::vector<int> s_shape = a.shape();
  // s_shape.pop_back();
  // s_shape[rank - 2] = std::min(m, n);

  // std::vector<int> vt_shape = a.shape();
  // vt_shape[rank - 2] = n;
  // vt_shape[rank - 1] = n;

  // // array u(std::vector<int>{m, m}, float32, nullptr, {});
  // // array s(std::vector<int>{k, 1}, float32, nullptr, {});
  // // array vt_full(std::vector<int>{n, n}, float32, nullptr, {});

  // array u(u_shape, float32, nullptr, {});
  // array s(s_shape, float32, nullptr, {});
  // array vt_full(vt_shape, float32, nullptr, {});

  // // const auto& U = outs[0];
  // // const auto& S = outs[1];
  // // const auto& Vt_full = outs[2];

  // // lapack clobbers the input, so we have to make a copy.
  // array in(a.shape(), float32, nullptr, {});
  // copy(a, in, a.flags().row_contiguous ? CopyType::Vector : CopyType::General);

  // svd_impl(in, u, s, vt_full);

  // auto vt = slice(vt_full, {0, 0}, {k, n});

  // const float *matrix = a.data<float>();

  // auto s_pinv = diag(1.0 / s);
  // auto ut = transpose(u);
  // auto v = transpose(vt);
  // auto a_pinv = matmul(v, matmul(s_pinv, ut));
  // // auto A_again = matmul(matmul(a, a_pinv), a);
  // // auto isClose = allclose(a, A_again).item<bool>();

  // pinv.set_data(allocator::malloc_or_wait(pinv.nbytes()));
  // pinv = matmul(v, matmul(s_pinv, ut));

  // /// Hack to a.t
  // auto at = transpose(a);
  // copy(at, pinv, a.flags().row_contiguous ? CopyType::Vector : CopyType::General);

  // const auto prng_key = random::key(42);
  // const auto rand_pinv = random::normal(a_pinv.shape(), prng_key);
  
  // //copy(rand_pinv, pinv, rand_pinv.flags().row_contiguous ? CopyType::Vector : CopyType::General);

  // // pinv = rand_pinv;

  /////////////////////////////////////////////// OLD COMMIT ////////////////////////
  
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

  // array u(u_shape, float32, nullptr, {});
  // array s(s_shape, float32, nullptr, {});
  // array vt(vt_shape, float32, nullptr, {});

  array u(u_shape, float32, nullptr, {});
  array s(s_shape, float32, nullptr, {});
  array vt(vt_shape, float32, nullptr, {});
  array result({N, K}, float32, nullptr, {});
  result.set_data(allocator::malloc_or_wait(result.nbytes()));

  // auto outs = linalg::svd(a);
  // auto U = outs[0];
  // auto S = outs[1];
  // auto Vt = outs[2];
  // auto V = slice(transpose(Vt), {0, 0}, {N, K});
  // result = matmul(V, matmul(diag(1.0 / S), transpose(U)));
  // pinv.set_data(allocator::malloc_or_wait(pinv.nbytes()));
  // pinv = result;

  // array v({N, K}, float32, nullptr, {});
  // auto x = slice(transpose(vt), {0, 0}, {N, K});

  // result = matmul(x, matmul(diag(1.0 / s), transpose(u)));

  // copy(slice(transpose(vt), {0, 0}, {N, K}), v, CopyType::General);
  // copy(vt, v, vt.flags().row_contiguous ? CopyType::Vector : CopyType::General);
  // v = transpose(v);

  // auto result = matmul(matmul(v_slice, diag(1.0/s)),  transpose(u));
  // // Σ^+ = 1 ./ Σ aka element-wise reciprocal of Σ diagonal matrix
  // // then, compute A^+ = VΣ^+U^*
  // // Work-in-progress
  // auto s_pinv = diag(1.0 / s);
  // // auto s_plus = 1.0 / s; // TODO: Only run on diagonal elements
  // // s_plus = diag(s_plus);
  // // s_plus = diagonal(s_plus);
  // auto v = transpose(vt);
  // auto v_slice = transpose(vt_slice);
  // auto u_slice_transpose = transpose(u_slice);
  // auto ut = transpose(u);
  // // auto inner = matmul(ut, s_plus);
  // auto result = matmul(v_slice, matmul(s_pinv, u_slice_transpose));
  // pinv = matmul(matmul(v, diag(1.0/s)),  transpose(u));


  // Compute SVD
  // LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda, s, u, ldu, vt, ldvt, superb);
  svd_impl(a, u, s, vt);

  // Invert the singular values
  float *s_data = s.data<float>();
  for (int i = 0; i < k; i++) {
      if (s_data[i] > 1e-10) {
          s_data[i] = 1.0 / s_data[i];
      } else {
          s_data[i] = 0.0;
      }
  }

  // Compute U * Sigma^+
  // A of shape M x N. The leading dimension is N since lapack receives Aᵀ.
  const int lda = n;
  // U of shape M x M. (N x N in lapack).
  const int ldu = n;
  // Vᵀ of shape N x N. (M x M in lapack).
  const int ldvt = m;


  float *u_data = u.data<float>();
  for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
          u_data[i * ldu + j] *= s_data[j];
      }
  }

  // Compute A^+ = V^T * (U * Sigma^+)
  pinv.set_data(allocator::malloc_or_wait(pinv.nbytes()));
  float *pinv_data = pinv.data<float>();
  float *vt_data = vt.data<float>();
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, m, k, 1.0, vt_data, ldvt, u_data, ldu, 0.0, pinv_data, m);

  // copy(vt, pinv, vt.flags().row_contiguous ? CopyType::Vector : CopyType::General);
}

void PseudoInverse::eval(const std::vector<array>& inputs, array& output) {
  pseudoinverse_impl(inputs[0], output);
}

} // namespace mlx::core