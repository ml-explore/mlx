// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/svd.h"
#include "mlx/primitives.h"
#include "mlx/ops.h"

// #include "mlx/primitives.h"
// #include "mlx/fast_primitives.h"

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

  // Rows and cols of the original matrix in row-major order.
  const int M = a.shape(-2);
  const int N = a.shape(-1);
  const int K = std::min(M, N);

  const int pinv_shape_M = pinv.shape(-2);
  const int pinv_shape_N = pinv.shape(-1);
  if (pinv_shape_N != M) { // Failing because pinv is not 5x4 -- it is showing up as 4x5
    std::stringstream ss;
    ss << "unexpected N for pinv; got N " << pinv_shape_N << " but expected N is " << N;
    throw std::runtime_error(ss.str());
  }

  // A of shape M x N. The leading dimension is N since lapack receives Aᵀ.
  const int lda = N;
  // U of shape M x M. (N x N in lapack).
  const int ldu = N;
  // Vᵀ of shape N x N. (M x M in lapack).
  const int ldvt = M;

  array u(std::vector<int>{M, M}, float32, nullptr, {});
  array s(std::vector<int>{K, K}, float32, nullptr, {});
  array vt(std::vector<int>{M, N}, float32, nullptr, {});

  svd_impl(a, u, s, vt);

  auto u_slice = slice(u, {0, 0}, {u.shape(0), s.shape(0)});
  auto vt_slice = slice(vt, {0, 0}, {M, K});

  // Σ^+ = 1 ./ Σ aka element-wise reciprocal of Σ diagonal matrix
  // then, compute A^+ = VΣ^+U^*
  // Work-in-progress
  auto s_plus = 1.0 / s; // TODO: Only run on diagonal elements
  // s_plus = diag(s_plus);
  // s_plus = diagonal(s_plus);
  auto v = transpose(vt);
  auto v_slice = transpose(vt_slice);
  auto u_slice_transpose = transpose(u_slice);
  auto ut = transpose(u);
  // auto inner = matmul(ut, s_plus);
  auto result = matmul(v_slice, matmul(s_plus, u_slice_transpose));
  //.. result = matmul(  5x4  , matmul( 4x4  ,     4x4    ));


  result = matmul(v, matmul(s_plus, ut));

  // u   is  4x5
  // s   is  4x4
  // vt  is  5x5
  // s_pinv is 4x4

  // ut  is  5x4
  // v   is  5x5

  // need: V @ S_pinv @ Ut
  // From Python / NumPy
  // >>> U.shape
  // (4, 4)

  // >>> S.shape
  // (4,)

  // >>> Vt.shape
  // (4, 5)

  // >>> A_pinv = Vt.T @ S_pinv @ U.T
  // >>> 5x4 =    5x4 @  4x4   @ 4x4
  
// Actual: 
// A_pinv = Vt.T @ S_pinv @ U.T
// matmul(matmul(v, s_pinv), ut)

  pinv.set_data(allocator::malloc_or_wait(pinv.nbytes()));
  pinv = result;
}

void PseudoInverse::eval(const std::vector<array>& inputs, array& output) {
  pseudoinverse_impl(inputs[0], output);
}

} // namespace mlx::core