// Copyright © 2023-2024 Apple Inc.

#include "mlx/primitives.h"
#include "mlx/fast_primitives.h"

namespace mlx::core {

void pseudoinverse_impl(const array& a, array& pinv) {
    // If  A = U Σ V ∗ {\displaystyle A=U\Sigma V^{*}} is the singular value decomposition of  A
    //  {\displaystyle A}, then  A + = V Σ + U ∗ {\displaystyle A^{+}=V\Sigma ^{+}U^{*}}.
    // if SVD(A) = UΣV^*  then A^+ = VΣ^+U^*. For a rectangular diagonal matrix
    // such as Σ {\displaystyle \Sigma }, we get the pseudoinverse by taking the reciprocal of each
    // non-zero element on the diagonal, leaving the zeros in place, and then transposing the matrix

    // Rows and cols of the original matrix in row-major order.
    const int M = a.shape(-2);
    const int N = a.shape(-1);
    const int K = std::min(M, N);

    // A of shape M x N. The leading dimension is N since lapack receives Aᵀ.
    const int lda = N;
    // U of shape M x M. (N x N in lapack).
    const int ldu = N;
    // Vᵀ of shape N x N. (M x M in lapack).
    const int ldvt = M;

    // lapack clobbers the input, so we have to make a copy.
    // auto u_shape = std::vector<int>{M, N};
    array u(std::vector<int>{M, N}, float32, nullptr, {});
    array s(std::vector<int>{K, K}, float32, nullptr, {});
    array vt(std::vector<int>{N, N}, float32, nullptr, {});

    // TODO: Perhaps use svd_impl(a, u, s, vt); 
    // and then compute A^+ = VΣ^+U^*
}

void PseudoInverse::eval(const std::vector<array>& inputs, array& output) {
    pseudoinverse_impl(inputs[0], output);
}

} // namespace mlx::core