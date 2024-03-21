// Copyright © 2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/lapack_helper.h"
#include "mlx/linalg.h"
#include "mlx/primitives.h"

namespace mlx::core {

void svd_impl(const array& a, array& u, array& s, array& vt) {
  // Lapack uses the column-major convention. To avoid having to transpose
  // the input and then transpose the outputs, we swap the indices/sizes of the
  // matrices and take advantage of the following identity (see
  // https://math.stackexchange.com/a/30077)
  //    A = UΣVᵀ
  //    Aᵀ = VΣUᵀ
  // As a result some of the indices/sizes are swapped as noted above.

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

  size_t num_matrices = a.size() / (M * N);

  // lapack clobbers the input, so we have to make a copy.
  array in(a.shape(), float32, nullptr, {});
  copy(a, in, a.flags().row_contiguous ? CopyType::Vector : CopyType::General);

  // Allocate outputs.
  u.set_data(allocator::malloc_or_wait(u.nbytes()));
  s.set_data(allocator::malloc_or_wait(s.nbytes()));
  vt.set_data(allocator::malloc_or_wait(vt.nbytes()));

  static constexpr auto job_u = "V";
  static constexpr auto job_vt = "V";
  static constexpr auto range = "A";

  // Will contain the number of singular values after the call has returned.
  int ns = 0;
  float workspace_dimension = 0;

  // Will contain the indices of eigenvectors that failed to converge (not used
  // here but required by lapack).
  auto iwork = array::Data{allocator::malloc_or_wait(sizeof(int) * 12 * K)};

  static const int lwork_query = -1;

  static const int ignored_int = 0;
  static const float ignored_float = 0;

  int info;

  // Compute workspace size.
  MLX_LAPACK_FUNC(sgesvdx)
  (
      /* jobu = */ job_u,
      /* jobvt = */ job_vt,
      /* range = */ range,
      // M and N are swapped since lapack expects column-major.
      /* m = */ &N,
      /* n = */ &M,
      /* a = */ nullptr,
      /* lda = */ &lda,
      /* vl = */ &ignored_float,
      /* vu = */ &ignored_float,
      /* il = */ &ignored_int,
      /* iu = */ &ignored_int,
      /* ns = */ &ns,
      /* s = */ nullptr,
      /* u = */ nullptr,
      /* ldu = */ &ldu,
      /* vt = */ nullptr,
      /* ldvt = */ &ldvt,
      /* work = */ &workspace_dimension,
      /* lwork = */ &lwork_query,
      /* iwork = */ static_cast<int*>(iwork.buffer.raw_ptr()),
      /* info = */ &info);

  if (info != 0) {
    std::stringstream ss;
    ss << "svd_impl: sgesvdx_ workspace calculation failed with code " << info;
    throw std::runtime_error(ss.str());
  }

  const int lwork = workspace_dimension;
  auto scratch = array::Data{allocator::malloc_or_wait(sizeof(float) * lwork)};

  // Loop over matrices.
  for (int i = 0; i < num_matrices; i++) {
    MLX_LAPACK_FUNC(sgesvdx)
    (
        /* jobu = */ job_u,
        /* jobvt = */ job_vt,
        /* range = */ range,
        // M and N are swapped since lapack expects column-major.
        /* m = */ &N,
        /* n = */ &M,
        /* a = */ in.data<float>() + M * N * i,
        /* lda = */ &lda,
        /* vl = */ &ignored_float,
        /* vu = */ &ignored_float,
        /* il = */ &ignored_int,
        /* iu = */ &ignored_int,
        /* ns = */ &ns,
        /* s = */ s.data<float>() + K * i,
        // According to the identity above, lapack will write Vᵀᵀ as U.
        /* u = */ vt.data<float>() + N * N * i,
        /* ldu = */ &ldu,
        // According to the identity above, lapack will write Uᵀ as Vᵀ.
        /* vt = */ u.data<float>() + M * M * i,
        /* ldvt = */ &ldvt,
        /* work = */ static_cast<float*>(scratch.buffer.raw_ptr()),
        /* lwork = */ &lwork,
        /* iwork = */ static_cast<int*>(iwork.buffer.raw_ptr()),
        /* info = */ &info);

    if (info != 0) {
      std::stringstream ss;
      ss << "svd_impl: sgesvdx_ failed with code " << info;
      throw std::runtime_error(ss.str());
    }

    if (ns != K) {
      std::stringstream ss;
      ss << "svd_impl: expected " << K << " singular values, but " << ns
         << " were computed.";
      throw std::runtime_error(ss.str());
    }
  }
}

void SVD::eval(const std::vector<array>& inputs, std::vector<array>& outputs) {
  if (!(inputs[0].dtype() == float32)) {
    throw std::runtime_error("[SVD::eval] only supports float32.");
  }
  svd_impl(inputs[0], outputs[0], outputs[1], outputs[2]);
}

std::pair<std::vector<array>, std::vector<int>> SVD::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto ax = axes[0] >= 0 ? 0 : -1;
  auto a = axes[0] > 0 ? moveaxis(inputs[0], axes[0], 0, stream()) : inputs[0];
  return {{linalg::svd(a, stream())}, {ax, ax, ax}};
}

} // namespace mlx::core
