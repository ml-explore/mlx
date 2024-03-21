// Copyright © 2023-2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/common/copy.h"
#include "mlx/linalg.h"
#include "mlx/primitives.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <lapack.h>
#endif

namespace mlx::core {

void inverse_impl(const array& a, array& inv) {
  // Lapack uses the column-major convention. We take advantage of the following
  // identity to avoid transposing (see
  // https://math.stackexchange.com/a/340234):
  //   (A⁻¹)ᵀ = (Aᵀ)⁻¹

  // The inverse is computed in place, so just copy the input to the output.
  copy(a, inv, a.flags().row_contiguous ? CopyType::Vector : CopyType::General);

  const int N = a.shape(-1);
  const size_t num_matrices = a.size() / (N * N);

  int info;
  auto ipiv = array::Data{allocator::malloc_or_wait(sizeof(int) * N)};

  for (int i = 0; i < num_matrices; i++) {
    // Compute LU factorization.
    sgetrf_(
        /* m = */ &N,
        /* n = */ &N,
        /* a = */ inv.data<float>() + N * N * i,
        /* lda = */ &N,
        /* ipiv = */ static_cast<int*>(ipiv.buffer.raw_ptr()),
        /* info = */ &info);

    if (info != 0) {
      std::stringstream ss;
      ss << "inverse_impl: LU factorization failed with error code " << info;
      throw std::runtime_error(ss.str());
    }

    static const int lwork_query = -1;
    float workspace_size = 0;

    // Compute workspace size.
    sgetri_(
        /* m = */ &N,
        /* a = */ nullptr,
        /* lda = */ &N,
        /* ipiv = */ nullptr,
        /* work = */ &workspace_size,
        /* lwork = */ &lwork_query,
        /* info = */ &info);

    if (info != 0) {
      std::stringstream ss;
      ss << "inverse_impl: LU workspace calculation failed with error code "
         << info;
      throw std::runtime_error(ss.str());
    }

    const int lwork = workspace_size;
    auto scratch =
        array::Data{allocator::malloc_or_wait(sizeof(float) * lwork)};

    // Compute inverse.
    sgetri_(
        /* m = */ &N,
        /* a = */ inv.data<float>() + N * N * i,
        /* lda = */ &N,
        /* ipiv = */ static_cast<int*>(ipiv.buffer.raw_ptr()),
        /* work = */ static_cast<float*>(scratch.buffer.raw_ptr()),
        /* lwork = */ &lwork,
        /* info = */ &info);

    if (info != 0) {
      std::stringstream ss;
      ss << "inverse_impl: inversion failed with error code " << info;
      throw std::runtime_error(ss.str());
    }
  }
}

void Inverse::eval(const std::vector<array>& inputs, array& output) {
  if (inputs[0].dtype() != float32) {
    throw std::runtime_error("[Inverse::eval] only supports float32.");
  }
  inverse_impl(inputs[0], output);
}

std::pair<std::vector<array>, std::vector<int>> Inverse::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {
  auto ax = axes[0] >= 0 ? 0 : -1;
  auto a = axes[0] > 0 ? moveaxis(inputs[0], axes[0], 0, stream()) : inputs[0];
  return {{linalg::inv(a, stream())}, {ax}};
}

} // namespace mlx::core
