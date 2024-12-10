// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/allocator.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/lapack.h"
#include "mlx/primitives.h"

namespace mlx::core {

void lu_factor_impl(const array& a, array& lu, array& pivots) {
  int M = a.shape(-2);
  int N = a.shape(-1);

  // Copy a into lu and make it col contiguous
  auto ndim = lu.ndim();
  auto flags = lu.flags();
  flags.col_contiguous = ndim == 2;
  flags.row_contiguous = false;
  flags.contiguous = true;
  auto strides = lu.strides();
  strides[ndim - 1] = M;
  strides[ndim - 2] = 1;
  lu.set_data(
      allocator::malloc_or_wait(lu.nbytes()), lu.nbytes(), strides, flags);
  copy_inplace(
      a, lu, a.shape(), a.strides(), strides, 0, 0, CopyType::GeneralGeneral);

  float* a_ptr = lu.data<float>();

  pivots.set_data(allocator::malloc_or_wait(pivots.nbytes()));
  int* pivots_ptr = pivots.data<int>();

  int info;
  size_t num_matrices = a.size() / (M * N);
  for (size_t i = 0; i < num_matrices; ++i) {
    // Compute LU factorization of A
    MLX_LAPACK_FUNC(sgetrf)
    (/* m */ &M,
     /* n */ &N,
     /* a */ a_ptr,
     /* lda */ &M,
     /* ipiv */ pivots_ptr,
     /* info */ &info);

    if (info != 0) {
      std::stringstream ss;
      ss << "[LUF::eval_cpu] sgetrf_ failed with code " << info
         << ((info > 0) ? " because matrix is singular"
                        : " because argument had an illegal value");
      throw std::runtime_error(ss.str());
    }

    // Advance pointers to the next matrix
    a_ptr += M * N;
    pivots_ptr += pivots.shape(-1);
  }
}

void LUF::eval(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  lu_factor_impl(inputs[0], outputs[0], outputs[1]);
}

} // namespace mlx::core
