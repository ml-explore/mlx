// Copyright © 2024 Apple Inc.

#include <cassert>

#include "mlx/allocator.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/lapack.h"
#include "mlx/primitives.h"

namespace mlx::core {

void solve_impl(const array& a, const array& b, array& out) {
  int N = a.shape(-2);
  int NRHS = out.shape(-1);
  std::vector<int> ipiv(N);

  // copy b into out and make it col-contiguous
  auto flags = out.flags();
  auto ndim = b.ndim();
  flags.col_contiguous = ndim <= 2;
  flags.row_contiguous = false;
  flags.contiguous = true;
  auto strides = out.strides();
  if (ndim >= 2) {
    std::swap(strides[ndim - 1], strides[ndim - 2]);
  }

  out.set_data(
      allocator::malloc_or_wait(out.nbytes()), out.nbytes(), strides, flags);
  copy_inplace(b, out, CopyType::GeneralGeneral);

  // lapack clobbers the input, so we have to make a copy. the copy doesn't need
  // to be col-contiguous because sgetrs has a transpose parameter (trans='T').
  array a_cpy(a.shape(), float32, nullptr, {});
  copy(
      a,
      a_cpy,
      a.flags().row_contiguous ? CopyType::Vector : CopyType::General);

  float* a_ptr = a_cpy.data<float>();
  float* out_ptr = out.data<float>();
  int* ipiv_ptr = ipiv.data();

  int info;
  size_t num_matrices = a.size() / (N * N);
  for (size_t i = 0; i < num_matrices; i++) {
    // Compute LU factorization of A
    MLX_LAPACK_FUNC(sgetrf)
    (/* m */ &N,
     /* n */ &N,
     /* a */ a_ptr,
     /* lda */ &N,
     /* ipiv */ ipiv_ptr,
     /* info */ &info);

    if (info != 0) {
      std::stringstream ss;
      ss << "[Solve::eval_cpu] sgetrf_ failed with code " << info
         << ((info > 0) ? " because matrix is singular"
                        : " becuase argument had an illegal value");
      throw std::runtime_error(ss.str());
    }

    // Solve the system using the LU factors from sgetrf
    static constexpr char trans = 'T';
    MLX_LAPACK_FUNC(sgetrs)
    (
        /* trans */ &trans,
        /* n */ &N,
        /* nrhs */ &NRHS,
        /* a */ a_ptr,
        /* lda */ &N,
        /* ipiv */ ipiv_ptr,
        /* b */ out_ptr,
        /* ldb */ &N,
        /* info */ &info);

    if (info != 0) {
      std::stringstream ss;
      ss << "[Solve::eval_cpu] sgetrs_ failed with code " << info;
      throw std::runtime_error(ss.str());
    }

    // Advance pointers to the next matrix
    a_ptr += N * N;
    out_ptr += N * NRHS;
  }
}

void Solve::eval(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  solve_impl(inputs[0], inputs[1], outputs[0]);
}

} // namespace mlx::core
