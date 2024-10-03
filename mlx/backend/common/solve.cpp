// Copyright © 2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/lapack_helper.h"
#include "mlx/primitives.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#else
#include <lapack.h>
#endif

#include <cassert>

namespace mlx::core {

namespace {

// Wrapper to account for differences in
// LAPACK implementations (basically how to pass the 'trans' string to fortran).
int sgetrs_wrapper(char trans, int N, int NRHS, int* ipiv, float* a, float* b) {
  int info;

#ifdef LAPACK_FORTRAN_STRLEN_END
  sgetrs_(
      /* trans */ &trans,
      /* n */ &N,
      /* nrhs */ &NRHS,
      /* a */ a,
      /* lda */ &N,
      /* ipiv */ ipiv,
      /* b */ b,
      /* ldb */ &N,
      /* info */ &info,
      /* trans_len = */ static_cast<size_t>(1));
#else
  sgetrs_(
      /* trans */ &trans,
      /* n */ &N,
      /* nrhs */ &NRHS,
      /* a */ a,
      /* lda */ &N,
      /* ipiv */ ipiv,
      /* b */ b,
      /* ldb */ &N,
      /* info */ &info);
#endif

  return info;
}

} // namespace

void solve_impl(const array& a, const array& b, array& out) {
  int N = a.shape(-2);
  int NRHS = out.shape(-1);
  std::vector<int> ipiv(N);

  // copy b into out and make it col-contiguous
  auto flags = out.flags();
  flags.col_contiguous = true;
  flags.row_contiguous = false;
  std::vector<size_t> strides(a.ndim(), 0);
  std::copy(out.strides().begin(), out.strides().end(), strides.begin());
  strides[a.ndim() - 2] = 1;
  strides[a.ndim() - 1] = N;

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
      ss << "solve_impl: sgetrf_ failed with code " << info
         << ((info > 0) ? " because matrix is singular"
                        : " becuase argument had an illegal value");
      throw std::runtime_error(ss.str());
    }

    static constexpr char trans = 'T';
    // Solve the system using the LU factors from sgetrf
    info = sgetrs_wrapper(trans, N, NRHS, ipiv_ptr, a_ptr, out_ptr);

    if (info != 0) {
      std::stringstream ss;
      ss << "solve_impl: sgetrs_ failed with code " << info;
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
  if (inputs[0].dtype() != float32 || inputs[1].dtype() != float32) {
    throw std::runtime_error("[Solve::eval] only supports float32.");
  }
  solve_impl(inputs[0], inputs[1], outputs[0]);
}

} // namespace mlx::core
