// Copyright © 2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/primitives.h"

namespace mlx::core {

template <typename T>
void svd_impl(const array& a, T* u_data, T* s_data, T* vt_data) {
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
  array in(a.shape(), a.dtype(), nullptr, {});
  copy(a, in, a.flags().row_contiguous ? CopyType::Vector : CopyType::General);

  auto job_u = (u_data && vt_data) ? "V" : "N";
  auto job_vt = (u_data && vt_data) ? "V" : "N";
  static constexpr auto range = "A";

  // Will contain the number of singular values after the call has returned.
  int ns = 0;
  T workspace_dimension = 0;

  // Will contain the indices of eigenvectors that failed to converge (not used
  // here but required by lapack).
  auto iwork = array::Data{allocator::malloc_or_wait(sizeof(int) * 12 * K)};

  static const int lwork_query = -1;

  static const int ignored_int = 0;
  static const T ignored_float = 0;
  static T ignored_output = 0;

  int info;

  // Compute workspace size.
  gesvdx<T>(
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
    ss << "[SVD::eval_cpu] workspace calculation failed with code " << info;
    throw std::runtime_error(ss.str());
  }

  const int lwork = workspace_dimension;
  auto scratch = array::Data{allocator::malloc_or_wait(sizeof(T) * lwork)};

  // Loop over matrices.
  for (int i = 0; i < num_matrices; i++) {
    gesvdx<T>(
        /* jobu = */ job_u,
        /* jobvt = */ job_vt,
        /* range = */ range,
        // M and N are swapped since lapack expects column-major.
        /* m = */ &N,
        /* n = */ &M,
        /* a = */ in.data<T>() + M * N * i,
        /* lda = */ &lda,
        /* vl = */ &ignored_float,
        /* vu = */ &ignored_float,
        /* il = */ &ignored_int,
        /* iu = */ &ignored_int,
        /* ns = */ &ns,
        /* s = */ s_data + K * i,
        // According to the identity above, lapack will write Vᵀᵀ as U.
        /* u = */ vt_data ? vt_data + N * N * i : &ignored_output,
        /* ldu = */ &ldu,
        // According to the identity above, lapack will write Uᵀ as Vᵀ.
        /* vt = */ u_data ? u_data + M * M * i : &ignored_output,
        /* ldvt = */ &ldvt,
        /* work = */ static_cast<T*>(scratch.buffer.raw_ptr()),
        /* lwork = */ &lwork,
        /* iwork = */ static_cast<int*>(iwork.buffer.raw_ptr()),
        /* info = */ &info);

    if (info != 0) {
      std::stringstream ss;
      ss << "[SVD::eval_cpu] failed with code " << info;
      throw std::runtime_error(ss.str());
    }

    if (ns != K) {
      std::stringstream ss;
      ss << "[SVD::eval_cpu] expected " << K << " singular values, but " << ns
         << " were computed.";
      throw std::runtime_error(ss.str());
    }
  }
}

template <typename T>
void compute_svd(const array& a, bool compute_uv, std::vector<array>& outputs) {
  if (compute_uv) {
    array& u = outputs[0];
    array& s = outputs[1];
    array& vt = outputs[2];

    u.set_data(allocator::malloc_or_wait(u.nbytes()));
    s.set_data(allocator::malloc_or_wait(s.nbytes()));
    vt.set_data(allocator::malloc_or_wait(vt.nbytes()));

    svd_impl<T>(a, u.data<T>(), s.data<T>(), vt.data<T>());
  } else {
    array& s = outputs[0];

    s.set_data(allocator::malloc_or_wait(s.nbytes()));

    svd_impl<T>(a, nullptr, s.data<T>(), nullptr);
  }
}

void SVD::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  switch (inputs[0].dtype()) {
    case float32:
      compute_svd<float>(inputs[0], compute_uv_, outputs);
      break;
    case float64:
      compute_svd<double>(inputs[0], compute_uv_, outputs);
      break;
    default:
      throw std::runtime_error(
          "[SVD::eval_cpu] only supports float32 or float64.");
  }
}

} // namespace mlx::core
