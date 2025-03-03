// Copyright © 2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/primitives.h"

namespace mlx::core {

template <typename T>
void svd_impl(
    const array& a,
    std::vector<array>& outputs,
    bool compute_uv,
    Stream stream) {
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

  size_t num_matrices = a.size() / (M * N);

  // lapack clobbers the input, so we have to make a copy.
  array in(a.shape(), a.dtype(), nullptr, {});
  copy(
      a,
      in,
      a.flags().row_contiguous ? CopyType::Vector : CopyType::General,
      stream);

  // Allocate outputs.
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  auto in_ptr = in.data<T>();
  T* u_ptr;
  T* s_ptr;
  T* vt_ptr;

  if (compute_uv) {
    array& u = outputs[0];
    array& s = outputs[1];
    array& vt = outputs[2];

    u.set_data(allocator::malloc_or_wait(u.nbytes()));
    s.set_data(allocator::malloc_or_wait(s.nbytes()));
    vt.set_data(allocator::malloc_or_wait(vt.nbytes()));

    encoder.set_output_array(u);
    encoder.set_output_array(s);
    encoder.set_output_array(vt);

    s_ptr = s.data<T>();
    u_ptr = u.data<T>();
    vt_ptr = vt.data<T>();
  } else {
    array& s = outputs[0];

    s.set_data(allocator::malloc_or_wait(s.nbytes()));

    encoder.set_output_array(s);

    s_ptr = s.data<T>();
    u_ptr = nullptr;
    vt_ptr = nullptr;
  }

  encoder.dispatch([in_ptr, u_ptr, s_ptr, vt_ptr, M, N, K, num_matrices]() {
    // A of shape M x N. The leading dimension is N since lapack receives Aᵀ.
    const int lda = N;
    // U of shape M x M. (N x N in lapack).
    const int ldu = N;
    // Vᵀ of shape N x N. (M x M in lapack).
    const int ldvt = M;

    auto job_u = (u_ptr) ? "V" : "N";
    auto job_vt = (u_ptr) ? "V" : "N";
    static constexpr auto range = "A";

    // Will contain the number of singular values after the call has returned.
    int ns = 0;
    T workspace_dimension = 0;

    // Will contain the indices of eigenvectors that failed to converge (not
    // used here but required by lapack).
    auto iwork = array::Data{allocator::malloc_or_wait(sizeof(int) * 12 * K)};

    static const int lwork_query = -1;

    static const int ignored_int = 0;
    static const T ignored_float = 0;

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
          /* a = */ in_ptr + M * N * i,
          /* lda = */ &lda,
          /* vl = */ &ignored_float,
          /* vu = */ &ignored_float,
          /* il = */ &ignored_int,
          /* iu = */ &ignored_int,
          /* ns = */ &ns,
          /* s = */ s_ptr + K * i,
          // According to the identity above, lapack will write Vᵀᵀ as U.
          /* u = */ vt_ptr ? vt_ptr + N * N * i : nullptr,
          /* ldu = */ &ldu,
          // According to the identity above, lapack will write Uᵀ as Vᵀ.
          /* vt = */ u_ptr ? u_ptr + M * M * i : nullptr,
          /* ldvt = */ &ldvt,
          /* work = */ static_cast<T*>(scratch.buffer.raw_ptr()),
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
  });
  encoder.add_temporary(in);
}

template <typename T>
void compute_svd(
    const array& a,
    bool compute_uv,
    std::vector<array>& outputs,
    Stream stream) {}

void SVD::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  switch (inputs[0].dtype()) {
    case float32:
      svd_impl<float>(inputs[0], outputs, compute_uv_, stream());
      break;
    case float64:
      svd_impl<double>(inputs[0], outputs, compute_uv_, stream());
      break;
    default:
      throw std::runtime_error(
          "[SVD::eval_cpu] only supports float32 or float64.");
  }
}

} // namespace mlx::core
