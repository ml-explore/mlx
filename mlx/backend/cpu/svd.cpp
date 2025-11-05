// Copyright © 2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/primitives.h"

namespace mlx::core {

// Wrapper for cgesdd
inline void cgesdd_wrapper(
    const char* jobz,
    const int* m,
    const int* n,
    std::complex<float>* a,
    const int* lda,
    float* s,
    std::complex<float>* u,
    const int* ldu,
    std::complex<float>* vt,
    const int* ldvt,
    std::complex<float>* work,
    const int* lwork,
    float* rwork,
    int* iwork,
    int* info) {
#ifdef MLX_USE_ACCELERATE
  cgesdd_(
      jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
#else
  MLX_LAPACK_FUNC(cgesdd)(
      jobz, m, n, a, lda, s, u, ldu, vt, ldvt, work, lwork, rwork, iwork, info);
#endif
}

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
  copy_cpu(
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

    u.set_data(allocator::malloc(u.nbytes()));
    s.set_data(allocator::malloc(s.nbytes()));
    vt.set_data(allocator::malloc(vt.nbytes()));

    encoder.set_output_array(u);
    encoder.set_output_array(s);
    encoder.set_output_array(vt);

    s_ptr = s.data<T>();
    u_ptr = u.data<T>();
    vt_ptr = vt.data<T>();
  } else {
    array& s = outputs[0];

    s.set_data(allocator::malloc(s.nbytes()));

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

    auto jobz = (u_ptr) ? "A" : "N";

    T workspace_dimension = 0;

    // Will contain the indices of eigenvectors that failed to converge (not
    // used here but required by lapack).
    auto iwork = array::Data{allocator::malloc(sizeof(int) * 8 * K)};

    static const int lwork_query = -1;

    int info;

    // Compute workspace size.
    gesdd<T>(
        /* jobz = */ jobz,
        // M and N are swapped since lapack expects column-major.
        /* m = */ &N,
        /* n = */ &M,
        /* a = */ nullptr,
        /* lda = */ &lda,
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
    auto scratch = array::Data{allocator::malloc(sizeof(T) * lwork)};

    // Loop over matrices.
    for (int i = 0; i < num_matrices; i++) {
      gesdd<T>(
          /* jobz = */ jobz,
          // M and N are swapped since lapack expects column-major.
          /* m = */ &N,
          /* n = */ &M,
          /* a = */ in_ptr + M * N * i,
          /* lda = */ &lda,
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

template <>
void svd_impl<std::complex<float>>(
    const array& a,
    std::vector<array>& outputs,
    bool compute_uv,
    Stream stream) {
  using CT = std::complex<float>;
  using RT = float;

  const int M = a.shape(-2);
  const int N = a.shape(-1);
  const int K = std::min(M, N);

  size_t num_matrices = a.size() / (M * N);

  array in(a.shape(), a.dtype(), nullptr, {});
  copy_cpu(
      a,
      in,
      a.flags().row_contiguous ? CopyType::Vector : CopyType::General,
      stream);

  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  auto in_ptr = in.data<complex64_t>();
  complex64_t* u_ptr;
  float* s_ptr;
  complex64_t* vt_ptr;

  if (compute_uv) {
    array& u = outputs[0];
    array& s = outputs[1];
    array& vt = outputs[2];

    u.set_data(allocator::malloc(u.nbytes()));
    s.set_data(allocator::malloc(s.nbytes()));
    vt.set_data(allocator::malloc(vt.nbytes()));

    encoder.set_output_array(u);
    encoder.set_output_array(s);
    encoder.set_output_array(vt);

    s_ptr = s.data<float>();
    u_ptr = u.data<complex64_t>();
    vt_ptr = vt.data<complex64_t>();
  } else {
    array& s = outputs[0];

    s.set_data(allocator::malloc(s.nbytes()));

    encoder.set_output_array(s);

    s_ptr = s.data<float>();
    u_ptr = nullptr;
    vt_ptr = nullptr;
  }

  encoder.dispatch([in_ptr, u_ptr, s_ptr, vt_ptr, M, N, K, num_matrices]() {
    const int lda = N;
    const int ldu = N;
    const int ldvt = M;

    auto jobz = (u_ptr) ? "A" : "N";

    std::complex<float> workspace_dimension = 0;

    auto iwork = array::Data{allocator::malloc(sizeof(int) * 8 * K)};

    static const int lwork_query = -1;

    int info;

    std::complex<float> work_query;
    const int min_mn = std::min(M, N);
    const int lrwork = (u_ptr) ? std::max(1, 5 * min_mn * min_mn + 5 * min_mn)
                               : std::max(1, 7 * min_mn);
    auto rwork = array::Data{allocator::malloc(sizeof(float) * lrwork)};

    cgesdd_wrapper(
        /* jobz = */ jobz,
        // M and N are swapped since lapack expects column-major.
        /* m = */ &N,
        /* n = */ &M,
        /* a = */ nullptr,
        /* lda = */ &lda,
        /* s = */ nullptr,
        /* u = */ nullptr,
        /* ldu = */ &ldu,
        /* vt = */ nullptr,
        /* ldvt = */ &ldvt,
        /* work = */ &work_query,
        /* lwork = */ &lwork_query,
        /* rwork = */ static_cast<float*>(rwork.buffer.raw_ptr()),
        /* iwork = */ static_cast<int*>(iwork.buffer.raw_ptr()),
        /* info = */ &info);

    if (info != 0) {
      std::stringstream ss;
      ss << "[SVD::eval_cpu] workspace calculation failed with code " << info;
      throw std::runtime_error(ss.str());
    }

    const int lwork = static_cast<int>(work_query.real());
    auto scratch =
        array::Data{allocator::malloc(sizeof(std::complex<float>) * lwork)};

    for (int i = 0; i < num_matrices; i++) {
      cgesdd_wrapper(
          /* jobz = */ jobz,
          // M and N are swapped since lapack expects column-major.
          /* m = */ &N,
          /* n = */ &M,
          /* a = */ reinterpret_cast<std::complex<float>*>(in_ptr) + M * N * i,
          /* lda = */ &lda,
          /* s = */ s_ptr + K * i,
          // According to the identity above, lapack will write Vᵀᵀ as U.
          /* u = */ reinterpret_cast<std::complex<float>*>(vt_ptr) + N * N * i,
          /* ldu = */ &ldu,
          // According to the identity above, lapack will write Uᵀ as Vᵀ.
          /* vt = */ reinterpret_cast<std::complex<float>*>(u_ptr) + M * M * i,
          /* ldvt = */ &ldvt,
          /* work = */
          reinterpret_cast<std::complex<float>*>(scratch.buffer.raw_ptr()),
          /* lwork = */ &lwork,
          /* rwork = */ static_cast<float*>(rwork.buffer.raw_ptr()),
          /* iwork = */ static_cast<int*>(iwork.buffer.raw_ptr()),
          /* info = */ &info);

      if (info != 0) {
        std::stringstream ss;
        ss << "svd_impl: cgesdd failed with code " << info;
        throw std::runtime_error(ss.str());
      }
    }
  });
  encoder.add_temporary(in);
}

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
    case complex64:
      svd_impl<std::complex<float>>(inputs[0], outputs, compute_uv_, stream());
      break;
    default:
      throw std::runtime_error(
          "[SVD::eval_cpu] only supports float32, float64, or complex64.");
  }
}

} // namespace mlx::core
