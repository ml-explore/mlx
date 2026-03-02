// Copyright © 2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/primitives.h"

namespace mlx::core {

template <typename T, class Enable = void>
struct SVDWork {};

template <typename T>
struct SVDWork<
    T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  using R = T;

  int N;
  int M;
  int K;
  int lda;
  int ldu;
  int ldvt;
  char jobz;
  std::vector<array::Data> buffers;
  int lwork;

  SVDWork(int N, int M, int K, char jobz)
      : N(N), M(M), K(K), lda(N), ldu(N), ldvt(M), jobz(jobz) {
    T workspace_dimension = 0;

    // Will contain the indices of eigenvectors that failed to converge (not
    // used here but required by lapack).
    buffers.emplace_back(allocator::malloc(sizeof(int) * 8 * K));

    int lwork_query = -1;
    int info;

    // Compute workspace size.
    gesdd<T>(
        /* jobz = */ &jobz,
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
        /* iwork = */ static_cast<int*>(buffers[0].buffer.raw_ptr()),
        /* info = */ &info);

    if (info != 0) {
      std::stringstream ss;
      ss << "[SVD::eval_cpu] workspace calculation failed with code " << info;
      throw std::runtime_error(ss.str());
    }

    lwork = workspace_dimension;
    buffers.emplace_back(allocator::malloc(sizeof(T) * lwork));
  }

  void run(T* a, R* s, T* u, T* vt) {
    int info;
    gesdd<T>(
        /* jobz = */ &jobz,
        // M and N are swapped since lapack expects column-major.
        /* m = */ &N,
        /* n = */ &M,
        /* a = */ a,
        /* lda = */ &lda,
        /* s = */ s,
        // According to the identity above, lapack will write Vᵀᵀ as U.
        /* u = */ u,
        /* ldu = */ &ldu,
        // According to the identity above, lapack will write Uᵀ as Vᵀ.
        /* vt = */ vt,
        /* ldvt = */ &ldvt,
        /* work = */ static_cast<T*>(buffers[1].buffer.raw_ptr()),
        /* lwork = */ &lwork,
        /* iwork = */ static_cast<int*>(buffers[0].buffer.raw_ptr()),
        /* info = */ &info);

    if (info != 0) {
      std::stringstream ss;
      ss << "svd_impl: sgesvdx_ failed with code " << info;
      throw std::runtime_error(ss.str());
    }
  }
};

template <>
struct SVDWork<std::complex<float>> {
  using T = std::complex<float>;
  using R = float;

  int N;
  int M;
  int K;
  int lda;
  int ldu;
  int ldvt;
  char jobz;
  std::vector<array::Data> buffers;
  int lwork;

  SVDWork(int N, int M, int K, char jobz)
      : N(N), M(M), K(K), lda(N), ldu(N), ldvt(M), jobz(jobz) {
    T workspace_dimension = 0;

    // Will contain the indices of eigenvectors that failed to converge (not
    // used here but required by lapack).
    buffers.emplace_back(allocator::malloc(sizeof(int) * 8 * K));

    const int lrwork =
        jobz == 'A' ? std::max(1, 5 * K * K + 5 * K) : std::max(1, 7 * K);
    buffers.emplace_back(allocator::malloc(sizeof(float) * lrwork));

    int lwork_query = -1;
    int work_query = -1;
    int info;

    // Compute workspace size.
    gesdd<T>(
        /* jobz = */ &jobz,
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
        /* rwork = */ static_cast<float*>(buffers[1].buffer.raw_ptr()),
        /* iwork = */ static_cast<int*>(buffers[0].buffer.raw_ptr()),
        /* info = */ &info);

    if (info != 0) {
      std::stringstream ss;
      ss << "[SVD::eval_cpu] workspace calculation failed with code " << info;
      throw std::runtime_error(ss.str());
    }

    lwork = workspace_dimension.real();
    buffers.emplace_back(allocator::malloc(sizeof(T) * lwork));
  }

  void run(T* a, R* s, T* u, T* vt) {
    int info;
    gesdd<T>(
        /* jobz = */ &jobz,
        // M and N are swapped since lapack expects column-major.
        /* m = */ &N,
        /* n = */ &M,
        /* a = */ a,
        /* lda = */ &lda,
        /* s = */ s,
        // According to the identity above, lapack will write Vᵀᵀ as U.
        /* u = */ u,
        /* ldu = */ &ldu,
        // According to the identity above, lapack will write Uᵀ as Vᵀ.
        /* vt = */ vt,
        /* ldvt = */ &ldvt,
        /* work = */ static_cast<T*>(buffers[2].buffer.raw_ptr()),
        /* lwork = */ &lwork,
        /* rwork = */ static_cast<float*>(buffers[1].buffer.raw_ptr()),
        /* iwork = */ static_cast<int*>(buffers[0].buffer.raw_ptr()),
        /* info = */ &info);

    if (info != 0) {
      std::stringstream ss;
      ss << "svd_impl: sgesvdx_ failed with code " << info;
      throw std::runtime_error(ss.str());
    }
  }
};

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

  using R = typename SVDWork<T>::R;

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
  R* s_ptr;
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

    s_ptr = s.data<R>();
    u_ptr = u.data<T>();
    vt_ptr = vt.data<T>();
  } else {
    array& s = outputs[0];

    s.set_data(allocator::malloc(s.nbytes()));

    encoder.set_output_array(s);

    s_ptr = s.data<R>();
    u_ptr = nullptr;
    vt_ptr = nullptr;
  }

  encoder.dispatch([in_ptr, u_ptr, s_ptr, vt_ptr, M, N, K, num_matrices]() {
    auto jobz = (u_ptr) ? 'A' : 'N';
    SVDWork<T> svd_work(N, M, K, jobz);
    // Loop over matrices.
    for (int i = 0; i < num_matrices; i++) {
      svd_work.run(
          in_ptr + M * N * i,
          s_ptr + K * i,
          vt_ptr ? vt_ptr + N * N * i : nullptr,
          u_ptr ? u_ptr + M * M * i : nullptr);
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
