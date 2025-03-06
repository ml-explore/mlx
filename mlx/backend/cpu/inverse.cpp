// Copyright © 2023-2024 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/lapack.h"
#include "mlx/primitives.h"

namespace mlx::core {

template <typename T>
void general_inv(array& inv, int N, int i) {
  int info;
  auto ipiv = array::Data{allocator::malloc_or_wait(sizeof(int) * N)};
  // Compute LU factorization.
  getrf<T>(
      /* m = */ &N,
      /* n = */ &N,
      /* a = */ inv.data<T>() + N * N * i,
      /* lda = */ &N,
      /* ipiv = */ static_cast<int*>(ipiv.buffer.raw_ptr()),
      /* info = */ &info);

  if (info != 0) {
    std::stringstream ss;
    ss << "[Inverse::eval_cpu] LU factorization failed with error code "
       << info;
    throw std::runtime_error(ss.str());
  }

  static const int lwork_query = -1;
  T workspace_size = 0;

  // Compute workspace size.
  getri<T>(
      /* m = */ &N,
      /* a = */ nullptr,
      /* lda = */ &N,
      /* ipiv = */ nullptr,
      /* work = */ &workspace_size,
      /* lwork = */ &lwork_query,
      /* info = */ &info);

  if (info != 0) {
    std::stringstream ss;
    ss << "[Inverse::eval_cpu] LU workspace calculation failed with error code "
       << info;
    throw std::runtime_error(ss.str());
  }

  const int lwork = workspace_size;
  auto scratch = array::Data{allocator::malloc_or_wait(sizeof(T) * lwork)};

  // Compute inverse.
  getri<T>(
      /* m = */ &N,
      /* a = */ inv.data<T>() + N * N * i,
      /* lda = */ &N,
      /* ipiv = */ static_cast<int*>(ipiv.buffer.raw_ptr()),
      /* work = */ static_cast<T*>(scratch.buffer.raw_ptr()),
      /* lwork = */ &lwork,
      /* info = */ &info);

  if (info != 0) {
    std::stringstream ss;
    ss << "[Inverse::eval_cpu] inversion failed with error code " << info;
    throw std::runtime_error(ss.str());
  }
}

template <typename T>
void tri_inv(array& inv, int N, int i, bool upper) {
  const char uplo = upper ? 'L' : 'U';
  const char diag = 'N';
  T* data = inv.data<T>() + N * N * i;
  int info;
  trtri<T>(
      /* uplo = */ &uplo,
      /* diag = */ &diag,
      /* N = */ &N,
      /* a = */ data,
      /* lda = */ &N,
      /* info = */ &info);

  // zero out the other triangle
  if (upper) {
    for (int i = 0; i < N; i++) {
      std::fill(data, data + i, 0.0f);
      data += N;
    }
  } else {
    for (int i = 0; i < N; i++) {
      std::fill(data + i + 1, data + N, 0.0f);
      data += N;
    }
  }

  if (info != 0) {
    std::stringstream ss;
    ss << "[Inverse::eval_cpu] triangular inversion failed with error code "
       << info;
    throw std::runtime_error(ss.str());
  }
}

template <typename T>
void inverse_impl(
    const array& a,
    array& inv,
    bool tri,
    bool upper,
    Stream stream) {
  // Lapack uses the column-major convention. We take advantage of the following
  // identity to avoid transposing (see
  // https://math.stackexchange.com/a/340234):
  //   (A⁻¹)ᵀ = (Aᵀ)⁻¹

  // The inverse is computed in place, so just copy the input to the output.
  copy(
      a,
      inv,
      a.flags().row_contiguous ? CopyType::Vector : CopyType::General,
      stream);

  const int N = a.shape(-1);
  const size_t num_matrices = a.size() / (N * N);

  for (int i = 0; i < num_matrices; i++) {
    if (tri) {
      tri_inv<T>(inv, N, i, upper);
    } else {
      general_inv<T>(inv, N, i);
    }
  }
}

void Inverse::eval_cpu(const std::vector<array>& inputs, array& output) {
  switch (inputs[0].dtype()) {
    case float32:
      inverse_impl<float>(inputs[0], output, tri_, upper_, stream());
      break;
    case float64:
      inverse_impl<double>(inputs[0], output, tri_, upper_, stream());
      break;
    default:
      throw std::runtime_error(
          "[Inverse::eval_cpu] only supports float32 or float64.");
  }
}

} // namespace mlx::core
