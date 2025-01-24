// Copyright © 2023-2024 Apple Inc.

#include <cstring>

#include "mlx/array.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/lapack.h"
#include "mlx/backend/common/utils.h"
#include "mlx/primitives.h"

#define DEFAULT(primitive)                                                 \
  void primitive::eval_cpu(const std::vector<array>& inputs, array& out) { \
    primitive::eval(inputs, out);                                          \
  }

#define DEFAULT_MULTI(primitive)                                       \
  void primitive::eval_cpu(                                            \
      const std::vector<array>& inputs, std::vector<array>& outputs) { \
    primitive::eval(inputs, outputs);                                  \
  }

namespace mlx::core {

DEFAULT(Convolution)
DEFAULT(QuantizedMatmul)
DEFAULT(Reduce)
DEFAULT(Scan)
DEFAULT(Softmax)

namespace {

inline void matmul_common_general(
    const array& a_pre,
    const array& b_pre,
    array& out,
    float alpha = 1.0f,
    float beta = 0.0f) {
  auto check_transpose = [](const array& arr) {
    auto stx = arr.strides()[arr.ndim() - 2];
    auto sty = arr.strides()[arr.ndim() - 1];
    if (stx == arr.shape(-1) && sty == 1) {
      return std::make_tuple(false, stx, arr);
    } else if (stx == 1 && sty == arr.shape(-2)) {
      return std::make_tuple(true, sty, arr);
    } else {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy(arr, arr_copy, CopyType::General);
      stx = arr.shape(-1);
      return std::make_tuple(false, stx, arr_copy);
    }
  };

  auto [a_transposed, lda, a] = check_transpose(a_pre);
  auto [b_transposed, ldb, b] = check_transpose(b_pre);
  size_t M = a.shape(-2);
  size_t N = b.shape(-1);
  size_t K = a.shape(-1);
  if (M == 0 || N == 0) {
    return;
  }
  if (K == 0) {
    std::memset(static_cast<void*>(out.data<float>()), 0, out.nbytes());
    return;
  }

  for (int i = 0; i < (a.size() / (M * K)); ++i) {
    cblas_sgemm(
        CblasRowMajor,
        a_transposed ? CblasTrans : CblasNoTrans, // transA
        b_transposed ? CblasTrans : CblasNoTrans, // transB
        M,
        N,
        K,
        alpha, // alpha
        a.data<float>() + elem_to_loc(M * K * i, a.shape(), a.strides()),
        lda,
        b.data<float>() + elem_to_loc(K * N * i, b.shape(), b.strides()),
        ldb,
        beta, // beta
        out.data<float>() + M * N * i,
        out.shape(-1) // ldc
    );
  }
}

} // namespace

void Matmul::eval_cpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[Matmul::eval_cpu] Currently only supports float32.");
  }
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  return matmul_common_general(inputs[0], inputs[1], out);
}

void AddMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[AddMM::eval_cpu] Currently only supports float32.");
  }

  // Fill output with C
  auto& c = inputs[2];
  CopyType ctype = c.data_size() == 1 ? CopyType::Scalar : CopyType::General;
  copy(c, out, ctype);

  return matmul_common_general(inputs[0], inputs[1], out, alpha_, beta_);
}

} // namespace mlx::core
