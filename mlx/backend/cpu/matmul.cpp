// Copyright © 2023-2024 Apple Inc.

#include <cstring>
#include "mlx/array.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/gemm.h"
#include "mlx/primitives.h"

namespace mlx::core {

void matmul_general(
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

  if (out.dtype() == float32) {
    matmul<float>(a, b, out, a_transposed, b_transposed, lda, ldb, alpha, beta);
  } else if (out.dtype() == float16) {
    matmul<float16_t>(
        a, b, out, a_transposed, b_transposed, lda, ldb, alpha, beta);
  } else if (out.dtype() == bfloat16) {
    matmul<bfloat16_t>(
        a, b, out, a_transposed, b_transposed, lda, ldb, alpha, beta);
  } else if (out.dtype() == float64) {
    matmul<double>(
        a, b, out, a_transposed, b_transposed, lda, ldb, alpha, beta);
  } else {
    throw std::runtime_error("[Matmul::eval_cpu] Invalid type.");
  }
}

void Matmul::eval_cpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc_or_wait(out.nbytes()));
  if (inputs[0].shape(-1) == 0) {
    std::memset(out.data<void>(), 0, out.nbytes());
    return;
  }
  return matmul_general(inputs[0], inputs[1], out);
}

void AddMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[AddMM::eval_cpu] Currently only supports float32.");
  }

  // Fill output with C
  auto& c = inputs[2];
  CopyType ctype = c.data_size() == 1
      ? CopyType::Scalar
      : (c.flags().row_contiguous ? CopyType::Vector : CopyType::General);
  copy(c, out, ctype);

  return matmul_general(inputs[0], inputs[1], out, alpha_, beta_);
}

} // namespace mlx::core
