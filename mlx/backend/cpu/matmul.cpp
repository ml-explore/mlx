// Copyright © 2023-2024 Apple Inc.

#include <cstring>
#include "mlx/array.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/gemm.h"
#include "mlx/primitives.h"

namespace mlx::core {

template <typename T>
void matmul_dispatch(
    const array& a,
    const array& b,
    array& out,
    bool a_transposed,
    bool b_transposed,
    size_t lda,
    size_t ldb,
    float alpha,
    float beta,
    Stream stream) {
  const T* a_ptr = a.data<T>();
  const T* b_ptr = b.data<T>();
  T* out_ptr = out.data<T>();
  size_t ldc = out.shape(-1);
  size_t batch_size = a.size() / (a.shape(-2) * a.shape(-1));
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  encoder.dispatch([a_ptr,
                    b_ptr,
                    out_ptr,
                    a_transposed,
                    b_transposed,
                    lda,
                    ldb,
                    ldc,
                    alpha,
                    beta,
                    batch_size,
                    a_shape = a.shape(),
                    a_strides = a.strides(),
                    b_shape = b.shape(),
                    b_strides = b.strides()]() {
    matmul<T>(
        a_ptr,
        b_ptr,
        out_ptr,
        a_transposed,
        b_transposed,
        lda,
        ldb,
        ldc,
        alpha,
        beta,
        batch_size,
        a_shape,
        a_strides,
        b_shape,
        b_strides);
  });
}

void matmul_general(
    const array& a_pre,
    const array& b_pre,
    array& out,
    Stream stream,
    float alpha = 1.0f,
    float beta = 0.0f) {
  std::vector<array> temps;
  auto check_transpose = [stream, &temps](const array& arr) {
    auto stx = arr.strides()[arr.ndim() - 2];
    auto sty = arr.strides()[arr.ndim() - 1];
    if (stx == arr.shape(-1) && sty == 1) {
      return std::make_tuple(false, stx, arr);
    } else if (stx == 1 && sty == arr.shape(-2)) {
      return std::make_tuple(true, sty, arr);
    } else {
      temps.push_back(array(arr.shape(), arr.dtype(), nullptr, {}));
      copy_cpu(arr, temps.back(), CopyType::General, stream);
      stx = arr.shape(-1);
      return std::make_tuple(false, stx, temps.back());
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
    matmul_dispatch<float>(
        a, b, out, a_transposed, b_transposed, lda, ldb, alpha, beta, stream);
  } else if (out.dtype() == float16) {
    matmul_dispatch<float16_t>(
        a, b, out, a_transposed, b_transposed, lda, ldb, alpha, beta, stream);
  } else if (out.dtype() == bfloat16) {
    matmul_dispatch<bfloat16_t>(
        a, b, out, a_transposed, b_transposed, lda, ldb, alpha, beta, stream);
  } else if (out.dtype() == float64) {
    matmul_dispatch<double>(
        a, b, out, a_transposed, b_transposed, lda, ldb, alpha, beta, stream);
  } else {
    throw std::runtime_error("[Matmul::eval_cpu] Invalid type.");
  }
  cpu::get_command_encoder(stream).add_temporaries(std::move(temps));
}

void Matmul::eval_cpu(const std::vector<array>& inputs, array& out) {
  out.set_data(allocator::malloc(out.nbytes()));
  if (inputs[0].shape(-1) == 0) {
    auto& encoder = cpu::get_command_encoder(stream());
    encoder.set_output_array(out);
    encoder.dispatch([out_ptr = out.data<void>(), nbytes = out.nbytes()]() {
      std::memset(out_ptr, 0, nbytes);
    });
    return;
  }
  matmul_general(inputs[0], inputs[1], out, stream());
}

void AddMM::eval_cpu(const std::vector<array>& inputs, array& out) {
  if (out.dtype() != float32) {
    throw std::runtime_error(
        "[AddMM::eval_cpu] Currently only supports float32.");
  }
  if (out.size() == 0) {
    out.set_data(allocator::malloc(out.nbytes()));
    return;
  }

  // Fill output with C
  auto& c = inputs[2];
  CopyType ctype = c.data_size() == 1
      ? CopyType::Scalar
      : (c.flags().row_contiguous ? CopyType::Vector : CopyType::General);
  copy_cpu(c, out, ctype, stream());
  if (inputs[0].shape(-1) == 0) {
    return;
  }
  matmul_general(inputs[0], inputs[1], out, stream(), alpha_, beta_);
}

} // namespace mlx::core
