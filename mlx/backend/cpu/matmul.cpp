// Copyright © 2023-2024 Apple Inc.

#include <cstring>
#include "mlx/array.h"
#include "mlx/backend/cpu/binary.h"
#include "mlx/backend/cpu/binary_ops.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/gemm.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#if defined(MLX_USE_HIGHWAY_KERNELS)
#include "mlx/backend/cpu/gemms/simd_low_precision_gemm.h"
#include "mlx/types/half_types.h"
#endif

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

#if defined(MLX_USE_HIGHWAY_KERNELS)
template <typename BT>
void matmul_mixed_lowp_dispatch(
    const array& a,
    const array& b,
    array& out,
    size_t lda,
    size_t ldb,
    float alpha,
    float beta,
    Stream stream) {
  const float* a_ptr = a.data<float>();
  const BT* b_ptr = b.data<BT>();
  float* out_ptr = out.data<float>();
  size_t ldc = out.shape(-1);
  size_t batch_size = a.size() / (a.shape(-2) * a.shape(-1));
  auto& encoder = cpu::get_command_encoder(stream);
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  encoder.dispatch([a_ptr,
                    b_ptr,
                    out_ptr,
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
    size_t M = a_shape[a_shape.size() - 2];
    size_t N = b_shape[b_shape.size() - 1];
    size_t K = a_shape.back();
    for (size_t i = 0; i < batch_size; ++i) {
      const float* a_batch = a_ptr + elem_to_loc(M * K * i, a_shape, a_strides);
      const BT* b_batch = b_ptr + elem_to_loc(K * N * i, b_shape, b_strides);
      float* out_batch = out_ptr + M * N * i;
      detail::mixed_lowp_fewrow_gemm_threaded(
          a_batch, b_batch, out_batch, M, N, K, lda, ldb, ldc, alpha, beta);
    }
  });
}

bool try_mixed_lowp_matmul(
    const array& a,
    const array& b,
    array& out,
    Stream stream,
    float alpha,
    float beta) {
  if (out.dtype() != float32 || a.dtype() != float32 ||
      (b.dtype() != float16 && b.dtype() != bfloat16)) {
    return false;
  }
  auto a_stx = a.strides()[a.ndim() - 2];
  auto a_sty = a.strides()[a.ndim() - 1];
  auto b_stx = b.strides()[b.ndim() - 2];
  auto b_sty = b.strides()[b.ndim() - 1];
  size_t M = a.shape(-2);
  size_t N = b.shape(-1);
  size_t K = a.shape(-1);
  if (M == 0 || N == 0 || K == 0) {
    return false;
  }
  bool a_row_contiguous = a_stx == a.shape(-1) && a_sty == 1;
  bool b_transposed = b_stx == 1 && b_sty == b.shape(-2);
  if (!a_row_contiguous || !b_transposed || M > detail::LOWP_FEWROW_MAX_M ||
      N * K < 65536) {
    return false;
  }
  if (b.dtype() == float16) {
    matmul_mixed_lowp_dispatch<float16_t>(
        a, b, out, a_stx, b_sty, alpha, beta, stream);
  } else {
    matmul_mixed_lowp_dispatch<bfloat16_t>(
        a, b, out, a_stx, b_sty, alpha, beta, stream);
  }
  return true;
}

bool has_mixed_lowp_rhs(const array& a, const array& b, const array& out) {
  return out.dtype() == float32 && a.dtype() == float32 &&
      (b.dtype() == float16 || b.dtype() == bfloat16);
}
#endif

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

#if defined(MLX_USE_HIGHWAY_KERNELS)
  if (try_mixed_lowp_matmul(a_pre, b_pre, out, stream, alpha, beta)) {
    return;
  }
  if (has_mixed_lowp_rhs(a_pre, b_pre, out)) {
    array b_f32(b_pre.shape(), float32, nullptr, {});
    CopyType ctype =
        b_pre.flags().row_contiguous ? CopyType::Vector : CopyType::General;
    copy_cpu(b_pre, b_f32, ctype, stream);
    matmul_general(a_pre, b_f32, out, stream, alpha, beta);
    cpu::get_command_encoder(stream).add_temporary(std::move(b_f32));
    return;
  }
#endif

  auto [a_transposed, lda, a] = check_transpose(a_pre);
  auto [b_transposed, ldb, b] = check_transpose(b_pre);
  size_t M = a.shape(-2);
  size_t N = b.shape(-1);
  if (M == 0 || N == 0) {
    return;
  }

  dispatch_inexact_types(out.dtype(), "[Matmul::eval_cpu]", [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    matmul_dispatch<T>(
        a, b, out, a_transposed, b_transposed, lda, ldb, alpha, beta, stream);
  });
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
  if (out.size() == 0) {
    out.set_data(allocator::malloc(out.nbytes()));
    return;
  }

  // Handle empty matrix case (K=0)
  if (inputs[0].shape(-1) == 0) {
    auto& c = inputs[2];
    if (beta_ == 1.0f) {
      CopyType ctype = c.data_size() == 1
          ? CopyType::Scalar
          : (c.flags().row_contiguous ? CopyType::Vector : CopyType::General);
      copy_cpu(c, out, ctype, stream());
    } else {
      array beta_scalar = array(beta_, c.dtype());
      auto& encoder = cpu::get_command_encoder(stream());
      binary_float_op_cpu(c, beta_scalar, out, detail::Multiply(), stream());
      encoder.add_temporary(std::move(beta_scalar));
    }
    return;
  }

  // Fill output with C
  auto& c = inputs[2];
  CopyType ctype = c.data_size() == 1
      ? CopyType::Scalar
      : (c.flags().row_contiguous ? CopyType::Vector : CopyType::General);
  copy_cpu(c, out, ctype, stream());
  matmul_general(inputs[0], inputs[1], out, stream(), alpha_, beta_);
}

} // namespace mlx::core
