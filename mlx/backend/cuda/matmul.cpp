// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/gemms/gemv.h"
#include "mlx/backend/cuda/gemms/grouped_gemm.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <numeric>

namespace mlx::core {

namespace {

std::tuple<bool, int64_t, array>
check_transpose(cu::CommandEncoder& enc, const Stream& s, const array& arr) {
  auto stx = arr.strides()[arr.ndim() - 2];
  auto sty = arr.strides()[arr.ndim() - 1];
  if (sty == 1 && stx == arr.shape(-1)) {
    return std::make_tuple(false, stx, arr);
  } else if (stx == 1 && sty == arr.shape(-2)) {
    return std::make_tuple(true, sty, arr);
  } else {
    array arr_copy = contiguous_copy_gpu(arr, s);
    enc.add_temporary(arr_copy);
    return std::make_tuple(false, arr.shape(-1), arr_copy);
  }
}

std::tuple<bool, int64_t, array>
ensure_batch_contiguous(const array& x, cu::CommandEncoder& encoder, Stream s) {
  if (x.flags().row_contiguous) {
    return std::make_tuple(false, x.strides(-2), x);
  }

  bool rc = true;
  for (int i = 0; i < x.ndim() - 3; i++) {
    rc &= (x.strides(i + 1) * x.shape(i)) == x.strides(i);
  }
  if (rc) {
    return check_transpose(encoder, s, x);
  }

  array x_copy = contiguous_copy_gpu(x, s);
  encoder.add_temporary(x_copy);
  return std::make_tuple(false, x_copy.strides(-2), x_copy);
}

array ensure_row_contiguous(
    const array& x,
    cu::CommandEncoder& encoder,
    Stream s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    encoder.add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

void gemm_and_bias(
    cu::CommandEncoder& encoder,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    array& out,
    const array& a,
    const array& b,
    const std::optional<array>& bias = std::nullopt,
    float alpha = 1.0f) {
  // Check and collapse batch dimensions
  auto [batch_shape, a_batch_strides, b_batch_strides] = collapse_batches(a, b);

  auto batch_count = out.size() / (M * N);

  // Collapse batches into M if needed
  if (batch_count > 1 && !a_transposed && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == K && a_batch_strides.back() == M * K &&
      b_batch_strides.back() == 0) {
    M *= batch_shape.back();
    batch_count = 1;

    a_batch_strides = {0};
    b_batch_strides = {0};
    batch_shape = {1};
  }

  // Use gemmv when possible
  if (!bias && cu::can_use_gemv(M, N, K, a_transposed, b_transposed)) {
    cu::gemv(
        a,
        b,
        out,
        M,
        N,
        K,
        batch_count,
        batch_shape,
        a_batch_strides,
        b_batch_strides,
        encoder);
    return;
  }

  // Invoke cublasLt
  CublasGemm gemm(
      encoder.device(),
      a.dtype(),
      a_transposed,
      M,
      K,
      lda,
      b_transposed,
      K,
      N,
      ldb,
      batch_shape.back(),
      a_batch_strides.back(),
      b_batch_strides.back());
  if (bias) {
    if (a.dtype() == complex64) {
      throw std::runtime_error(
          "[gemm_and_bias] complex64 bias epilogue isn’t supported in cublasLtMatmul.");
    }
    gemm.set_bias(encoder, *bias);
  }
  gemm.run(
      encoder, out, a, b, batch_shape, a_batch_strides, b_batch_strides, alpha);
}

void gather_mm_rhs(
    const array& a_,
    const array& b_,
    const array& indices_,
    array& out,
    cu::CommandEncoder& encoder,
    Stream s) {
  if (a_.size() / a_.shape(-2) / a_.shape(-1) != indices_.size()) {
    throw std::runtime_error("[gather_mm] Broadcasting lhs is not supported.");
  }

  int group_count = b_.size() / b_.shape(-1) / b_.shape(-2);
  if (group_count > 1024) {
    throw std::runtime_error(
        "[gather_mm] Group count can not be larger than 1024.");
  }

  auto [a_transposed, lda, a] = ensure_batch_contiguous(a_, encoder, s);
  auto [b_transposed, ldb, b] = ensure_batch_contiguous(b_, encoder, s);
  auto indices = ensure_row_contiguous(indices_, encoder, s);

  cutlass_grouped_gemm_unaligned(
      a_transposed,
      lda,
      b_transposed,
      ldb,
      group_count,
      a,
      b,
      indices,
      out,
      encoder);
}

} // namespace

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Matmul::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 2);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  // Return 0s if either input is empty.
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero(0, a_pre.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  gemm_and_bias(
      encoder, M, N, K, a_transposed, lda, b_transposed, ldb, out, a, b);
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("AddMM::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 3);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  auto c = inputs[2];

  /////////////////////////////////////////////////////////////////////////////
  // Init checks and prep

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  /////////////////////////////////////////////////////////////////////////////
  // Dispatch to GEMM with epilogue or AddMM

  if (beta_ == 1 && a.dtype() != complex64 && c.strides(-1) == 1 &&
      c.data_size() == out.shape(-1)) {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));
    gemm_and_bias(
        encoder,
        M,
        N,
        K,
        a_transposed,
        lda,
        b_transposed,
        ldb,
        out,
        a,
        b,
        c,
        alpha_);
    return;
  }

  int64_t ldc;
  {
    auto stx = c.strides()[c.ndim() - 2];
    auto sty = c.strides()[c.ndim() - 1];
    if (sty == 1 && stx == c.shape(-1)) {
      ldc = stx;
      out.set_data(cu::malloc_async(out.nbytes(), encoder));
    } else if (sty == 1 && stx == 0) {
      ldc = 0;
      out.set_data(cu::malloc_async(out.nbytes(), encoder));
    } else {
      // Copy C into out and set C to out
      ldc = c.shape(-1);
      copy_gpu(c, out, CopyType::General, s);
      c = out;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Check and collapse batch dimensions

  auto [batch_shape, a_batch_strides, b_batch_strides, c_batch_strides] =
      collapse_batches(a, b, c);

  auto batch_count = out.size() / (M * N);

  // Collapse batches into M if needed
  if (batch_count > 1 && !a_transposed && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == K && a_batch_strides.back() == M * K &&
      c_batch_strides.back() == M * c.strides()[c.ndim() - 2] &&
      b_batch_strides.back() == 0) {
    M *= batch_shape.back();
    batch_count = 1;

    a_batch_strides = {0};
    b_batch_strides = {0};
    c_batch_strides = {0};
    batch_shape = {1};
  }

  /////////////////////////////////////////////////////////////////////////////
  // Invoke cublasLt with AddMM settings

  CublasGemm gemm(
      cu::device(s.device),
      a.dtype(),
      a_transposed,
      M,
      K,
      lda,
      b_transposed,
      K,
      N,
      ldb,
      ldc,
      batch_shape.back(),
      a_batch_strides.back(),
      b_batch_strides.back(),
      c_batch_strides.back());
  gemm.run(
      encoder,
      out,
      a,
      b,
      c,
      batch_shape,
      a_batch_strides,
      b_batch_strides,
      c_batch_strides,
      alpha_,
      beta_);
}

void GatherMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("GatherMM::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 4);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& lhs_indices = inputs[2];
  auto& rhs_indices = inputs[3];

  // Return 0s if either input is empty.
  if (a.size() == 0 || b.size() == 0) {
    array zero(0, a.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  // Extract shapes from inputs.
  int M = a.shape(-2);
  int N = b.shape(-1);
  int K = a.shape(-1);

  // We are walking a in order and b is also in order so we can batch up the
  // matmuls and reuse reading a and b.
  if (M == 1 && right_sorted_ == true) {
    gather_mm_rhs(a, b, rhs_indices, out, encoder, s);
    return;
  }

  throw std::runtime_error("NYI");
}

} // namespace mlx::core
