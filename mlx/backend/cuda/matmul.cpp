// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/block_mask.h"
#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/gemms/gather_gemm.h"
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

void BlockMaskedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("BlockMaskedMM::eval_gpu");
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[BlockMaskedMM] Does not yet support non-floating point types.");
  }
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];

  // Return 0s if either input is empty.
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero(0, a_pre.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  if (M == 0 || N == 0) {
    return;
  }
  if (K == 0) {
    array zero(0, a_pre.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  bool has_op_mask = inputs.size() > 3;
  bool has_out_mask = inputs.size() == 3 || inputs.size() == 5;

  int64_t batch_count = out.size() / (int64_t(M) * N);

  bool a_transposed;
  int64_t lda;
  array a = a_pre;
  bool b_transposed;
  int64_t ldb;
  array b = b_pre;

  if (has_op_mask) {
    // Fused copy + mask in a single pass per matrix.
    auto& lhs_mask = inputs[inputs.size() - 2];
    auto& rhs_mask = inputs[inputs.size() - 1];
    a = copy_with_block_mask(
        encoder, a_pre, lhs_mask, block_size_, M, K, batch_count);
    b = copy_with_block_mask(
        encoder, b_pre, rhs_mask, block_size_, K, N, batch_count);
    a_transposed = false;
    lda = K;
    b_transposed = false;
    ldb = N;
  } else {
    std::tie(a_transposed, lda, a) = check_transpose(encoder, s, a_pre);
    std::tie(b_transposed, ldb, b) = check_transpose(encoder, s, b_pre);
  }

  // Run GEMM.
  gemm_and_bias(
      encoder, M, N, K, a_transposed, lda, b_transposed, ldb, out, a, b);

  // Apply output mask.
  if (has_out_mask) {
    auto& out_mask = inputs[2];
    apply_block_mask(encoder, out, out_mask, block_size_, M, N, batch_count);
  }
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

  // Extract shapes from inputs.
  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  auto [a_transposed, lda, a] = ensure_batch_contiguous(a_pre, encoder, s);
  auto [b_transposed, ldb, b] = ensure_batch_contiguous(b_pre, encoder, s);
  auto lhs_indices = ensure_row_contiguous(inputs[2], encoder, s);
  auto rhs_indices = ensure_row_contiguous(inputs[3], encoder, s);

  // We are walking a in order and b is also in order so we can batch up the
  // matmuls and reuse reading a and b.
  if (M == 1 && right_sorted_ == true) {
    cutlass_grouped_gemm_unaligned(
        a_transposed,
        lda,
        b_transposed,
        ldb,
        b.size() / b.shape(-1) / b.shape(-2), // group_count
        a,
        b,
        rhs_indices,
        out,
        encoder);
    return;
  }

  auto use_gemv = cu::can_use_gemv(M, N, K, a_transposed, b_transposed);
  if (M == 1 && use_gemv) {
    gather_mv(b, a, rhs_indices, lhs_indices, out, N, K, encoder);
    return;
  }
  if (N == 1 && use_gemv) {
    gather_mv(a, b, lhs_indices, rhs_indices, out, M, K, encoder);
    return;
  }

  cutlass_gather_mm(
      a_transposed, b_transposed, a, b, lhs_indices, rhs_indices, out, encoder);
}

void SegmentedMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("SegmentedMM::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 3);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  auto& segments_pre = inputs[2];

  // Return zeros if output is empty or either input is empty.
  if (out.size() == 0 || a_pre.size() == 0 || b_pre.size() == 0) {
    array zero(0, a_pre.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int num_segments = segments_pre.size() / 2;

  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);
  auto segments = ensure_row_contiguous(segments_pre, encoder, s);

  cutlass_segmented_mm(
      a_transposed,
      lda,
      b_transposed,
      ldb,
      num_segments,
      M,
      N,
      a,
      b,
      segments,
      out,
      encoder);
}

} // namespace mlx::core
