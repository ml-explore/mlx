// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/quantized/quantized.h"
#include <nvtx3/nvtx3.hpp>
#include "mlx/backend/common/matmul.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/cublas_qqmm.h"
#include "mlx/backend/cuda/quantized/qqmm_utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

namespace mlx::core {

namespace {

inline array ensure_row_contiguous(
    const array& x,
    cu::CommandEncoder& enc,
    const Stream& s) {
  if (!x.flags().row_contiguous) {
    array x_copy = contiguous_copy_gpu(x, s);
    enc.add_temporary(x_copy);
    return x_copy;
  } else {
    return x;
  }
}

inline array ensure_row_contiguous_matrix(
    const array& x,
    cu::CommandEncoder& enc,
    const Stream& s) {
  if (x.ndim() < 2) {
    if (x.strides()[0] == 1) {
      return x;
    }
  } else {
    auto stride_0 = x.strides()[x.ndim() - 2];
    auto stride_1 = x.strides()[x.ndim() - 1];
    if (stride_0 == x.shape(-1) && stride_1 == 1) {
      return x;
    }
  }
  array x_copy = contiguous_copy_gpu(x, s);
  enc.add_temporary(x_copy);
  return x_copy;
}

array pad_and_repack_scales(
    const array& scale,
    cu::CommandEncoder& encoder,
    const Stream& s) {
  // Compute padded dimensions for full tiles (128 rows × 4 cols)
  auto [pad_outer, pad_inner] =
      get_padded_scale_dims(scale.shape(-2), scale.shape(-1));
  // cuBLAS requirements for scale factor layout:
  // 1. Dimensions must be padded to full tiles (128 rows × 4 cols)
  // 2. Out-of-bounds values must be filled with zeros
  // 3. Starting addresses must be 16-byte aligned
  // https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
  // Note: cu::malloc_async already provides 256-byte alignment
  array scale_tiled(
      cu::malloc_async(pad_outer * pad_inner, encoder.stream()),
      Shape{pad_outer, pad_inner},
      scale.dtype());
  repack_scales(scale, scale_tiled, encoder, s);

  encoder.add_temporary(scale_tiled);
  return scale_tiled;
}

} // namespace

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("Quantize::eval_gpu");
  auto& s = stream();
  auto& d = cu::device(s.device);
  auto& enc = d.get_command_encoder(s);

  if (dequantize_) {
    auto wq = ensure_row_contiguous(inputs[0], enc, s);
    auto scales = ensure_row_contiguous(inputs[1], enc, s);
    auto& w = outputs[0];

    w.set_data(cu::malloc_async(w.nbytes(), enc.stream()));

    if (mode_ == QuantizationMode::Affine) {
      auto biases = ensure_row_contiguous(inputs[2], enc, s);
      affine_dequantize(wq, scales, biases, w, group_size_, bits_, enc, s);
    } else {
      fp_dequantize(wq, scales, w, group_size_, bits_, enc, s);
    }
  } else {
    auto w = ensure_row_contiguous(inputs[0], enc, s);
    auto& wq = outputs[0];
    auto& scales = outputs[1];

    wq.set_data(cu::malloc_async(wq.nbytes(), enc.stream()));
    scales.set_data(cu::malloc_async(scales.nbytes(), enc.stream()));
    if (mode_ == QuantizationMode::Affine) {
      auto& biases = outputs[2];
      biases.set_data(cu::malloc_async(biases.nbytes(), enc.stream()));
      affine_quantize(w, wq, scales, biases, group_size_, bits_, enc, s);
    } else {
      fp_quantize(w, wq, scales, group_size_, bits_, enc, s);
    }
  }
}

namespace {
void qqmm_impl(
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
    const array& a_scale,
    const array& b_scale,
    QuantizationMode mode,
    float alpha = 1.0f) {
  // Invoke CublasQQMM
  auto [batch_shape, a_batch_strides, b_batch_strides] = collapse_batches(a, b);
  auto batch_count = out.size() / (M * N);

  std::string_view qmode = quantization_mode_to_string(mode);
  if (batch_count > 1 && !a_transposed && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == K && a_batch_strides.back() == M * K &&
      b_batch_strides.back() == 0) {
    M *= batch_shape.back();
    batch_count = 1;

    a_batch_strides = {0};
    b_batch_strides = {0};
    batch_shape = {1};
  }

  CublasQQMM qqmm(
      encoder.device(),
      a_transposed,
      M,
      K,
      lda,
      b_transposed,
      N,
      K,
      ldb,
      qmode,
      batch_shape.back(),
      a_batch_strides.back(),
      b_batch_strides.back());

  qqmm.run(
      encoder,
      out,
      a,
      b,
      a_scale,
      b_scale,
      batch_shape,
      a_batch_strides,
      b_batch_strides,
      alpha);
}
} // namespace

void DualQuantizedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  nvtx3::scoped_range r("DualQuantizedMatmul::eval_gpu");
  // WIP need to add primitive
  // TODO: for now minimalistic implementation without batching support
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 4);
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& scale_a_pre = inputs[2];
  auto& scale_b_pre = inputs[3];
  // Return 0s if either input is empty.
  if (a.size() == 0 || b.size() == 0) {
    array zero(0, a.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }
  out.set_data(cu::malloc_async(out.nbytes(), encoder.stream()));

  int M = a.shape(-2);
  int N = b.shape(-2); // b always transposed
  int K_packed = a.shape(-1);
  int K = K_packed * (32 / bits_);

  // Repack scales from linear to tiled layout for tensor cores
  array scale_a_tiled = pad_and_repack_scales(scale_a_pre, encoder, s);
  array scale_b_tiled = pad_and_repack_scales(scale_b_pre, encoder, s);

  bool a_transposed = false; // a is normal (M x K)
  bool b_transposed = true; // b is transposed (N x K -> K x N)
  int64_t lda = K; // Leading dimension of a (packed)
  int64_t ldb = K; // Leading dimension of b (packed)

  qqmm_impl(
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
      scale_a_tiled,
      scale_b_tiled,
      mode_);
}

} // namespace mlx::core
