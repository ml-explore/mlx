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
  // Calculate batch size (all dimensions except last 2)
  size_t batch_size = scale.size() / (scale.shape(-2) * scale.shape(-1));
  size_t collapsed_outer = batch_size * scale.shape(-2);

  auto [pad_outer, pad_inner] =
      get_padded_scale_dims(collapsed_outer, scale.shape(-1));

  Shape out_shape = {pad_outer, pad_inner};

  // cuBLAS requirements for scale factor layout:
  // 1. Dimensions must be padded to full tiles (128 rows × 4 cols)
  // 2. Out-of-bounds values must be filled with zeros
  // 3. Starting addresses must be 16-byte aligned
  // https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
  // Note: cu::malloc_async already provides 256-byte alignment
  array scale_tiled(
      cu::malloc_async(pad_outer * pad_inner, encoder),
      out_shape,
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

    w.set_data(cu::malloc_async(w.nbytes(), enc));

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

    wq.set_data(cu::malloc_async(wq.nbytes(), enc));
    scales.set_data(cu::malloc_async(scales.nbytes(), enc));
    if (mode_ == QuantizationMode::Affine) {
      auto& biases = outputs[2];
      biases.set_data(cu::malloc_async(biases.nbytes(), enc));
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
  std::string qmode = quantization_mode_to_string(mode);
  
  CublasQQMM qqmm(
      encoder.device(),
      a_transposed,
      M,
      K,
      lda,
      b_transposed,
      K,
      N,
      ldb,
      1, // batch_count
      0, // a_batch_stride
      0, // b_batch_stride
      qmode);

  qqmm.run(encoder, out, a, b, a_scale, b_scale, alpha);
}
} // namespace

void DualQuantizedMatmul::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  nvtx3::scoped_range r("DualQuantizedMatmul::eval_gpu");
  // for now it is size of 4: bf16 x, bf16 w, w_q, scale_w
  // for the inference & vjp
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 4 || inputs.size() == 3);
  auto& x = inputs[0]; // activations bf16
  auto& w_q = inputs[1]; // quantized weights
  auto& scale_w_pre = inputs[2];

  auto quantize_activation =
      [&](const array& input, cu::CommandEncoder& encoder, const Stream& s) {
        auto x = ensure_row_contiguous(input, encoder, s);
        auto xq_shape = x.shape();
        xq_shape.back() = x.shape(-1) * bits_ / 32;
        auto sshape = x.shape();
        sshape.back() = x.shape(-1) / group_size_;
        array x_q(
            cu::malloc_async(x.size() * bits_ / 8, encoder), xq_shape, uint32);
        array scales_x(
            cu::malloc_async(x.size() / group_size_ * sizeof(uint8), encoder),
            sshape,
            uint8);
        fp_quantize(x, x_q, scales_x, group_size_, bits_, encoder, s);
        encoder.add_temporary(scales_x);
        encoder.add_temporary(x_q);
        return std::make_pair(x_q, scales_x);
      };

  auto [x_q, scale_x_pre] = quantize_activation(inputs[0], encoder, s);
  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  int M = x_q.shape(-2);
  int N = transpose_ ? w_q.shape(-2) : w_q.shape(-1);
  int K_packed = x_q.shape(-1);
  int K = K_packed * (32 / bits_);

  // Repack scales from linear to tiled layout for tensor cores
  array scale_x = pad_and_repack_scales(scale_x_pre, encoder, s);
  array scale_w = pad_and_repack_scales(scale_w_pre, encoder, s);

  bool x_transposed = false;
  bool w_transposed = transpose_;
  int64_t lda = K;
  int64_t ldb = transpose_ ? K : N;

  qqmm_impl(
      encoder,
      M,
      N,
      K,
      x_transposed,
      lda,
      w_transposed,
      ldb,
      out,
      x_q,
      w_q,
      scale_x,
      scale_w,
      mode_);
}

} // namespace mlx::core
