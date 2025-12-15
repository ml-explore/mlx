// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/cublas_qqmm.h"
#include "mlx/backend/cuda/quantized/qqmm_utils.h"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

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
//
https: // docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
  // Note: cu::malloc_async already provides 256-byte alignment
  array scale_tiled(
      cu::malloc_async(pad_outer * pad_inner, encoder),
      Shape{pad_outer, pad_inner},
      scale.dtype());
  repack_scales(scale, scale_tiled, encoder, s);

  encoder.add_temporary(scale_tiled);
  return scale_tiled;
}
} // namespace

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
    Dtype out_dtype,
    QuantizationMode mode,
    float alpha = 1.0f) {
  // Invoke CublasQQMM
  std::string qmode = quantization_mode_to_string(mode);

  // Currently only supports non-batched QQMM operations
  // that covers all use cases for training, we will just collapse (batch,
  // seq_len) into (tokens)
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
      out_dtype,
      qmode);

  qqmm.run(encoder, out, a, b, a_scale, b_scale, alpha);
}
} // namespace

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("QQMatmul::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  assert(inputs.size() == 2 || inputs.size() == 3);

  auto quantize =
      [&](const array& input, cu::CommandEncoder& encoder, const Stream& s) {
        auto x = ensure_row_contiguous(input, encoder, s);
        auto xq_shape = x.shape();
        xq_shape.back() = x.shape(-1) * bits_ / 32;
        auto sshape = x.shape();
        std::tie(sshape[x.ndim() - 2], sshape[x.ndim() - 1]) =
            get_padded_scale_dims(x.shape(-2), x.shape(-1) / group_size_);
        sshape.back() = x.shape(-1) / group_size_;
        auto scales_size = x.size() / (x.shape(-1) * x.shape(-2)) *
            (sshape[x.ndim() - 2] * sshape[x.ndim() - 1]);
        auto xq_size = x.size() * bits_ / 8;
        array x_q(cu::malloc_async(xq_size, encoder), xq_shape, uint32);
        array scales_x(cu::malloc_async(scales_size, encoder), sshape, uint8);
        fp_quantize(x, x_q, scales_x, group_size_, bits_, encoder, s);
        encoder.add_temporary(scales_x);
        encoder.add_temporary(x_q);
        return std::make_pair(x_q, scales_x);
      };
  // todo declare wq and scales_pre
  auto [x_q, scale_x_pre] = quantize(inputs[0], encoder, s);
  auto [w_q, scale_w_pre] = [&]() {
    if (inputs[1].dtype() != uint32) {
      return quantize(inputs[1], encoder, s);
    } else {
      return std::make_pair(inputs[1], inputs[2]);
    }
  }();
  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  auto out_dtype = out.dtype();

  int M = x_q.shape(-2);
  int N = w_q.shape(-2); // always transposed
  int K_packed = x_q.shape(-1);
  int K = K_packed * (32 / bits_);

  // // Repack scales from linear to tiled layout for tensor cores
  array scale_x = pad_and_repack_scales(scale_x_pre, encoder, s);
  array scale_w = pad_and_repack_scales(scale_w_pre, encoder, s);

  bool x_transposed = false;
  bool w_transposed = true; // always transposed
  int64_t lda = K;
  int64_t ldb = K;

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
      out_dtype,
      mode_);
}

} // namespace mlx::core
