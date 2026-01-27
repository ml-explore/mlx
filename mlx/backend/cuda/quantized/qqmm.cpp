// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/qmv.h"
#include "mlx/backend/cuda/quantized/qqmm_impl.h"
#include "mlx/backend/cuda/quantized/qqmm_utils.h"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/quantized/quantized_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace {

array pad_and_swizzle_scales(
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
  // https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
  // Note: cu::malloc_async already provides 256-byte alignment
  array scale_tiled(
      cu::malloc_async(pad_outer * pad_inner, encoder),
      Shape{pad_outer, pad_inner},
      scale.dtype());
  swizzle_scales(scale, scale_tiled, encoder, s);

  encoder.add_temporary(scale_tiled);
  return scale_tiled;
}

} // namespace

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(
      (inputs.size() == 3 && inputs[1].dtype() == uint32) ||
      (inputs.size() == 2));
  nvtx3::scoped_range r("QQMatmul::eval_gpu");

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  auto& device = encoder.device();

  bool w_quantized = (inputs[1].dtype() == uint32);
  if (w_quantized && inputs[0].shape(-2) == 1) {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    bool donate_x = inputs[0].is_donatable();
    array x = ensure_row_contiguous(inputs[0], encoder, s);
    // If x is a copy it should be donatable
    donate_x |= x.is_donatable();
    auto xhat = donate_x
        ? x
        : array(cu::malloc_async(x.nbytes(), encoder), x.shape(), x.dtype());
    if (!donate_x) {
      encoder.add_temporary(xhat);
    }
    fp_quantize_dequantize(x, xhat, group_size_, bits_, encoder, s);

    // Make sure the last two dims of w and s are contiguous
    array w = ensure_row_contiguous_matrix(inputs[1], encoder, s);
    array scales = ensure_row_contiguous_matrix(inputs[2], encoder, s);

    bool non_batched = w.ndim() == 2;
    int K = x.shape(-1);
    int M = non_batched ? x.size() / K : x.shape(-2);
    int N = out.shape(-1);

    fp_qmv(w, scales, xhat, out, bits_, group_size_, M, N, K, encoder);
    return;
  }

  auto cc = device.compute_capability_major() * 100 +
      device.compute_capability_minor() * 10;
  if (cc < 1000) {
    throw std::runtime_error(
        "[QQMatmul::eval_gpu] QQMM is only supported on GPUs with compute capability 10.0 or higher.");
  }
  auto quantize = [&](const array& input,
                      cu::CommandEncoder& encoder,
                      const Stream& s) -> std::pair<array, array> {
    auto x = ensure_contiguous(input, encoder, s);
    auto xq_shape = x.shape();
    xq_shape.back() = x.shape(-1) * bits_ / 32;

    auto sshape = x.shape();
    const int64_t scales_inner = x.shape(-1) / group_size_;
    auto [pad_outer, pad_inner] =
        get_padded_scale_dims(x.shape(-2), scales_inner);
    sshape[x.ndim() - 2] = pad_outer;
    sshape[x.ndim() - 1] = pad_inner;
    sshape.back() = scales_inner;

    // Allocate outputs
    const int64_t xq_bytes = x.size() * bits_ / 8;
    const int64_t batch = x.size() / (x.shape(-2) * x.shape(-1));
    const int64_t scales_bytes = batch * (pad_outer * pad_inner);

    array x_q(cu::malloc_async(xq_bytes, encoder), std::move(xq_shape), uint32);
    array scales_x(
        cu::malloc_async(scales_bytes, encoder), std::move(sshape), uint8);

    fp_quantize(x, x_q, scales_x, group_size_, bits_, encoder, s);

    encoder.add_temporary(x_q);
    encoder.add_temporary(scales_x);
    return {x_q, scales_x};
  };
  auto [x_q, scale_x_pre] = quantize(inputs[0], encoder, s);
  auto [w_q, scale_w_pre] = !w_quantized ? quantize(inputs[1], encoder, s)
                                         : std::make_pair(inputs[1], inputs[2]);

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  auto out_dtype = out.dtype();

  int M = x_q.shape(-2);
  int N = w_q.shape(-2); // always transposed
  int K_packed = x_q.shape(-1);
  int K = K_packed * (32 / bits_);

  // Repack scales from linear to tiled layout for tensor cores
  array scale_x = pad_and_swizzle_scales(scale_x_pre, encoder, s);
  array scale_w = pad_and_swizzle_scales(scale_w_pre, encoder, s);

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
