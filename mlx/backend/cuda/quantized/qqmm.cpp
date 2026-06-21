// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/backend/cuda/quantized/qqmm_impl.h"
#include "mlx/backend/cuda/quantized/qqmm_utils.h"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/quantized/quantized_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace {

std::tuple<array, array> quantize_input(
    const array& input,
    cu::CommandEncoder& encoder,
    const Stream& s,
    QuantizationMode mode,
    int bits,
    int group_size,
    std::optional<array> global_scale = std::nullopt) {
  const array x = ensure_contiguous(input, encoder, s);

  // Compute output shapes
  auto xq_shape = x.shape();
  xq_shape.back() = x.shape(-1) * bits / 32;

  const int64_t scales_inner = x.shape(-1) / group_size;
  auto [pad_outer, pad_inner] =
      get_padded_scale_dims(x.shape(-2), scales_inner);

  auto sshape = x.shape();
  sshape[x.ndim() - 2] = pad_outer;
  sshape[x.ndim() - 1] = pad_inner;
  sshape.back() = scales_inner;

  // Allocate outputs
  const int64_t xq_bytes = x.size() * bits / 8;
  const int64_t batch = x.size() / (x.shape(-2) * x.shape(-1));
  const int64_t scales_bytes = batch * (pad_outer * pad_inner);

  array x_q(cu::malloc_async(xq_bytes, encoder), std::move(xq_shape), uint32);
  array scales_x(
      cu::malloc_async(scales_bytes, encoder), std::move(sshape), uint8);
  encoder.add_temporary(x_q);
  encoder.add_temporary(scales_x);
  // global_scale is not nullopt only for NVFP4
  fp_quantize(x, x_q, scales_x, group_size, bits, global_scale, encoder, s);
  return {std::move(x_q), std::move(scales_x)};
}

array quantize_dequantize_input(
    const array& x_pre,
    const std::optional<array>& global_scale_x,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s) {
  bool donate_x = x_pre.is_donatable();
  array x = ensure_row_contiguous(x_pre, encoder, s);
  // If x is a copy it should be donatable
  donate_x |= x.is_donatable();
  auto xhat = donate_x
      ? x
      : array(cu::malloc_async(x.nbytes(), encoder), x.shape(), x.dtype());
  if (!donate_x) {
    encoder.add_temporary(xhat);
  }
  fp_quantize_dequantize(x, xhat, group_size, bits, global_scale_x, encoder, s);
  return xhat;
}

GemmScalars create_nvfp4_scalars(
    const array& global_scale_x,
    const array& global_scale_w,
    cu::CommandEncoder& encoder) {
  // NVFP4 requires alpha/beta as device pointers
  // alpha = amax_x * amax_w / (448 * 6)^2
  // beta = 0
  array alpha(cu::malloc_async(sizeof(float), encoder), {}, float32);
  array beta(cu::malloc_async(sizeof(float), encoder), {}, float32);
  compute_qqmm_pointers(alpha, beta, global_scale_x, global_scale_w, encoder);
  encoder.add_temporary(alpha);
  encoder.add_temporary(beta);
  return {alpha, beta};
}

} // namespace

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("QQMatmul::eval_gpu");

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  auto& device = encoder.device();

  const array& x_pre = inputs[0];
  const array& w_pre = inputs[1];
  const array& scales_w_pre = inputs[2];

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  // - 2 inputs: x, w (non-quantized w)
  // - 3 inputs: x, w, scales_w (quantized w)
  bool w_quantized = (w_pre.dtype() == uint32);
  int base_size = w_quantized ? 3 : 2;
  assert(
      inputs.size() == base_size ||
      (mode_ == QuantizationMode::Nvfp4 && inputs.size() == base_size + 2));

  // For nvfp4, global scales are optional but must be both present or both
  // absent If present, they add 2 more inputs (global_scale_x, global_scale_w)
  bool has_global_scales =
      mode_ == QuantizationMode::Nvfp4 && inputs.size() > base_size;
  std::optional<array> global_scale_x = std::nullopt;
  std::optional<array> global_scale_w = std::nullopt;
  if (has_global_scales) {
    global_scale_x = inputs[inputs.size() - 2];
    global_scale_w = inputs[inputs.size() - 1];
  }

  // Quantize weights.
  auto [w_q, scales_w] = !w_quantized
      ? quantize_input(
            w_pre, encoder, s, mode_, bits_, group_size_, global_scale_w)
      : std::make_tuple(
            ensure_contiguous(w_pre, encoder, s),
            ensure_contiguous(scales_w_pre, encoder, s));

  // Reroute to qmm when: no support in cuBLAS, or doing GEMV.
  int M = x_pre.shape(-2);
  bool use_qmm = (device.compute_capability_major() < 10) || (M == 1);
  use_qmm = true;

  if (use_qmm) {
    array x = quantize_dequantize_input(
        x_pre, global_scale_x, bits_, group_size_, encoder, s);
    if (M < 8) {
      qmv(x,
          w_q,
          scales_w,
          std::nullopt,
          global_scale_w,
          out,
          bits_,
          group_size_,
          mode_,
          encoder);
    } else {
      qmm_naive(
          x,
          w_q,
          scales_w,
          std::nullopt,
          global_scale_w,
          std::nullopt,
          std::nullopt,
          out,
          true, // transpose
          bits_,
          group_size_,
          mode_,
          encoder);
    }
    return;
  }

  // Quantize activation.
  auto [x_q, scales_x] = quantize_input(
      x_pre, encoder, s, mode_, bits_, group_size_, global_scale_x);

  int N = w_q.shape(-2); // transposed
  int K = x_q.shape(-1) * (32 / bits_);

  bool x_transposed = false;
  bool w_transposed = true; // always transposed
  int64_t lda = K;
  int64_t ldb = K;

  // Repack scales to tiled layout for tensor cores
  scales_x = pad_and_swizzle_scales(scales_x, encoder, s);
  scales_w = pad_and_swizzle_scales(scales_w, encoder, s);

  GemmScalars scalars;
  if (has_global_scales) {
    scalars = create_nvfp4_scalars(*global_scale_x, *global_scale_w, encoder);
  }

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
      scales_x,
      scales_w,
      mode_,
      scalars);
}

} // namespace mlx::core
