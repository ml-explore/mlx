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
  bool w_quantized = (inputs[1].dtype() == uint32);
  int base_size = w_quantized ? 3 : 2;

  assert(
      inputs.size() == base_size ||
      (mode_ == QuantizationMode::Nvfp4 && inputs.size() == base_size + 2));

  if (w_quantized && inputs[0].shape(-2) == 1) {
    out.set_data(cu::malloc_async(out.nbytes(), encoder));

    // For nvfp4, get global scale for x from inputs if present
    bool has_global_scale =
        mode_ == QuantizationMode::Nvfp4 && inputs.size() > base_size;
    std::optional<array> global_scale = std::nullopt;
    if (has_global_scale) {
      global_scale = inputs[inputs.size() - 2];
    }

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
    fp_quantize_dequantize(
        x, xhat, group_size_, bits_, global_scale, encoder, s);

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

  // - 2 inputs: x, w (non-quantized w)
  // - 3 inputs: x, w, scales_w (quantized w)

  // For nvfp4, global scales are optional but must be both present or both
  // absent If present, they add 2 more inputs (global_scale_x, global_scale_w)
  bool has_global_scales =
      mode_ == QuantizationMode::Nvfp4 && inputs.size() > base_size;

  // For nvfp4, get global scales from inputs if present
  std::optional<array> global_scale_x = std::nullopt;
  std::optional<array> global_scale_w = std::nullopt;
  if (has_global_scales) {
    global_scale_x = inputs[inputs.size() - 2];
    global_scale_w = inputs[inputs.size() - 1];
  }

  // Quantize inputs (or use pre-quantized)
  auto [x_q, scale_x_pre] = quantize_input(
      inputs[0], encoder, s, mode_, bits_, group_size_, global_scale_x);
  auto [w_q, scale_w_pre] = !w_quantized
      ? quantize_input(
            inputs[1], encoder, s, mode_, bits_, group_size_, global_scale_w)
      : std::make_tuple(
            ensure_contiguous(inputs[1], encoder, s),
            ensure_contiguous(inputs[2], encoder, s));

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  int M = x_q.shape(-2);
  int N = w_q.shape(-2); // transposed
  int K = x_q.shape(-1) * (32 / bits_);

  bool x_transposed = false;
  bool w_transposed = true; // always transposed
  int64_t lda = K;
  int64_t ldb = K;

  // Repack scales to tiled layout for tensor cores
  array scale_x = pad_and_swizzle_scales(scale_x_pre, encoder, s);
  array scale_w = pad_and_swizzle_scales(scale_w_pre, encoder, s);

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
      scale_x,
      scale_w,
      mode_,
      scalars);
}

} // namespace mlx::core
