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

using QuantizedTensor = std::tuple<array, array, std::optional<array>>;

struct GemmScalars {
  std::optional<array> alpha_device;
  std::optional<array> beta_device;

  bool uses_device_pointers() const {
    return alpha_device.has_value();
  }
};

inline array ensure_row_contiguous(
    const array& x,
    cu::CommandEncoder& enc,
    const Stream& s) {
  if (x.flags().row_contiguous) {
    return x;
  }
  array x_copy = contiguous_copy_gpu(x, s);
  enc.add_temporary(x_copy);
  return x_copy;
}

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
  // https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
  array scale_tiled(
      cu::malloc_async(pad_outer * pad_inner, encoder),
      Shape{pad_outer, pad_inner},
      scale.dtype());
  swizzle_scales(scale, scale_tiled, encoder, s);
  encoder.add_temporary(scale_tiled);
  return scale_tiled;
}

QuantizedTensor quantize_input(
    const array& input,
    cu::CommandEncoder& encoder,
    const Stream& s,
    QuantizationMode mode,
    int bits,
    int group_size) {
  const array x = ensure_row_contiguous(input, encoder, s);

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

  // For NVFP4: compute tensor-wide amax for global scaling
  std::optional<array> tensor_amax = std::nullopt;
  if (mode == QuantizationMode::Nvfp4) {
    array amax(cu::malloc_async(sizeof(float), encoder), {}, float32);
    encoder.add_temporary(amax);
    all_reduce(encoder, x, amax, Reduce::ReduceType::AbsMax);
    tensor_amax = amax;
  }

  fp_quantize(x, x_q, scales_x, tensor_amax, group_size, bits, encoder, s);
  return {std::move(x_q), std::move(scales_x), std::move(tensor_amax)};
}

QuantizedTensor get_weight_tensors(
    const std::vector<array>& inputs,
    cu::CommandEncoder& encoder,
    const Stream& s,
    QuantizationMode mode,
    int bits,
    int group_size) {
  // Check if weights need quantization
  if (inputs[1].dtype() != uint32) {
    return quantize_input(inputs[1], encoder, s, mode, bits, group_size);
  }

  // Weights are pre-quantized
  if (mode == QuantizationMode::Nvfp4) {
    // NVFP4: inputs = [x, w_q, scale_w, tensor_amax_w]
    return {inputs[1], inputs[2], inputs[3]};
  }
  // MXFP8: inputs = [x, w_q, scale_w] (no tensor_amax)
  return {inputs[1], inputs[2], std::nullopt};
}

GemmScalars create_nvfp4_scalars(
    const array& tensor_amax_x,
    const array& tensor_amax_w,
    cu::CommandEncoder& encoder) {
  // NVFP4 requires alpha/beta as device pointers
  // alpha = amax_x * amax_w / (448 * 6)^2
  // beta = 0
  array alpha(cu::malloc_async(sizeof(float), encoder), {}, float32);
  array beta(cu::malloc_async(sizeof(float), encoder), {}, float32);
  compute_qqmm_pointers(alpha, beta, tensor_amax_x, tensor_amax_w, encoder);
  encoder.add_temporary(alpha);
  encoder.add_temporary(beta);
  return {alpha, beta};
}

void run_qqmm(
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
    const GemmScalars& scalars) {
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
      out.dtype(),
      qmode);

  if (scalars.uses_device_pointers()) {
    qqmm.run(
        encoder,
        out,
        a,
        b,
        a_scale,
        b_scale,
        *scalars.alpha_device,
        *scalars.beta_device);
  } else {
    qqmm.run(encoder, out, a, b, a_scale, b_scale);
  }
}

} // namespace

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("QQMatmul::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  // Check compute capability (requires Blackwell or newer)
  auto& device = encoder.device();
  int cc = device.compute_capability_major() * 100 +
      device.compute_capability_minor() * 10;
  if (cc < 1000) {
    throw std::runtime_error(
        "[QQMatmul::eval_gpu] QQMM requires compute capability 10.0+");
  }

  size_t expected_size = (mode_ == QuantizationMode::Nvfp4) ? 4 : 3;
  assert(
      (inputs.size() == expected_size && inputs[1].dtype() == uint32) ||
      (inputs.size() == 2));

  // Quantize inputs (or use pre-quantized)
  auto [x_q, scale_x_pre, tensor_amax_x] =
      quantize_input(inputs[0], encoder, s, mode_, bits_, group_size_);
  auto [w_q, scale_w_pre, tensor_amax_w] =
      get_weight_tensors(inputs, encoder, s, mode_, bits_, group_size_);

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  int M = x_q.shape(-2);
  int N = w_q.shape(-2); // transposed
  int K = x_q.shape(-1) * (32 / bits_);

  bool x_transposed = false;
  bool w_transposed = true; // weights are always transposed
  int64_t lda = K;
  int64_t ldb = K;

  // Repack scales to tiled layout for tensor cores
  array scale_x = pad_and_swizzle_scales(scale_x_pre, encoder, s);
  array scale_w = pad_and_swizzle_scales(scale_w_pre, encoder, s);

  GemmScalars scalars;
  if (mode_ == QuantizationMode::Nvfp4) {
    scalars = create_nvfp4_scalars(*tensor_amax_x, *tensor_amax_w, encoder);
  }

  run_qqmm(
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
