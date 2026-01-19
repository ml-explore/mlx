// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/cublas_qqmm.h"
#include "mlx/backend/cuda/quantized/qqmm_utils.h"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

using QuantizedResult = std::tuple<array, array, std::optional<array>>;

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

QuantizedResult quantize_input(
    const array& input,
    cu::CommandEncoder& encoder,
    const Stream& s,
    QuantizationMode mode,
    int bits,
    int group_size) {
  const array x = ensure_row_contiguous(input, encoder, s);

  auto build_shapes = [&](const array& x_in) {
    auto xq_shape = x_in.shape();
    xq_shape.back() = x_in.shape(-1) * bits / 32;

    auto sshape = x_in.shape();
    const int64_t scales_inner = x_in.shape(-1) / group_size;
    auto [pad_outer, pad_inner] =
        get_padded_scale_dims(x_in.shape(-2), scales_inner);
    sshape[x_in.ndim() - 2] = pad_outer;
    sshape[x_in.ndim() - 1] = pad_inner;
    sshape.back() = scales_inner;

    return std::tuple{
        std::move(xq_shape),
        std::move(sshape),
        pad_outer,
        pad_inner,
    };
  };

  auto allocate_outputs = [&](const array& x_in) {
    auto [xq_shape, sshape, pad_outer, pad_inner] = build_shapes(x_in);

    const int64_t xq_bytes = x_in.size() * bits / 8;
    const int64_t batch = x_in.size() / (x_in.shape(-2) * x_in.shape(-1));
    const int64_t scales_bytes = batch * (pad_outer * pad_inner);

    array x_q(cu::malloc_async(xq_bytes, encoder), std::move(xq_shape), uint32);
    array scales_x(
        cu::malloc_async(scales_bytes, encoder), std::move(sshape), uint8);
    encoder.add_temporary(x_q);
    encoder.add_temporary(scales_x);

    return std::pair{std::move(x_q), std::move(scales_x)};
  };

  auto run_quant = [&](const array& x_in, std::optional<array> tensor_amax) {
    auto [x_q, scales_x] = allocate_outputs(x_in);
    fp_quantize(x_in, x_q, scales_x, tensor_amax, group_size, bits, encoder, s);
    return QuantizedResult{
        std::move(x_q), std::move(scales_x), std::move(tensor_amax)};
  };

  if (mode == QuantizationMode::Nvfp4) {
    array tensor_amax(cu::malloc_async(sizeof(float), encoder), {}, float32);
    encoder.add_temporary(tensor_amax);
    all_reduce(encoder, x, tensor_amax, Reduce::ReduceType::AbsMax);
    return run_quant(x, tensor_amax);
  }
  return run_quant(x, std::nullopt);
}

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
    const float alpha) {
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
  auto& device = encoder.device();
  auto cc = device.compute_capability_major() * 100 +
      device.compute_capability_minor() * 10;
  if (cc < 1000) {
    throw std::runtime_error(
        "[QQMatmul::eval_gpu] QQMM is only supported on GPUs with compute capability 10.0 or higher.");
  }
  auto quant_input_size = (mode_ == QuantizationMode::Nvfp4) ? 4 : 3;
  assert(
      (inputs.size() == quant_input_size && inputs[1].dtype() == uint32) ||
      (inputs.size() == 2));

  auto [x_q, scale_x_pre, tensor_amax_x] =
      quantize_input(inputs[0], encoder, s, mode_, bits_, group_size_);
  auto [w_q, scale_w_pre, tensor_amax_w] = (inputs[1].dtype() != uint32)
      ? quantize_input(inputs[1], encoder, s, mode_, bits_, group_size_)
      : QuantizedResult{inputs[1], inputs[2], std::optional<array>(inputs[3])};
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
