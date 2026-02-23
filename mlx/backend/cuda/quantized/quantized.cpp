// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/quantized/qmm.h"
#include "mlx/backend/cuda/quantized/qmv.h"
#include "mlx/backend/cuda/quantized/quantized_utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("QuantizedMatmul::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  const array& x = inputs[0];
  const array& w = inputs[1];
  const array& scales = inputs[2];
  std::optional<array> biases;
  if (inputs.size() > 3) {
    biases = inputs[3];
  }

  bool non_batched = w.ndim() == 2;
  int K = x.shape(-1);
  int N = out.shape(-1);
  int vec_batch = non_batched ? x.size() / K : x.shape(-2);

  if (transpose_ && vec_batch <= 8 && mode_ != QuantizationMode::Affine) {
    assert(!biases);
    fp_qmv(x, w, scales, out, bits_, group_size_, vec_batch, N, K, encoder, s);
    return;
  }

  if (transpose_ && encoder.device().compute_capability_major() == 9) {
    qmm_sm90(x, w, scales, biases, out, bits_, group_size_, mode_, encoder, s);
    return;
  }

  throw std::runtime_error("QMM NYI");
}

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
      // 0 -- xq, 1 -- scales, 2 -- could be global scale for nvfp4
      bool use_global_scale =
          mode_ == QuantizationMode::Nvfp4 && inputs.size() > 2;
      std::optional<array> global_scale =
          use_global_scale ? std::make_optional(inputs[2]) : std::nullopt;
      fp_dequantize(wq, scales, w, group_size_, bits_, global_scale, enc, s);
    }
  } else {
    auto w = ensure_contiguous(inputs[0], enc, s);
    auto& wq = outputs[0];
    auto& scales = outputs[1];

    wq.set_data(cu::malloc_async(wq.nbytes(), enc));
    scales.set_data(cu::malloc_async(scales.nbytes(), enc));

    if (mode_ == QuantizationMode::Affine) {
      auto& biases = outputs[2];
      biases.set_data(cu::malloc_async(biases.nbytes(), enc));
      affine_quantize(w, wq, scales, biases, group_size_, bits_, enc, s);
    } else {
      bool use_global_scale =
          mode_ == QuantizationMode::Nvfp4 && inputs.size() > 1;
      std::optional<array> global_scale =
          use_global_scale ? std::make_optional(inputs[1]) : std::nullopt;
      fp_quantize(w, wq, scales, group_size_, bits_, global_scale, enc, s);
    }
  }
}

} // namespace mlx::core
