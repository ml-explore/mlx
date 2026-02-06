// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

// Defined in scaled_dot_product_attention.hip
bool supports_sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool output_logsumexp);

void sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& sinks,
    Stream s);

namespace {

array prepare_sdpa_input(const array& x, Stream s) {
  // SDPA kernel requirements: last dim stride be 1, pointer aligned
  if (x.strides(-1) != 1) {
    array x_copy = contiguous_copy_gpu(x, s);
    auto& d = rocm::device(s.device);
    auto& encoder = d.get_command_encoder(s);
    encoder.add_temporary(x_copy);
    return x_copy;
  }
  return x;
}

} // namespace

namespace fast {

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  if (s.device == Device::cpu) {
    return true;
  }

  // Use fallback if we don't support the vector kernel
  return !supports_sdpa_vector(
      q, k, v, has_mask, has_arr_mask, do_causal, output_logsumexp);
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();

  array q = prepare_sdpa_input(inputs[0], s);
  array k = prepare_sdpa_input(inputs[1], s);
  array v = prepare_sdpa_input(inputs[2], s);
  auto& out = outputs[0];
  auto& stats = outputs[1];
  bool has_mask = inputs.size() - has_sinks_ > 3;
  bool has_arr_mask = has_mask && !do_causal_;

  std::optional<array> mask_arr;
  if (has_arr_mask) {
    mask_arr = prepare_sdpa_input(inputs[3], s);
  }

  if (supports_sdpa_vector(
          q, k, v, has_mask, has_arr_mask, do_causal_, output_logsumexp_)) {
    if (has_sinks_) {
      sdpa_vector(q, k, v, scale_, out, do_causal_, inputs.back(), s);
    } else {
      sdpa_vector(q, k, v, scale_, out, do_causal_, std::nullopt, s);
    }
  } else {
    // Fallback: compute attention manually
    // This path should rarely be hit due to use_fallback check
    throw std::runtime_error(
        "SDPA configuration not supported by ROCm kernel. "
        "Please use CPU fallback or adjust parameters.");
  }
}

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  // Always use fallback for VJP on ROCm for now
  return true;
}

void ScaledDotProductAttentionVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // VJP uses CPU fallback
  throw std::runtime_error(
      "SDPA VJP not yet implemented for ROCm. Using CPU fallback.");
}

} // namespace fast

} // namespace mlx::core
