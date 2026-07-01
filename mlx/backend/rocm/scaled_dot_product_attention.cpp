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

#ifdef MLX_HAS_ROCM_WMMA
// Defined in flash_attention_wmma.hip
bool supports_sdpa_flash_wmma(
    const array& q,
    const array& k,
    const array& v,
    bool has_arr_mask,
    bool output_logsumexp);

// LDS bytes the WMMA flash kernel needs for a given head dim.
int sdpa_flash_wmma_smem(int D);

void sdpa_flash_wmma(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    Stream s);
#endif

// Defined in flash_attention.hip
bool supports_sdpa_flash(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool output_logsumexp);

void sdpa_flash(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& mask,
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

bool prefer_flash_for_decode(
    const array& q,
    const array& k,
    bool has_arr_mask,
    bool has_sinks) {
  // The flash (prefill) kernel is catastrophically slow for single-query decode
  // over long contexts — profiled at ~4.7 ms/call at ~1200 keys vs the ~tens of
  // microseconds the vector decode kernel needs (it parallelizes over the KV
  // length). Default decode to the vector kernel; opt back into flash only via
  // env for experimentation.
  static const bool enable =
      std::getenv("MLX_SDPA_DECODE_FLASH") != nullptr;
  if (!enable) {
    return false;
  }
  if (has_arr_mask || has_sinks) {
    return false;
  }
  if (q.shape(2) != 1) {
    return false;
  }
  if (k.shape(2) < 512) {
    return false;
  }
  return q.dtype() == float16 || q.dtype() == bfloat16;
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
    bool /*is_training*/,
    bool output_logsumexp,
    Stream /*s*/) {
  return !supports_sdpa_vector(
             q, k, v, has_mask, has_arr_mask, do_causal, output_logsumexp) &&
      !supports_sdpa_flash(
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

  // Prefer WMMA flash attention when available (bf16/fp16, standard dims).
  // Gate on the device's runtime arch — a multi-arch wheel can include the
  // WMMA kernel even when running on a non-WMMA chip (e.g. gfx1030/1103).
#ifdef MLX_HAS_ROCM_WMMA
  // Gate WMMA on the LDS budget of the device actually running the op: the
  // kernel's tiled footprint must fit this device's shared-memory-per-block.
  bool wmma_supported =
      supports_sdpa_flash_wmma(q, k, v, has_arr_mask, output_logsumexp_) &&
      !has_sinks_ && rocm::device(s.device).has_native_wmma() &&
      sdpa_flash_wmma_smem(q.shape(-1)) <=
          rocm::device(s.device).max_shared_memory_per_block();
#else
  bool wmma_supported = false;
#endif
  bool vector_supported = supports_sdpa_vector(
      q, k, v, has_mask, has_arr_mask, do_causal_, output_logsumexp_);
  bool flash_supported = supports_sdpa_flash(
      q, k, v, has_mask, has_arr_mask, do_causal_, output_logsumexp_);
  bool flash_first = flash_supported &&
      prefer_flash_for_decode(q, k, has_arr_mask, has_sinks_);

  if (wmma_supported && q.shape(2) > 4) {
#ifdef MLX_HAS_ROCM_WMMA
    // Use WMMA kernel for prefill (qL > 4); decode still uses vector kernel
    sdpa_flash_wmma(q, k, v, scale_, out, do_causal_, s);
#endif
  } else if (flash_first) {
    if (has_sinks_) {
      sdpa_flash(q, k, v, scale_, out, do_causal_, mask_arr, inputs.back(), s);
    } else {
      sdpa_flash(q, k, v, scale_, out, do_causal_, mask_arr, std::nullopt, s);
    }
  } else if (vector_supported) {
    if (has_sinks_) {
      sdpa_vector(q, k, v, scale_, out, do_causal_, inputs.back(), s);
    } else {
      sdpa_vector(q, k, v, scale_, out, do_causal_, std::nullopt, s);
    }
  } else if (flash_supported) {
    if (has_sinks_) {
      sdpa_flash(q, k, v, scale_, out, do_causal_, mask_arr, inputs.back(), s);
    } else {
      sdpa_flash(q, k, v, scale_, out, do_causal_, mask_arr, std::nullopt, s);
    }
  } else {
    // This should not be reached — use_fallback() returns true for unsupported
    // configs, causing the framework to decompose SDPA into basic GPU ops
    // (matmul + softmax + matmul) before this primitive is created.
    throw std::runtime_error(
        "[ScaledDotProductAttention::eval_gpu] Unsupported configuration reached. "
        "This is a bug — use_fallback() should have returned true.");
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
