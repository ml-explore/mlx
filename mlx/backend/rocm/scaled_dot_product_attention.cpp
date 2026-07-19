// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>
#include <cstdlib>

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

// lse: optional float32 [B,H,qL,1] for training VJP (null when not requested).
void sdpa_flash_wmma(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    Stream s,
    array* lse = nullptr);

// Defined in flash_attention_bwd_wmma.hip
bool supports_sdpa_flash_wmma_bwd(
    const array& q,
    const array& k,
    const array& v,
    bool has_arr_mask,
    bool has_sinks);
int sdpa_flash_bwd_smem(int D);

void sdpa_flash_wmma_bwd(
    const array& q,
    const array& k,
    const array& v,
    const array& o,
    const array& lse,
    const array& d_o,
    float scale,
    bool do_causal,
    array& d_q,
    array& d_k,
    array& d_v,
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
  static const bool enable = std::getenv("MLX_SDPA_DECODE_FLASH") != nullptr;
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
    Stream s) {
  // Vector + scalar flash reject output_logsumexp. WMMA flash can emit LSE and
  // must be counted here or MLX_SDPA_FLASH_VJP never attaches (frontend only
  // requests LSE when VJP is non-fallback, and VJP needs LSE forward).
  // Note: this is static — no has_sinks_ available. Sinks + LSE is rare; eval
  // still refuses sinks on the WMMA path.
  bool vector_ok = supports_sdpa_vector(
      q, k, v, has_mask, has_arr_mask, do_causal, output_logsumexp);
  bool flash_ok = supports_sdpa_flash(
      q, k, v, has_mask, has_arr_mask, do_causal, output_logsumexp);
#ifdef MLX_HAS_ROCM_WMMA
  static const bool wmma_disabled = std::getenv("MLX_SDPA_NO_WMMA") != nullptr;
  bool wmma_ok = !wmma_disabled &&
      supports_sdpa_flash_wmma(q, k, v, has_arr_mask, output_logsumexp) &&
      s.device == Device::gpu && rocm::device(s.device).has_native_wmma() &&
      sdpa_flash_wmma_smem(q.shape(-1)) <=
          rocm::device(s.device).max_shared_memory_per_block();
  // LSE train path needs qL > 4 so eval_gpu takes WMMA (vector is decode-only).
  if (output_logsumexp) {
    wmma_ok = wmma_ok && q.shape(2) > 4;
  }
#else
  bool wmma_ok = false;
  (void)s;
#endif
  return !vector_ok && !flash_ok && !wmma_ok;
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
  // Escape hatch for A/B profiling: MLX_SDPA_NO_WMMA=1 forces the scalar flash
  // / vector path even where the matrix-core kernel is available.
  static const bool wmma_disabled = std::getenv("MLX_SDPA_NO_WMMA") != nullptr;
  bool wmma_supported = !wmma_disabled &&
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
    // Use WMMA kernel for prefill (qL > 4); decode still uses vector kernel.
    // When output_logsumexp_ is set (training + fast VJP), write LSE into
    // outputs[1] for the fused backward. VJP itself is still gated separately.
    array* lse_ptr = nullptr;
    if (output_logsumexp_ && outputs.size() > 1) {
      lse_ptr = &stats;
    }
    sdpa_flash_wmma(q, k, v, scale_, out, do_causal_, s, lse_ptr);
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
  // Opt-in fused flash VJP (LSE forward + tiled bwd). Default OFF until
  // lemonseed train A/B is green. Kill-switch: MLX_SDPA_NO_FLASH_VJP=1.
  static const bool no_flash_vjp =
      std::getenv("MLX_SDPA_NO_FLASH_VJP") != nullptr;
  static const bool want =
      std::getenv("MLX_SDPA_FLASH_VJP") != nullptr && !no_flash_vjp;
  if (!want || s.device == Device::cpu) {
    return true;
  }
#ifdef MLX_HAS_ROCM_WMMA
  // Shape gates match supports_sdpa_flash_wmma_bwd (no mask/sinks, D=64).
  if (q.dtype() != bfloat16 && q.dtype() != float16) {
    return true;
  }
  if (q.shape(-1) != 64) {
    return true;
  }
  if (!rocm::device(s.device).has_native_wmma()) {
    return true;
  }
  if (sdpa_flash_bwd_smem(q.shape(-1)) >
      rocm::device(s.device).max_shared_memory_per_block()) {
    return true;
  }
  return false;
#else
  (void)q;
  return true;
#endif
}

void ScaledDotProductAttentionVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
#ifdef MLX_HAS_ROCM_WMMA
  auto& s = stream();
  assert(inputs.size() >= 6);
  int primals_size = static_cast<int>(inputs.size()) - 3;
  bool has_arr_mask = primals_size > 3 + static_cast<int>(has_sinks_);

  array q = prepare_sdpa_input(inputs[0], s);
  array k = prepare_sdpa_input(inputs[1], s);
  array v = prepare_sdpa_input(inputs[2], s);
  array o = prepare_sdpa_input(inputs[primals_size], s);
  array stats = prepare_sdpa_input(inputs[primals_size + 1], s);
  array d_o = prepare_sdpa_input(inputs[primals_size + 2], s);

  if (has_arr_mask || has_sinks_) {
    throw std::runtime_error(
        "SDPA flash VJP does not support array masks or sinks yet.");
  }
  if (!supports_sdpa_flash_wmma_bwd(
          q, k, v, /*has_arr_mask=*/false, has_sinks_) ||
      sdpa_flash_bwd_smem(q.shape(-1)) >
          rocm::device(s.device).max_shared_memory_per_block()) {
    throw std::runtime_error(
        "SDPA flash VJP shape/LDS not supported (need D=64 bf16/fp16, LDS fit).");
  }

  assert(outputs.size() == 3);
  auto& d_q = outputs[0];
  auto& d_k = outputs[1];
  auto& d_v = outputs[2];

  sdpa_flash_wmma_bwd(
      q, k, v, o, stats, d_o, scale_, do_causal_, d_q, d_k, d_v, s);
#else
  throw std::runtime_error("SDPA flash VJP requires MLX_HAS_ROCM_WMMA.");
#endif
}

} // namespace fast

} // namespace mlx::core
