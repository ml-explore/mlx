// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/fast_primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

// ROCm does not have cuDNN equivalent (MIOpen) integrated yet
// These functions return false to indicate fallback should be used

bool supports_sdpa_rocm(
    const array& q,
    const array& k,
    const array& v,
    bool do_causal,
    Stream s) {
  // MIOpen integration not yet implemented
  return false;
}

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
  // Always use fallback on ROCm until MIOpen integration is complete
  return true;
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return false;
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error(
      "ScaledDotProductAttention::eval_gpu requires MIOpen integration for ROCm. "
      "Please use the CPU fallback or wait for MIOpen support.");
}

bool ScaledDotProductAttentionVJP::use_fallback(const array& q, Stream s) {
  // Always use fallback on ROCm
  return true;
}

void ScaledDotProductAttentionVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error(
      "ScaledDotProductAttentionVJP::eval_gpu requires MIOpen integration for ROCm. "
      "Please use the CPU fallback or wait for MIOpen support.");
}

} // namespace fast

} // namespace mlx::core
