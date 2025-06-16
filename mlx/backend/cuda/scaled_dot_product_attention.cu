// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/fast_primitives.h"

namespace mlx::core {

namespace cu {} // namespace cu

namespace fast {

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    Stream s) {
  if (detail::in_grad_tracing()) {
    return true;
  }
  if (s.device == Device::cpu) {
    return true;
  }

  const int value_head_dim = v.shape(-1);
  const int query_head_dim = q.shape(-1);
  const int query_sequence_length = q.shape(2);
  const int key_sequence_length = k.shape(2);

  const bool sdpa_vector_supported_head_dim =
      query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 96 || query_head_dim == 128 ||
       query_head_dim == 256);
  const bool supports_sdpa_vector = (query_sequence_length <= 1) &&
      (query_sequence_length <= key_sequence_length) &&
      sdpa_vector_supported_head_dim;

  return !supports_sdpa_vector;
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  nvtx3::scoped_range r("ScaledDotProductAttention::eval_gpu");
}

} // namespace fast

} // namespace mlx::core
