// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/scaled_dot_product_attention.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

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

  return !supports_sdpa_vector(q, k, v, has_mask, has_arr_mask, do_causal);
}

void ScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  nvtx3::scoped_range r("ScaledDotProductAttention::eval_gpu");

  auto& s = stream();

  assert(inputs.size() == 3 || inputs.size() == 4);
  const auto& q = inputs[0];
  const auto& k = inputs[1];
  const auto& v = inputs[2];

  if (has_sinks_) {
    sdpa_vector(q, k, v, scale_, out, do_causal_, inputs.back(), s);
  } else {
    sdpa_vector(q, k, v, scale_, out, do_causal_, std::nullopt, s);
  }
}

} // namespace fast

} // namespace mlx::core
