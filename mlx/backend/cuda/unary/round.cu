// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/unary/unary.cuh"

namespace mlx::core {
void Round::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Round::eval_gpu");
  assert(inputs.size() == 1);
  const auto& in = inputs[0];
  auto& s = out.primitive().stream();
  if (issubdtype(in.dtype(), inexact)) {
    unary_op_gpu<cu::Round>(inputs, out, name(), s);
  } else {
    // No-op integer types
    out.copy_shared_buffer(in);
  }
}
} // namespace mlx::core
