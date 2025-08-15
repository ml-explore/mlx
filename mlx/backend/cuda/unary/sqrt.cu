// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/unary/unary.cuh"

namespace mlx::core {
void Sqrt::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Sqrt::eval_gpu");
  auto& s = out.primitive().stream();
  if (recip_) {
    unary_op_gpu<cu::Rsqrt>(inputs, out, "Rsqrt", s);
  } else {
    unary_op_gpu<cu::Sqrt>(inputs, out, "Sqrt", s);
  }
}
} // namespace mlx::core
