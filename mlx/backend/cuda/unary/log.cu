// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/unary/unary.cuh"

namespace mlx::core {
void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Log::eval_gpu");
  auto& s = out.primitive().stream();
  switch (base_) {
    case Base::e:
      unary_op_gpu<cu::Log>(inputs, out, name(), s);
      break;
    case Base::two:
      unary_op_gpu<cu::Log2>(inputs, out, name(), s);
      break;
    case Base::ten:
      unary_op_gpu<cu::Log10>(inputs, out, name(), s);
      break;
  }
}
} // namespace mlx::core
