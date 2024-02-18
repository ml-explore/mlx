#include "mlx/primitives.h"

namespace mlx::core {
namespace {}

void Pooling::eval_gpu(const std::vector<array>& inputs, array& output) {
  throw std::runtime_error("[Pooling] only max_pool_1d is supported for now.");
}
} // namespace mlx::core