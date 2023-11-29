#include "mlx/primitives.h"

namespace mlx::core {

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& in = inputs[0];
  throw std::runtime_error("[FFT] NYI for Metal backend.");
}

} // namespace mlx::core
