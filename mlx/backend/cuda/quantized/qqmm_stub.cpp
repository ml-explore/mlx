#include <stdexcept>
#include "mlx/primitives.h"

namespace mlx::core {
void QQMatmul::eval_gpu(const std::vector<array>&, array&) {
  throw std::runtime_error(
      "QQMatmul is only built for SM100+ and CUDA version >= 12.8.");
}
} // namespace mlx::core