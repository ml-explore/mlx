#include "mlx/extensions.h"

namespace mlx::core::ext {

std::vector<array> Extended::vjp(
    const std::vector<array>& primals,
    const std::vector<array>& cotangents,
    const std::vector<int>& argnums,
    const std::vector<array>& outputs) {}

std::vector<array> Extended::jvp(
    const std::vector<array>& primals,
    const std::vector<array>& tangents,
    const std::vector<int>& argnums) {}

std::pair<std::vector<array>, std::vector<int>> Extended::vmap(
    const std::vector<array>& inputs,
    const std::vector<int>& axes) {}

array rms_norm(
    array x,
    const array& w,
    float eps,
    bool precise /* = false */,
    StreamOrDevice s /* = {} */) {
  auto in_type = x.dtype();
  if (precise) {
    x = astype(x, float32, s);
  }
  auto inv_rms =
      rsqrt(add(mean(square(x, s), -1, true, s), array(eps, x.dtype()), s), s);
  return multiply(w, astype(multiply(x, inv_rms, s), in_type, s), s);
}

} // namespace mlx::core::ext
