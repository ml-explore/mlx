// Copyright © 2023 Apple Inc.

#include <numeric>
#include <set>

#include "mlx/linalg.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::linalg {

array norm(
    const array& a,
    const std::variant<std::monostate, int, std::string>& ord,
    const std::vector<int>& axes,
    StreamOrDevice s = {}) {
  if (a.ndim() < 1) {
    throw std::invalid_argument(
        "[linalg.norm] Requires array with at least one dimension.");
  }
  // Axes validation
  if (axes.size() > 2) {
    throw std::invalid_argument(
        "[linalg.norm] Invalid number of dimensions to axes.");
  }
  auto valid_axes = [&] {
    if (axes.size() == 2 && (a.ndim() != 2 || axes[0] == axes[1])) {
      throw std::invalid_argument(
          axes[0] == axes[1]
              ? "[linalg.norm] Two axes specified, but they are the same."
              : "[linalg.norm] Two axes specified, but array is not 2D.");
    }
    std::vector<size_t> v;
    std::ranges::transform(axes, std::back_inserter(v), [&](int ax) {
      return ax < 0 ? ax + a.ndim() : ax;
    });
    return v;
  }();

  // Ord validation
  if (valid_axes.size() < 2) {
    if (valid_axes.size() == 1 && std::holds_alternative<std::string>(ord)) {
      throw std::invalid_argument(
          "[linalg.norm] String ord (fro, nuc) not supported for vectors.");
    }
    int ord_int = std::get<int>(ord);
    if (ord_int != 0 && ord_int != 1 && ord_int != 2) {
      throw std::invalid_argument(
          "[linalg.norm] Invalid norm order. Must be one of {1, 2, 3}.");
    }
  }
  return array(std::make_unique<Norm>(to_stream(a), ord, valid_axes));
}
array norm(
    const array& a,
    const std::variant<std::monostate, int, std::string>& ord,
    StreamOrDevice s = {}) {
  return norm(a, ord, {}, s);
}
array norm(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s = {}) {
  return norm(a, std::monostate{}, axes, s);
}
array norm(const array& a, StreamOrDevice s = {}) {
  return norm(a, std::monostate{}, {}, s);
}
} // namespace mlx::core::linalg