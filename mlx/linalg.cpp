// Copyright © 2023 Apple Inc.

#include <iostream>
#include <numeric>
#include <set>
#include <variant>

#include "mlx/array.h"
#include "mlx/linalg.h"
#include "mlx/ops.h"

namespace mlx::core::linalg {

array vector_norm(
    const array& a,
    const std::variant<double, std::string>& ord,
    const std::vector<int>& axes,
    bool keepdims,
    StreamOrDevice s) {
  return std::visit(
      overloaded{
          [&](double p) {
            if (p >= 1)
              return power(
                  sum(power(abs(a, s), array(p), s), axes, keepdims, s),
                  array(1.0 / p),
                  s);
            else if (p == 0)
              return sum(
                  where(a != 0, array(1), array(0), s), axes, keepdims, s);
            else
              throw std::invalid_argument(
                  "[core.linalg.norm] p norm is defined only for p >= 1.");
          },
          [&](const std::string& norm_type) {
            if (norm_type == "inf")
              return max(abs(a, s), axes, keepdims, s);
            else if (norm_type == "-inf")
              return min(abs(a, s), axes, keepdims, s);
            else
              throw std::invalid_argument(
                  "[core.linalg.norm] Unsupported norm type for a vector.");
          }},
      ord);
}
array vector_norm(
    const array& a,
    const std::variant<double, std::string>& ord,
    bool keepdims,
    StreamOrDevice s) {
  return vector_norm(
      reshape(a, {static_cast<int>(a.size())}), ord, {-1}, keepdims, s);
}
array vector_norm(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims,
    StreamOrDevice s) {
  return vector_norm(a, 2.0, axes, keepdims, s);
}
array vector_norm(const array& a, bool keepdims, StreamOrDevice s) {
  return vector_norm(a, 2.0, keepdims, s);
}
} // namespace mlx::core::linalg