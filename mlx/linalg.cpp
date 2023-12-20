// Copyright © 2023 Apple Inc.

#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <variant>
#include <vector>

#include "mlx/array.h"
#include "mlx/linalg.h"
#include "mlx/ops.h"
#include "utils.h"

namespace mlx::core::linalg {

inline std::vector<int> get_shape_reducing_over_all_dims(int num_axes) {
  std::vector<int> shape(num_axes);
  std::iota(shape.begin(), shape.end(), 0);
  return shape;
}

array norm(
    const array& a,
    const std::vector<int>& axis,
    bool keepdims,
    StreamOrDevice s) {
  auto num_axes = axis.size();

  if (num_axes == 0 || num_axes == 1 || num_axes == 2)
    return sqrt(sum(
        abs(a, s) * abs(a, s),
        num_axes ? axis : get_shape_reducing_over_all_dims(a.shape().size()),
        keepdims,
        s));

  std::stringstream error_stream;
  error_stream << "Invalid axis values" << axis;
  throw std::invalid_argument(error_stream.str());
}
} // namespace mlx::core::linalg