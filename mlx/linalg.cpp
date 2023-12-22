// Copyright © 2023 Apple Inc.

#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/array.h"
#include "mlx/linalg.h"
#include "mlx/ops.h"
#include "utils.h"

namespace mlx::core::linalg {

inline array vector_norm(
    const array& a,
    const double ord,
    const std::vector<int>& axis,
    bool keepdims,
    StreamOrDevice s) {
  if (ord == 0.0)
    return sum(a != 0, axis, keepdims, s);
  else if (ord == 1.0)
    return sum(abs(a, s), axis, keepdims, s);
  else if (ord == 2.0)
    return sqrt(sum(abs(a, s) * abs(a, s), axis, keepdims, s));
  else
    return power(
        sum(power(abs(a, s), array(ord), s), axis, keepdims, s),
        array(1.0 / ord));
}

inline array vector_norm(
    const array& a,
    const std::string& ord,
    const std::vector<int>& axis,
    bool keepdims,
    StreamOrDevice s) {
  if (ord == "inf")
    return max(abs(a, s), axis, keepdims, s);
  else if (ord == "-inf")
    return min(abs(a, s), axis, keepdims, s);
  std::stringstream error_stream;
  error_stream << "Invalid ord value " << ord;
  throw std::invalid_argument(error_stream.str());
}

inline array matrix_norm(
    const array& a,
    const double ord,
    const std::vector<int>& axis,
    bool keepdims,
    StreamOrDevice s) {
  auto row_axis = axis[0];
  auto col_axis = axis[1];
  if (!keepdims && col_axis > row_axis)
    col_axis -= 1;
  if (ord == -1.0)
    return min(sum(abs(a, s), row_axis, keepdims, s), col_axis, keepdims, s);
  if (ord == 1.0)
    return max(sum(abs(a, s), row_axis, keepdims, s), col_axis, keepdims, s);
  if (ord == 2.0 || ord == -2.0)
    throw std::logic_error("Singular value norms are not implemented.");
  std::stringstream error_stream;
  error_stream << "Invalid ord value " << ord << " for matrix norm";
  throw std::invalid_argument(error_stream.str());
}

inline array matrix_norm(
    const array& a,
    const std::string& ord,
    const std::vector<int>& axis,
    bool keepdims,
    StreamOrDevice s) {
  if (ord == "f" || ord == "fro")
    return sqrt(sum(abs(a, s) * abs(a, s), axis, keepdims, s));
  else if (ord == "inf")
    return matrix_norm(a, 1.0, {axis[1], axis[0]}, keepdims, s);
  else if (ord == "-inf")
    return matrix_norm(a, -1.0, {axis[1], axis[0]}, keepdims, s);
  if (ord == "nuc")
    throw std::logic_error("Nuclear norm is not implemented.");
  std::stringstream error_stream;
  error_stream << "Invalid ord value " << ord << " for matrix norm";
  throw std::invalid_argument(error_stream.str());
}

array norm(
    const array& a,
    const std::vector<int>& axis,
    bool keepdims,
    StreamOrDevice s) {
  auto num_axes = axis.size();

  if (num_axes == 0 || num_axes == 1 || num_axes == 2)
    return sqrt(
        sum(abs(a, s) * abs(a, s),
            num_axes ? axis : get_reduce_axes({}, a.ndim()),
            keepdims,
            s),
        s);

  std::stringstream error_stream;
  error_stream << "Invalid axis values " << axis;
  throw std::invalid_argument(error_stream.str());
}

array norm(
    const array& a,
    const double ord,
    const std::vector<int>& axis,
    bool keepdims,
    StreamOrDevice s) {
  std::vector<int> ax = axis;

  if (axis.empty())
    ax = get_reduce_axes({}, a.ndim());
  else
    ax = normalize_axes(ax, a.ndim());

  auto num_axes = ax.size();
  if (num_axes == 1)
    return vector_norm(a, ord, ax, keepdims, s);
  else if (num_axes == 2)
    return matrix_norm(a, ord, ax, keepdims, s);

  std::stringstream error_stream;
  error_stream << "Invalid axis values " << ax;
  throw std::invalid_argument(error_stream.str());
}

array norm(
    const array& a,
    const std::string& ord,
    const std::vector<int>& axis,
    bool keepdims,
    StreamOrDevice s) {
  std::vector<int> ax = axis;

  if (axis.empty())
    ax = get_reduce_axes({}, a.ndim());
  else
    ax = normalize_axes(ax, a.ndim());

  auto num_axes = ax.size();
  if (num_axes == 1)
    return vector_norm(a, ord, ax, keepdims, s);
  else if (num_axes == 2)
    return matrix_norm(a, ord, ax, keepdims, s);

  std::stringstream error_stream;
  error_stream << "Invalid axis values " << ax;
  throw std::invalid_argument(error_stream.str());
}

} // namespace mlx::core::linalg