// Copyright © 2023 Apple Inc.

#include <numeric>
#include <ostream>
#include <vector>

#include "mlx/linalg.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::linalg {

Dtype at_least_float(const Dtype& d) {
  return issubdtype(d, inexact) ? d : promote_types(d, float32);
}

inline array l2_norm(
    const array& a,
    const std::vector<int>& axis,
    bool keepdims,
    StreamOrDevice s) {
  if (issubdtype(a.dtype(), complexfloating)) {
    return sqrt(sum(abs(a, s) * abs(a, s), axis, keepdims, s), s);
  } else {
    return sqrt(sum(square(a, s), axis, keepdims, s), s);
  }
}

inline array vector_norm(
    const array& a,
    const double ord,
    const std::vector<int>& axis,
    bool keepdims,
    StreamOrDevice s) {
  auto dtype = at_least_float(a.dtype());
  if (ord == 0.0) {
    return astype(sum(not_equal(a, array(0), s), axis, keepdims, s), dtype, s);
  } else if (ord == 1.0) {
    return astype(sum(abs(a, s), axis, keepdims, s), dtype, s);
  } else if (ord == 2.0) {
    return l2_norm(a, axis, keepdims, s);
  } else if (ord == std::numeric_limits<double>::infinity()) {
    return astype(max(abs(a, s), axis, keepdims, s), dtype, s);
  } else if (ord == -std::numeric_limits<double>::infinity()) {
    return astype(min(abs(a, s), axis, keepdims, s), dtype, s);
  } else {
    return power(
        sum(power(abs(a, s), array(ord, dtype), s), axis, keepdims, s),
        array(1.0 / ord, dtype),
        s);
  }
}

inline array matrix_norm(
    const array& a,
    const double ord,
    const std::vector<int>& axis,
    bool keepdims,
    StreamOrDevice s) {
  auto dtype = at_least_float(a.dtype());
  auto row_axis = axis[0];
  auto col_axis = axis[1];
  if (ord == -1.0) {
    col_axis -= (!keepdims && col_axis > row_axis && col_axis > 0);
    return astype(
        min(sum(abs(a, s), row_axis, keepdims, s), col_axis, keepdims, s),
        dtype,
        s);
  } else if (ord == 1.0) {
    col_axis -= (!keepdims && col_axis > row_axis && col_axis > 0);
    return astype(
        max(sum(abs(a, s), row_axis, keepdims, s), col_axis, keepdims, s),
        dtype,
        s);
  } else if (ord == std::numeric_limits<double>::infinity()) {
    row_axis -= (!keepdims && row_axis > col_axis && row_axis > 0);
    return astype(
        max(sum(abs(a, s), col_axis, keepdims, s), row_axis, keepdims, s),
        dtype,
        s);
  } else if (ord == -std::numeric_limits<double>::infinity()) {
    row_axis -= (!keepdims && row_axis > col_axis && row_axis > 0);
    return astype(
        min(sum(abs(a, s), col_axis, keepdims, s), row_axis, keepdims, s),
        dtype,
        s);
  } else if (ord == 2.0 || ord == -2.0) {
    throw std::runtime_error(
        "[linalg::norm] Singular value norms are not implemented.");
  } else {
    std::ostringstream msg;
    msg << "[linalg::norm] Invalid ord " << ord << " for matrix norm.";
    throw std::invalid_argument(msg.str());
  }
}

inline array matrix_norm(
    const array& a,
    const std::string& ord,
    const std::vector<int>& axis,
    bool keepdims,
    StreamOrDevice s) {
  if (ord == "f" || ord == "fro") {
    return l2_norm(a, axis, keepdims, s);
  } else if (ord == "nuc") {
    throw std::runtime_error(
        "[linalg::norm] Nuclear norm not yet implemented.");
  } else {
    std::ostringstream msg;
    msg << "[linalg::norm] Invalid ord value '" << ord << "' for matrix norm.";
    throw std::invalid_argument(msg.str());
  }
}

array norm(
    const array& a,
    const std::optional<std::vector<int>>& axis /* = std::nullopt */,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  if (!axis) {
    return norm(flatten(a, s), std::vector<int>{0}, keepdims, s);
  }

  if (axis.value().size() > 2) {
    throw std::invalid_argument(
        "[linalg::norm] Received too many axes for norm.");
  }
  return l2_norm(a, axis.value(), keepdims, s);
}

array norm(
    const array& a,
    const double ord,
    const std::optional<std::vector<int>>& axis /* = std::nullopt */,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  std::vector<int> ax;
  if (!axis) {
    ax.resize(a.ndim());
    std::iota(ax.begin(), ax.end(), 0);
  } else {
    ax = axis.value();
  }
  if (ax.size() == 1) {
    return vector_norm(a, ord, ax, keepdims, s);
  } else if (ax.size() == 2) {
    return matrix_norm(a, ord, ax, keepdims, s);
  } else {
    throw std::invalid_argument(
        "[linalg::norm] Received too many axes for norm.");
  }
}

array norm(
    const array& a,
    const std::string& ord,
    const std::optional<std::vector<int>>& axis /* = std::nullopt */,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  std::vector<int> ax;
  if (!axis) {
    ax.resize(a.ndim());
    std::iota(ax.begin(), ax.end(), 0);
  } else {
    ax = axis.value();
  }
  if (ax.size() != 2) {
    std::ostringstream msg;
    msg << "[linalg::norm] Norm '" << ord << "' only supported for matrices,"
        << " but received " << ax.size() << " axis/axes.";
    throw std::invalid_argument(msg.str());
  }
  return matrix_norm(a, ord, ax, keepdims, s);
}

std::pair<array, array> qr(const array& a, StreamOrDevice s /* = {} */) {
  if (a.dtype() != float32) {
    std::ostringstream msg;
    msg << "[linalg::qr] Arrays must type float32. Received array "
        << "with type " << a.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (a.ndim() < 2) {
    std::ostringstream msg;
    msg << "[linalg::qr] Arrays must have >= 2 dimensions. Received array "
           "with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (a.shape(-1) != a.shape(-2)) {
    throw std::invalid_argument(
        "[linalg::qr] Support for non-square matrices NYI.");
  }

  auto out = array::make_arrays(
      {a.shape(), a.shape()},
      {a.dtype(), a.dtype()},
      std::make_shared<QRF>(to_stream(s)),
      {astype(a, a.dtype(), s)});
  return std::make_pair(out[0], out[1]);
}

std::vector<array> svd(const array& a, StreamOrDevice s /* = {} */) {
  if (a.dtype() != float32) {
    std::ostringstream msg;
    msg << "[linalg::svd] Input array must have type float32. Received array "
        << "with type " << a.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (a.ndim() < 2) {
    std::ostringstream msg;
    msg << "[linalg::svd] Input array must have >= 2 dimensions. Received array "
           "with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  const auto m = a.shape(-2);
  const auto n = a.shape(-1);
  const auto rank = a.ndim();

  std::vector<int> u_shape = a.shape();
  u_shape[rank - 2] = m;
  u_shape[rank - 1] = m;

  std::vector<int> s_shape = a.shape();
  s_shape.pop_back();
  s_shape[rank - 2] = std::min(m, n);

  std::vector<int> vt_shape = a.shape();
  vt_shape[rank - 2] = n;
  vt_shape[rank - 1] = n;

  return array::make_arrays(
      {u_shape, s_shape, vt_shape},
      {a.dtype(), a.dtype(), a.dtype()},
      std::make_shared<SVD>(to_stream(s)),
      {a});
}

array inv(const array& a, StreamOrDevice s /* = {} */) {
  if (a.dtype() != float32) {
    std::ostringstream msg;
    msg << "[linalg::inv] Arrays must type float32. Received array "
        << "with type " << a.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (a.ndim() < 2) {
    std::ostringstream msg;
    msg << "[linalg::inv] Arrays must have >= 2 dimensions. Received array "
           "with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (a.shape(-1) != a.shape(-2)) {
    throw std::invalid_argument(
        "[linalg::inv] Inverses are only defined for square matrices.");
  }

  return array(
      a.shape(), a.dtype(), std::make_shared<Inverse>(to_stream(s)), {a});
}

} // namespace mlx::core::linalg
