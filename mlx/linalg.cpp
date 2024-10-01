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

array inv_impl(const array& a, bool tri, bool upper, StreamOrDevice s) {
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
      a.shape(),
      a.dtype(),
      std::make_shared<Inverse>(to_stream(s), tri, upper),
      {a});
}

array inv(const array& a, StreamOrDevice s /* = {} */) {
  return inv_impl(a, /*tri=*/false, /*upper=*/true, s);
}

array tri_inv(
    const array& a,
    bool upper /* = true */,
    StreamOrDevice s /* = {} */) {
  return inv_impl(a, /*tri=*/true, upper, s);
}

array cholesky(
    const array& a,
    bool upper /* = false */,
    StreamOrDevice s /* = {} */) {
  if (a.dtype() != float32) {
    std::ostringstream msg;
    msg << "[linalg::cholesky] Arrays must type float32. Received array "
        << "with type " << a.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  if (a.ndim() < 2) {
    std::ostringstream msg;
    msg << "[linalg::cholesky] Arrays must have >= 2 dimensions. Received array "
           "with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  if (a.shape(-1) != a.shape(-2)) {
    throw std::invalid_argument(
        "[linalg::cholesky] Cholesky decomposition is only defined for square "
        "matrices.");
  }
  return array(
      a.shape(),
      a.dtype(),
      std::make_shared<Cholesky>(to_stream(s), upper),
      {a});
}

array pinv(const array& a, StreamOrDevice s /* = {} */) {
  if (a.dtype() != float32) {
    std::ostringstream msg;
    msg << "[linalg::pinv] Arrays must type float32. Received array "
        << "with type " << a.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (a.ndim() < 2) {
    std::ostringstream msg;
    msg << "[linalg::pinv] Arrays must have >= 2 dimensions. Received array "
        << "with " << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  int m = a.shape(-2);
  int n = a.shape(-1);
  int k = std::min(m, n);
  auto outs = linalg::svd(a, s);
  array U = outs[0];
  array S = outs[1];
  array V = outs[2];

  std::vector<int> starts(a.ndim(), 0);
  std::vector<int> ends = a.shape();
  int i = a.ndim() - 2;
  int j = a.ndim() - 1;

  // Prepare U
  ends[i] = m;
  ends[j] = k;
  U = swapaxes(slice(U, starts, ends, s), -1, -2, s);

  // Prepare V
  ends[i] = k;
  ends[j] = n;
  V = swapaxes(slice(V, starts, ends, s), -1, -2, s);

  // Prepare S
  S = expand_dims(S, -2, s);

  return matmul(divide(V, S, s), U);
}

array cholesky_inv(
    const array& L,
    bool upper /* = false */,
    StreamOrDevice s /* = {} */) {
  if (L.dtype() != float32) {
    std::ostringstream msg;
    msg << "[linalg::cholesky] Arrays must type float32. Received array "
        << "with type " << L.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  if (L.ndim() < 2) {
    std::ostringstream msg;
    msg << "[linalg::cholesky] Arrays must have >= 2 dimensions. Received array "
           "with "
        << L.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  if (L.shape(-1) != L.shape(-2)) {
    throw std::invalid_argument(
        "[linalg::cholesky] Cholesky inverse is only defined for square "
        "matrices.");
  }

  array L_inv = tri_inv(L, upper, s);
  if (upper) {
    return matmul(L_inv, swapaxes(L_inv, -1, -2, s), s);
  } else {
    return matmul(swapaxes(L_inv, -1, -2, s), L_inv, s);
  }
}

array cross(
    const array& a,
    const array& b,
    int axis /* = -1 */,
    StreamOrDevice s /* = {} */) {
  auto check_ax = [axis](const array& arr) {
    if (axis >= static_cast<int>(arr.ndim()) || axis + arr.ndim() < 0) {
      std::ostringstream msg;
      msg << "[linalg::cross] axis " << axis << " invalid for array with "
          << arr.ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if (arr.shape(axis) < 2 || arr.shape(axis) > 3) {
      throw std::invalid_argument(
          "[linalg::cross] The specified axis must have size 2 or 3.");
    }
  };
  check_ax(a);
  check_ax(b);

  bool a_2d = a.shape(axis) == 2;
  bool b_2d = b.shape(axis) == 2;

  auto out_type = promote_types(a.dtype(), b.dtype());
  auto ashape = a.shape();
  auto bshape = b.shape();

  ashape[axis < 0 ? axis + a.ndim() : axis] = 3;
  bshape[axis < 0 ? axis + b.ndim() : axis] = 3;
  auto out_shape = broadcast_shapes(ashape, bshape);

  if (axis < 0) {
    axis += out_shape.size();
  }

  out_shape[axis] = a_2d ? 2 : 3;
  auto a_ = broadcast_to(astype(a, out_type, s), out_shape, s);

  out_shape[axis] = b_2d ? 2 : 3;
  auto b_ = broadcast_to(astype(b, out_type, s), out_shape, s);

  auto a_splits = split(a_, a_2d ? 2 : 3, axis);
  auto b_splits = split(b_, b_2d ? 2 : 3, axis);

  std::vector<array> outputs;
  if (a_2d && b_2d) {
    auto z = zeros_like(a_splits[0], s);
    outputs.push_back(z);
    outputs.push_back(z);
  } else if (b_2d) {
    outputs.push_back(negative(multiply(a_splits[2], b_splits[1], s), s));
    outputs.push_back(multiply(a_splits[2], b_splits[0], s));
  } else if (a_2d) {
    outputs.push_back(multiply(a_splits[1], b_splits[2], s));
    outputs.push_back(negative(multiply(a_splits[0], b_splits[2], s), s));
  } else {
    outputs.push_back(subtract(
        multiply(a_splits[1], b_splits[2], s),
        multiply(a_splits[2], b_splits[1], s),
        s));
    outputs.push_back(subtract(
        multiply(a_splits[2], b_splits[0], s),
        multiply(a_splits[0], b_splits[2], s),
        s));
  }
  outputs.push_back(subtract(
      multiply(a_splits[0], b_splits[1], s),
      multiply(a_splits[1], b_splits[0], s),
      s));
  return concatenate(outputs, axis, s);
}

array eigvalsh(
    const array& a,
    bool upper /* = false */,
    StreamOrDevice s /* = {} */) {
  if (a.dtype() != float32) {
    std::ostringstream msg;
    msg << "[linalg::eigvalsh] Arrays must be type float32. Received array "
        << "with type " << a.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  if (a.ndim() < 2) {
    std::ostringstream msg;
    msg << "[linalg::eigvalsh] Arrays must have >= 2 dimensions. Received array "
           "with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  if (a.shape(-1) != a.shape(-2)) {
    throw std::invalid_argument(
        "[linalg::eigvalsh] Eigenvalues are only defined for square matrices.");
  }

  std::vector<int> out_shape(a.shape().begin(), a.shape().end() - 1);
  out_shape.back() = a.shape(-1);

  return array(
      out_shape,
      a.dtype(),
      std::make_shared<EighPrimitive>(to_stream(s), upper, false),
      {astype(a, a.dtype(), s)});
}

std::pair<array, array> eigh(
    const array& a,
    bool upper /* = false */,
    StreamOrDevice s /* = {} */) {
  if (a.dtype() != float32) {
    std::ostringstream msg;
    msg << "[linalg::eigh] Arrays must be type float32. Received array "
        << "with type " << a.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  if (a.ndim() < 2) {
    std::ostringstream msg;
    msg << "[linalg::eigh] Arrays must have >= 2 dimensions. Received array "
           "with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  if (a.shape(-1) != a.shape(-2)) {
    throw std::invalid_argument(
        "[linalg::eigh] Eigenvectors are only defined for square matrices.");
  }

  auto out = array::make_arrays(
      {std::vector<int>(a.shape().begin(), a.shape().end() - 1), a.shape()},
      {a.dtype(), a.dtype()},
      std::make_shared<EighPrimitive>(to_stream(s), upper, true),
      {astype(a, a.dtype(), s)});
  return std::make_pair(out[0], out[1]);
}

} // namespace mlx::core::linalg
