// Copyright © 2023 Apple Inc.

#include <numeric>
#include <ostream>
#include <vector>

#include "mlx/linalg.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::linalg {

void check_cpu_stream(const StreamOrDevice& s, const std::string& prefix) {
  if (to_stream(s).device == Device::gpu) {
    throw std::invalid_argument(
        prefix +
        " This op is not yet supported on the GPU. "
        "Explicitly pass a CPU stream to run it.");
  }
}
void check_float(Dtype dtype, const std::string& prefix) {
  if (dtype != float32 && dtype != float64) {
    std::ostringstream msg;
    msg << prefix << " Arrays must have type float32 or float64. "
        << "Received array with type " << dtype << ".";
    throw std::invalid_argument(msg.str());
  }
}

void check_float_or_complex(Dtype dtype, const std::string& prefix) {
  if (dtype != float32 && dtype != float64 && dtype != complex64) {
    std::ostringstream msg;
    msg << prefix << " Arrays must have type float32, float64 or complex64. "
        << "Received array with type " << dtype << ".";
    throw std::invalid_argument(msg.str());
  }
}

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
    row_axis = (axis[0] < 0) ? axis[0] + a.ndim() : axis[0];
    col_axis = (axis[1] < 0) ? axis[1] + a.ndim() : axis[1];
    auto a_matrix = (row_axis > col_axis)
        ? moveaxis(moveaxis(a, row_axis, -1, s), col_axis, -1, s)
        : moveaxis(moveaxis(a, col_axis, -1, s), row_axis, -2, s);
    a_matrix = svd(a_matrix, false, s).at(0);
    a_matrix = (ord == 2.0) ? max(a_matrix, -1, false, s)
                            : min(a_matrix, -1, false, s);
    if (keepdims) {
      std::vector<int> sorted_axes = (row_axis < col_axis)
          ? std::vector<int>{row_axis, col_axis}
          : std::vector<int>{col_axis, row_axis};
      a_matrix = expand_dims(a_matrix, sorted_axes, s);
    }
    return astype(a_matrix, dtype, s);
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
    int row_axis = (axis[0] < 0) ? axis[0] + a.ndim() : axis[0];
    int col_axis = (axis[1] < 0) ? axis[1] + a.ndim() : axis[1];
    auto a_matrix = (row_axis > col_axis)
        ? moveaxis(moveaxis(a, row_axis, -1, s), col_axis, -1, s)
        : moveaxis(moveaxis(a, col_axis, -1, s), row_axis, -2, s);
    a_matrix = sum(svd(a_matrix, false, s).at(0), -1, false, s);
    if (keepdims) {
      std::vector<int> sorted_axes = (row_axis < col_axis)
          ? std::vector<int>{row_axis, col_axis}
          : std::vector<int>{col_axis, row_axis};
      a_matrix = expand_dims(a_matrix, sorted_axes, s);
    }
    return a_matrix;
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
  check_cpu_stream(s, "[linalg::qr]");
  check_float(a.dtype(), "[linalg::qr]");

  if (a.ndim() < 2) {
    std::ostringstream msg;
    msg << "[linalg::qr] Arrays must have >= 2 dimensions. Received array "
           "with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  int k = std::min(a.shape(-2), a.shape(-1));
  auto q_shape = a.shape();
  q_shape.back() = k;
  auto r_shape = a.shape();
  r_shape[r_shape.size() - 2] = k;
  auto out = array::make_arrays(
      {std::move(q_shape), std::move(r_shape)},
      {a.dtype(), a.dtype()},
      std::make_shared<QRF>(to_stream(s)),
      {astype(a, a.dtype(), s)});
  return std::make_pair(out[0], out[1]);
}

std::vector<array>
svd(const array& a, bool compute_uv, StreamOrDevice s /* = {} */) {
  check_cpu_stream(s, "[linalg::svd]");
  check_float_or_complex(a.dtype(), "[linalg::svd]");

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

  auto s_shape = a.shape();
  s_shape.pop_back();
  s_shape[rank - 2] = std::min(m, n);

  auto s_dtype = a.dtype() == complex64 ? float32 : a.dtype();

  if (!compute_uv) {
    return {array(
        std::move(s_shape),
        s_dtype,
        std::make_shared<SVD>(to_stream(s), compute_uv),
        {a})};
  }

  auto u_shape = a.shape();
  u_shape[rank - 2] = m;
  u_shape[rank - 1] = m;

  auto vt_shape = a.shape();
  vt_shape[rank - 2] = n;
  vt_shape[rank - 1] = n;

  return array::make_arrays(
      {u_shape, s_shape, vt_shape},
      {a.dtype(), s_dtype, a.dtype()},
      std::make_shared<SVD>(to_stream(s), compute_uv),
      {a});
}

array inv_impl(const array& a, bool tri, bool upper, StreamOrDevice s) {
  check_cpu_stream(s, "[linalg::inv]");
  check_float(a.dtype(), "[linalg::inv]");

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
    bool upper /* = false */,
    StreamOrDevice s /* = {} */) {
  return inv_impl(a, /*tri=*/true, upper, s);
}

array cholesky(
    const array& a,
    bool upper /* = false */,
    StreamOrDevice s /* = {} */) {
  check_cpu_stream(s, "[linalg::cholesky]");
  check_float(a.dtype(), "[linalg::cholesky]");
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
  check_cpu_stream(s, "[linalg::pinv]");
  check_float(a.dtype(), "[linalg::pinv]");

  if (a.ndim() < 2) {
    std::ostringstream msg;
    msg << "[linalg::pinv] Arrays must have >= 2 dimensions. Received array "
        << "with " << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  int m = a.shape(-2);
  int n = a.shape(-1);
  int k = std::min(m, n);
  auto outs = linalg::svd(a, true, s);
  array U = outs[0];
  array S = outs[1];
  array V = outs[2];

  Shape starts(a.ndim(), 0);
  auto ends = a.shape();
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

  auto rcond = 10. * std::max(m, n) * finfo(a.dtype()).eps;
  auto cutoff = multiply(array(rcond, a.dtype()), max(S, -1, true, s), s);
  auto rS =
      where(greater(S, cutoff, s), reciprocal(S, s), array(0.0f, a.dtype()), s);

  return matmul(multiply(V, rS, s), U, s);
}

array cholesky_inv(
    const array& L,
    bool upper /* = false */,
    StreamOrDevice s /* = {} */) {
  check_cpu_stream(s, "[linalg::cholesky_inv]");
  check_float(L.dtype(), "[linalg::cholesky_inv]");

  if (L.ndim() < 2) {
    std::ostringstream msg;
    msg << "[linalg::cholesky_inv] Arrays must have >= 2 dimensions. Received array "
           "with "
        << L.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  if (L.shape(-1) != L.shape(-2)) {
    throw std::invalid_argument(
        "[linalg::cholesky_inv] Cholesky inverse is only defined for square "
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

void validate_eig(
    const array& a,
    const StreamOrDevice& stream,
    const std::string& fname) {
  check_cpu_stream(stream, fname);
  check_float_or_complex(a.dtype(), fname);

  if (a.ndim() < 2) {
    std::ostringstream msg;
    msg << fname << " Arrays must have >= 2 dimensions. Received array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  if (a.shape(-1) != a.shape(-2)) {
    throw std::invalid_argument(fname + " Only defined for square matrices.");
  }
}

array eigvalsh(
    const array& a,
    std::string UPLO /* = "L" */,
    StreamOrDevice s /* = {} */) {
  validate_eig(a, s, "[linalg::eigvalsh]");
  Shape out_shape(a.shape().begin(), a.shape().end() - 1);
  Dtype eigval_type = a.dtype() == complex64 ? float32 : a.dtype();
  return array(
      std::move(out_shape),
      eigval_type,
      std::make_shared<Eigh>(to_stream(s), UPLO, false),
      {a});
}

std::pair<array, array> eigh(
    const array& a,
    std::string UPLO /* = "L" */,
    StreamOrDevice s /* = {} */) {
  validate_eig(a, s, "[linalg::eigh]");
  Dtype eigval_type = a.dtype() == complex64 ? float32 : a.dtype();
  auto out = array::make_arrays(
      {Shape(a.shape().begin(), a.shape().end() - 1), a.shape()},
      {eigval_type, a.dtype()},
      std::make_shared<Eigh>(to_stream(s), UPLO, true),
      {a});
  return std::make_pair(out[0], out[1]);
}

array eigvals(const array& a, StreamOrDevice s /* = {} */) {
  validate_eig(a, s, "[linalg::eigvals]");
  Shape out_shape(a.shape().begin(), a.shape().end() - 1);
  return array(
      std::move(out_shape),
      complex64,
      std::make_shared<Eig>(to_stream(s), false),
      {a});
}

std::pair<array, array> eig(const array& a, StreamOrDevice s /* = {} */) {
  validate_eig(a, s, "[linalg::eig]");
  auto out = array::make_arrays(
      {Shape(a.shape().begin(), a.shape().end() - 1), a.shape()},
      {complex64, complex64},
      std::make_shared<Eig>(to_stream(s), true),
      {a});
  return std::make_pair(out[0], out[1]);
}

void validate_lu(
    const array& a,
    const StreamOrDevice& stream,
    const std::string& fname) {
  check_cpu_stream(stream, fname);
  check_float(a.dtype(), fname);

  if (a.ndim() < 2) {
    std::ostringstream msg;
    msg << fname
        << " Arrays must have >= 2 dimensions. Received array "
           "with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
}

std::vector<array> lu_helper(const array& a, StreamOrDevice s /* = {} */) {
  int m = a.shape()[a.shape().size() - 2];
  int n = a.shape()[a.shape().size() - 1];

  Shape pivots_shape(a.shape().begin(), a.shape().end() - 2);
  pivots_shape.push_back(std::min(m, n));

  Shape row_idx_shape(a.shape().begin(), a.shape().end() - 1);

  return array::make_arrays(
      {a.shape(), pivots_shape, row_idx_shape},
      {a.dtype(), uint32, uint32},
      std::make_shared<LUF>(to_stream(s)),
      {astype(a, a.dtype(), s)});
}

std::vector<array> lu(const array& a, StreamOrDevice s /* = {} */) {
  validate_lu(a, s, "[linalg::lu]");

  auto out = lu_helper(a, s);
  auto& LU = out[0];
  auto& row_pivots = out[2];
  auto L = tril(LU, /* k = */ -1, s);
  auto U = triu(LU, /* k = */ 0, s);

  int M = a.shape(-2);
  int N = a.shape(-1);
  int K = std::min(M, N);
  if (N != K) {
    auto start = Shape(L.ndim(), 0);
    auto stop = L.shape();
    stop.back() = K;
    L = slice(L, std::move(start), std::move(stop), s);
  } else if (M != K) {
    auto start = Shape(U.ndim(), 0);
    auto stop = U.shape();
    stop[U.ndim() - 2] = K;
    U = slice(U, std::move(start), std::move(stop), s);
  }
  L = add(L, eye(M, K, s), s);
  return {row_pivots, L, U};
}

std::pair<array, array> lu_factor(const array& a, StreamOrDevice s /* = {} */) {
  validate_lu(a, s, "[linalg::lu_factor]");
  auto out = lu_helper(a, s);
  return std::make_pair(out[0], out[1]);
}

void validate_solve(
    const array& a,
    const array& b,
    const StreamOrDevice& stream,
    const std::string& fname) {
  check_cpu_stream(stream, fname);
  if (a.ndim() < 2) {
    std::ostringstream msg;
    msg << fname << " First input must have >= 2 dimensions. "
        << "Received array with " << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  if (b.ndim() < 1) {
    std::ostringstream msg;
    msg << fname << " Second input must have >= 1 dimensions. "
        << "Received array with " << b.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  if (a.shape(-1) != a.shape(-2)) {
    std::ostringstream msg;
    msg << fname << " First input must be a square matrix. "
        << "Received array with shape " << a.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  int lastDim = b.ndim() > 1 ? -2 : -1;
  if (a.shape(-1) != b.shape(lastDim)) {
    std::ostringstream msg;
    msg << fname << " Last dimension of first input with shape " << a.shape()
        << " must match second to last dimension of"
        << " second input with shape " << b.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto out_type = promote_types(a.dtype(), b.dtype());
  if (out_type != float32 && out_type != float64) {
    std::ostringstream msg;
    msg << fname
        << " Input arrays must promote to float32 or float64. "
           " Received arrays with type "
        << a.dtype() << " and " << b.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
}

array solve(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  validate_solve(a, b, s, "[linalg::solve]");

  // P, L, U matrices
  const auto luf = lu(a, s);
  auto perm = argsort(luf[0], -1, s);
  int take_axis = -1;
  if (b.ndim() >= 2) {
    perm = expand_dims(perm, -1, s);
    take_axis -= 1;
  }
  auto pb = take_along_axis(b, perm, take_axis, s);
  auto y = solve_triangular(luf[1], pb, /* upper = */ false, s);
  return solve_triangular(luf[2], y, /* upper = */ true, s);
}

array solve_triangular(
    const array& a,
    const array& b,
    bool upper /* = false */,
    StreamOrDevice s /* = {} */) {
  validate_solve(a, b, s, "[linalg::solve_triangular]");
  auto a_inv = tri_inv(a, upper, s);
  return matmul(a_inv, b, s);
}

void validate_det(
    const array& a,
    const StreamOrDevice& stream,
    const std::string& fname) {
  check_cpu_stream(stream, fname);
  if (issubdtype(a.dtype(), complexfloating)) {
    throw std::invalid_argument(fname + " Complex inputs are not supported.");
  }
  if (a.ndim() < 2) {
    std::ostringstream msg;
    msg << fname
        << " Arrays must have >= 2 dimensions. Received array "
           "with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  if (a.shape(-1) != a.shape(-2)) {
    throw std::invalid_argument(fname + " Only defined for square matrices.");
  }
}

array det_raw_small(const array& a, StreamOrDevice s) {
  int n = a.shape(-1);

  // Empty 0x0 matrix: determinant is the empty product = 1
  if (n == 0) {
    Shape out_shape(a.shape().begin(), a.shape().end() - 2);
    return broadcast_to(array(1.0f, a.dtype()), std::move(out_shape), s);
  }

  // Helper to extract a[..., i, j] from the last two dims
  auto elem = [&](int i, int j) {
    auto starts = Shape(a.ndim(), 0);
    auto stops = a.shape();
    starts[a.ndim() - 2] = i;
    stops[a.ndim() - 2] = i + 1;
    starts[a.ndim() - 1] = j;
    stops[a.ndim() - 1] = j + 1;
    return squeeze(squeeze(slice(a, starts, stops, s), -1, s), -1, s);
  };

  if (n == 1) {
    return elem(0, 0);
  } else if (n == 2) {
    return subtract(
        multiply(elem(0, 0), elem(1, 1), s),
        multiply(elem(0, 1), elem(1, 0), s),
        s);
  } else {
    // 3x3: a00*(a11*a22 - a12*a21) - a01*(a10*a22 - a12*a20) + a02*(a10*a21 -
    // a11*a20)
    auto a00 = elem(0, 0), a01 = elem(0, 1), a02 = elem(0, 2);
    auto a10 = elem(1, 0), a11 = elem(1, 1), a12 = elem(1, 2);
    auto a20 = elem(2, 0), a21 = elem(2, 1), a22 = elem(2, 2);
    return add(
        subtract(
            multiply(
                a00,
                subtract(multiply(a11, a22, s), multiply(a12, a21, s), s),
                s),
            multiply(
                a01,
                subtract(multiply(a10, a22, s), multiply(a12, a20, s), s),
                s),
            s),
        multiply(
            a02, subtract(multiply(a10, a21, s), multiply(a11, a20, s), s), s),
        s);
  }
}

std::pair<array, array> slogdet_impl(const array& input, StreamOrDevice s) {
  int n = input.shape(-1);
  auto dtype = input.dtype();

  // Small-matrix fast path
  if (n <= 3) {
    auto raw = det_raw_small(input, s);
    auto abs_raw = abs(raw, s);
    auto sgn = sign(raw, s);
    auto logabs = log(abs_raw, s);
    return std::make_pair(sgn, logabs);
  }

  // General LU-based path
  auto [LU, pivots] = lu_factor(input, s);

  // Extract diagonal of U
  auto diag = diagonal(LU, 0, -2, -1, s);

  // Permutation parity: count positions where pivot[i] != i
  int k = std::min(input.shape(-2), input.shape(-1));
  auto iota = arange(0, k, uint32, s);
  auto parity = astype(
      sum(not_equal(pivots, iota, s),
          /* axis = */ -1,
          /* keepdims = */ false,
          s),
      int32,
      s);

  // Count negative diagonal elements
  auto num_neg = astype(
      sum(less(diag, array(0.0f, dtype), s),
          /* axis = */ -1,
          /* keepdims = */ false,
          s),
      int32,
      s);

  // sign = (-1)^(parity + num_neg)
  auto total = add(parity, num_neg, s);
  auto sign_val = astype(
      subtract(
          array(1, int32),
          multiply(array(2, int32), remainder(total, array(2, int32), s), s),
          s),
      dtype,
      s);

  // logabsdet = sum(log(abs(diag)))
  auto logabsdet =
      sum(log(abs(diag, s), s), /* axis = */ -1, /* keepdims = */ false, s);

  // Handle singular matrices: any zero on diagonal
  auto is_zero =
      any(equal(diag, array(0.0f, dtype), s),
          /* axis = */ -1,
          /* keepdims = */ false,
          s);
  sign_val = where(is_zero, array(0.0f, dtype), sign_val, s);
  logabsdet = where(
      is_zero,
      array(-std::numeric_limits<float>::infinity(), dtype),
      logabsdet,
      s);

  return std::make_pair(sign_val, logabsdet);
}

std::pair<array, array> slogdet(const array& a, StreamOrDevice s /* = {} */) {
  validate_det(a, s, "[linalg::slogdet]");

  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return slogdet_impl(input, s);
}

array det(const array& a, StreamOrDevice s /* = {} */) {
  validate_det(a, s, "[linalg::det]");

  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  int n = input.shape(-1);

  // Small-matrix fast path: compute directly, skip log/exp round-trip
  if (n <= 3) {
    return det_raw_small(input, s);
  }

  // General case: det = sign * exp(logabsdet)
  auto [sign_val, logabsdet] = slogdet_impl(input, s);
  return multiply(sign_val, exp(logabsdet, s), s);
}

} // namespace mlx::core::linalg
