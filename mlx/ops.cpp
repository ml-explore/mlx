// Copyright © 2023-2024 Apple Inc.

// Required for using M_PI in MSVC.
#define _USE_MATH_DEFINES
#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <set>
#include <sstream>

#include "mlx/backend/cuda/cuda.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

std::tuple<Shape, std::vector<int>, bool> compute_reduce_shape(
    const std::vector<int>& axes,
    const Shape& shape) {
  bool is_noop = true;
  std::set<int> axes_set;
  auto ndim = shape.size();
  for (auto ax : axes) {
    int ax_ = (ax < 0) ? ax + ndim : ax;
    if (ax_ < 0 || ax_ >= ndim) {
      std::ostringstream msg;
      msg << "Invalid axis " << ax << " for array with " << ndim
          << " dimensions.";
      throw std::out_of_range(msg.str());
    }
    axes_set.insert(ax_);
  }
  if (axes_set.size() != axes.size()) {
    throw std::invalid_argument("Duplicate axes detected in reduction.");
  }
  Shape out_shape;
  for (int i = 0; i < ndim; ++i) {
    if (axes_set.count(i) == 0) {
      out_shape.push_back(shape[i]);
    } else {
      out_shape.push_back(1);
    }
    is_noop &= (out_shape.back() == shape[i]);
  }
  std::vector<int> sorted_axes(axes_set.begin(), axes_set.end());
  return {out_shape, sorted_axes, is_noop};
}

Dtype at_least_float(const Dtype& d) {
  return issubdtype(d, inexact) ? d : promote_types(d, float32);
}

array indices_or_default(
    std::optional<array> indices,
    const array& x,
    StreamOrDevice s) {
  if (indices.has_value()) {
    return indices.value();
  }

  Shape shape(x.shape().begin(), x.shape().end() - 2);
  int total =
      std::reduce(shape.begin(), shape.end(), 1, std::multiplies<int>());
  return reshape(arange(total, uint32, s), std::move(shape), s);
}

void validate_quantized_input(
    std::string_view tag,
    const array& w,
    const array& scales,
    int group_size,
    int bits,
    const std::optional<array>& biases = std::nullopt) {
  if (w.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[" << tag << "] The weight matrix should be uint32 "
        << "but received " << w.dtype();
    throw std::invalid_argument(msg.str());
  }

  if (biases && scales.shape() != biases->shape()) {
    std::ostringstream msg;
    msg << "[" << tag << "] Scales and biases should have the same shape. "
        << "Received scales with shape " << scales.shape()
        << " and biases with " << biases->shape();
    throw std::invalid_argument(msg.str());
  }

  if (!std::equal(
          w.shape().begin(), w.shape().end() - 2, scales.shape().begin())) {
    std::ostringstream msg;
    msg << "[" << tag
        << "] Weight and scales should have the same batch shape. "
        << "Received weight with shape " << w.shape() << ", scales with "
        << scales.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  if (w.shape(-1) * 32 / bits != scales.shape(-1) * group_size) {
    std::ostringstream msg;
    msg << "[" << tag << "] The shapes of the weight and scales are "
        << "incompatible based on bits and group_size. w.shape() == "
        << w.shape() << " and scales.shape() == " << scales.shape()
        << " with group_size=" << group_size << " and bits=" << bits;
    throw std::invalid_argument(msg.str());
  }
}

std::pair<int, int> extract_quantized_matmul_dims(
    std::string_view tag,
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    bool transpose,
    int group_size,
    int bits) {
  validate_quantized_input(tag, w, scales, group_size, bits, biases);

  int x_inner_dims = x.shape(-1);

  // Calculate the expanded w's dims
  int w_inner_dims = (transpose) ? w.shape(-1) * 32 / bits : w.shape(-2);
  int w_outer_dims = (transpose) ? w.shape(-2) : w.shape(-1) * 32 / bits;

  if (w_inner_dims != x_inner_dims) {
    std::ostringstream msg;
    msg << "[" << tag << "] Last dimension of first input with "
        << "shape (..., " << x_inner_dims << ") does not match "
        << "the expanded quantized matrix (" << w_inner_dims << ", "
        << w_outer_dims << ") computed from shape " << w.shape()
        << " with group_size=" << group_size << ", bits=" << bits
        << " and transpose=" << std::boolalpha << transpose;
    throw std::invalid_argument(msg.str());
  }

  return {w_inner_dims, w_outer_dims};
}

} // namespace

array arange(
    double start,
    double stop,
    double step,
    Dtype dtype,
    StreamOrDevice s /* = {} */) {
  if (dtype == bool_) {
    std::ostringstream msg;
    msg << bool_ << " not supported for arange.";
    throw std::invalid_argument(msg.str());
  }
  if (std::isnan(start) || std::isnan(step) || std::isnan(stop)) {
    throw std::invalid_argument("[arange] Cannot compute length.");
  }

  if (std::isinf(start) || std::isinf(stop)) {
    throw std::invalid_argument("[arange] Cannot compute length.");
  }

  // Check if start and stop specify a valid range because if not, we have to
  // return an empty array
  if (std::isinf(step) &&
      ((step > 0 && start < stop) || (step < 0 && start > stop))) {
    return array({start}, dtype);
  }

  double real_size = std::ceil((stop - start) / step);

  if (real_size > INT_MAX) {
    throw std::invalid_argument("[arange] Maximum size exceeded.");
  }

  int size = std::max(static_cast<int>(real_size), 0);
  return array(
      {size},
      dtype,
      std::make_shared<Arange>(to_stream(s), start, stop, step),
      {});
}
array arange(
    double start,
    double stop,
    double step,
    StreamOrDevice s /* = {} */) {
  return arange(start, stop, step, float32, to_stream(s));
}
array arange(
    double start,
    double stop,
    Dtype dtype,
    StreamOrDevice s /* = {} */) {
  return arange(start, stop, 1.0, dtype, to_stream(s));
}
array arange(double start, double stop, StreamOrDevice s /* = {} */) {
  return arange(start, stop, 1.0, float32, to_stream(s));
}
array arange(double stop, Dtype dtype, StreamOrDevice s /* = {} */) {
  return arange(0.0, stop, 1.0, dtype, to_stream(s));
}
array arange(double stop, StreamOrDevice s /* = {} */) {
  return arange(0.0, stop, 1.0, float32, to_stream(s));
}
array arange(int start, int stop, int step, StreamOrDevice s /* = {} */) {
  return arange(
      static_cast<double>(start),
      static_cast<double>(stop),
      static_cast<double>(step),
      int32,
      to_stream(s));
}
array arange(int start, int stop, StreamOrDevice s /* = {} */) {
  return arange(
      static_cast<double>(start),
      static_cast<double>(stop),
      1.0,
      int32,
      to_stream(s));
}
array arange(int stop, StreamOrDevice s /* = {} */) {
  return arange(0.0, static_cast<double>(stop), 1.0, int32, to_stream(s));
}

array linspace(
    double start,
    double stop,
    int num /* = 50 */,
    Dtype dtype /* = float32 */,
    StreamOrDevice s /* = {} */) {
  if (num < 0) {
    std::ostringstream msg;
    msg << "[linspace] number of samples, " << num << ", must be non-negative.";
    throw std::invalid_argument(msg.str());
  }
  if (num == 1) {
    return astype(array({start}), dtype, s);
  }
  auto inner_type = dtype == float64 ? float64 : float32;
  array t =
      divide(arange(0, num, inner_type, s), array(num - 1, inner_type), s);
  array t_bar = subtract(array(1, inner_type), t, s);
  return astype(
      add(multiply(t_bar, array(start, inner_type), s),
          multiply(t, array(stop, inner_type), s),
          s),
      dtype,
      s);
}

array astype(array a, Dtype dtype, StreamOrDevice s /* = {} */) {
  if (dtype == a.dtype()) {
    return a;
  }
  auto copied_shape = a.shape(); // |a| will be moved
  return array(
      std::move(copied_shape),
      dtype,
      std::make_shared<AsType>(to_stream(s), dtype),
      {std::move(a)});
}

array as_strided(
    array a,
    Shape shape,
    Strides strides,
    size_t offset,
    StreamOrDevice s /* = {} */) {
  auto copied_shape = shape; // |shape| will be moved
  auto dtype = a.dtype(); // |a| will be moved
  return array(
      std::move(copied_shape),
      dtype,
      std::make_shared<AsStrided>(
          to_stream(s), std::move(shape), std::move(strides), offset),
      // Force the input array to be contiguous.
      {flatten(std::move(a), s)});
}

array copy(array a, StreamOrDevice s /* = {} */) {
  auto copied_shape = a.shape(); // |a| will be moved
  auto dtype = a.dtype();
  return array(
      std::move(copied_shape),
      dtype,
      std::make_shared<Copy>(to_stream(s)),
      {std::move(a)});
}

array full_impl(array vals, Dtype dtype, StreamOrDevice s /* = {} */) {
  return array(
      vals.shape(),
      dtype,
      std::make_shared<Full>(to_stream(s)),
      {astype(vals, dtype, s)});
}

array full(Shape shape, array vals, Dtype dtype, StreamOrDevice s /* = {} */) {
  if (std::any_of(shape.begin(), shape.end(), [](auto i) { return i < 0; })) {
    throw std::invalid_argument("[full] Negative dimensions not allowed.");
  }
  return full_impl(broadcast_to(vals, std::move(shape), s), dtype, s);
}

array full(Shape shape, array vals, StreamOrDevice s /* = {} */) {
  auto dtype = vals.dtype(); // |vals| will be moved
  return full(std::move(shape), std::move(vals), dtype, to_stream(s));
}

array full_like(
    const array& a,
    array vals,
    Dtype dtype,
    StreamOrDevice s /* = {} */) {
  auto inputs = broadcast_arrays({a, std::move(vals)}, s);
  return full_impl(std::move(inputs[1]), dtype, s);
}

array full_like(const array& a, array vals, StreamOrDevice s /* = {} */) {
  return full_like(a, std::move(vals), a.dtype(), to_stream(s));
}

array zeros(const Shape& shape, Dtype dtype, StreamOrDevice s /* = {} */) {
  return full(shape, array(0, dtype), to_stream(s));
}

array zeros_like(const array& a, StreamOrDevice s /* = {} */) {
  return full_like(a, 0, a.dtype(), to_stream(s));
}

array ones(const Shape& shape, Dtype dtype, StreamOrDevice s /* = {} */) {
  return full(shape, array(1, dtype), to_stream(s));
}

array ones_like(const array& a, StreamOrDevice s /* = {} */) {
  return full_like(a, 1, a.dtype(), to_stream(s));
}

array eye(int n, int m, int k, Dtype dtype, StreamOrDevice s /* = {} */) {
  if (n <= 0 || m <= 0) {
    throw std::invalid_argument("[eye] N and M must be positive integers.");
  }
  array result = zeros({n, m}, dtype, s);
  if (k >= m || -k >= n) {
    return result;
  }

  int diagonal_length = k >= 0 ? std::min(n, m - k) : std::min(n + k, m);

  std::vector<array> indices;
  auto s1 = std::max(0, -k);
  auto s2 = std::max(0, k);
  indices.push_back(arange(s1, diagonal_length + s1, int32, s));
  indices.push_back(arange(s2, diagonal_length + s2, int32, s));
  array ones_array = ones({diagonal_length, 1, 1}, dtype, s);
  return scatter(result, indices, ones_array, {0, 1}, s);
}

array identity(int n, Dtype dtype, StreamOrDevice s /* = {} */) {
  return eye(n, n, 0, dtype, s);
}

array tri(int n, int m, int k, Dtype type, StreamOrDevice s /* = {} */) {
  auto l = expand_dims(arange(n, s), 1, s);
  auto r = expand_dims(arange(-k, m - k, s), 0, s);
  return astype(greater_equal(l, r, s), type, s);
}

array tril(array x, int k /* = 0 */, StreamOrDevice s /* = {} */) {
  if (x.ndim() < 2) {
    throw std::invalid_argument("[tril] array must be at least 2-D");
  }
  auto mask = tri(x.shape(-2), x.shape(-1), k, bool_, s);
  return where(mask, x, array(0, x.dtype()), s);
}

array triu(array x, int k /* = 0 */, StreamOrDevice s /* = {} */) {
  if (x.ndim() < 2) {
    throw std::invalid_argument("[triu] array must be at least 2-D");
  }
  auto mask = tri(x.shape(-2), x.shape(-1), k - 1, bool_, s);
  return where(mask, array(0, x.dtype()), x, s);
}

array reshape(const array& a, Shape shape, StreamOrDevice s /* = {} */) {
  if (a.shape() == shape) {
    return a;
  }
  auto out_shape = Reshape::output_shape(a, shape);
  return array(
      std::move(out_shape),
      a.dtype(),
      std::make_shared<Reshape>(to_stream(s), std::move(shape)),
      {a});
}

array unflatten(
    const array& a,
    int axis,
    Shape shape,
    StreamOrDevice s /* = {} */) {
  if (shape.empty()) {
    throw std::invalid_argument(
        "[unflatten] Shape to unflatten to cannot be empty.");
  }
  auto ndim = static_cast<int>(a.ndim());
  auto ax = axis < 0 ? axis + ndim : axis;
  if (ax < 0 || ax >= ndim) {
    std::ostringstream msg;
    msg << "[unflatten] Invalid axes " << ax << " for array with " << a.ndim()
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  size_t size = 1;
  int infer_idx = -1;
  for (int i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1) {
      if (infer_idx >= 0) {
        throw std::invalid_argument(
            "[Unflatten] Can only infer one dimension.");
      }
      infer_idx = i;
    } else {
      size *= shape[i];
    }
  }
  if (infer_idx >= 0) {
    shape[infer_idx] = a.shape(ax) / size;
    size *= shape[infer_idx];
  }
  if (size != a.shape(ax)) {
    std::ostringstream msg;
    msg << "[Unflatten] Cannot unflatten axis " << axis << " with size "
        << a.shape(ax) << " into shape " << shape << ".";
    throw std::invalid_argument(msg.str());
  }
  if (shape.size() == 1) {
    return a;
  }

  auto out_shape = Unflatten::output_shape(a, ax, shape);
  return array(
      std::move(out_shape),
      a.dtype(),
      std::make_shared<Unflatten>(to_stream(s), ax, std::move(shape)),
      {a});
}

array flatten(
    const array& a,
    int start_axis,
    int end_axis /* = -1 */,
    StreamOrDevice s /* = {} */) {
  auto ndim = static_cast<int>(a.ndim());
  auto start_ax = start_axis + (start_axis < 0 ? ndim : 0);
  auto end_ax = end_axis + (end_axis < 0 ? ndim : 0);
  start_ax = std::max(0, start_ax);
  end_ax = std::min(ndim - 1, end_ax);
  if (a.ndim() == 0) {
    return reshape(a, {1}, s);
  }
  if (end_ax < start_ax) {
    throw std::invalid_argument(
        "[flatten] start_axis must be less than or equal to end_axis");
  }
  if (start_ax >= ndim) {
    std::ostringstream msg;
    msg << "[flatten] Invalid start_axis " << start_axis << " for array with "
        << ndim << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (end_ax < 0) {
    std::ostringstream msg;
    msg << "[flatten] Invalid end_axis " << end_axis << " for array with "
        << ndim << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (start_ax == end_ax) {
    return a;
  }
  return array(
      Flatten::output_shape(a, start_ax, end_ax),
      a.dtype(),
      std::make_shared<Flatten>(to_stream(s), start_ax, end_ax),
      {a});
}

array flatten(const array& a, StreamOrDevice s /* = {} */) {
  return flatten(a, 0, a.ndim() - 1, s);
}

array hadamard_transform(
    const array& a,
    std::optional<float> scale_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  if (a.size() == 0) {
    throw std::invalid_argument(
        "[hadamard_transform] Does not support empty arrays.");
  }
  // Default to an orthonormal Hadamard matrix scaled by 1/sqrt(N)
  int n = a.ndim() > 0 ? a.shape(-1) : 1;
  float scale = scale_.has_value() ? *scale_ : 1.0f / std::sqrt(n);
  auto dtype = issubdtype(a.dtype(), floating) ? a.dtype() : float32;

  // Nothing to do for a scalar
  if (n == 1) {
    if (scale == 1) {
      return a;
    }

    return multiply(a, array(scale, dtype), s);
  }

  return array(
      a.shape(),
      dtype,
      std::make_shared<Hadamard>(to_stream(s), scale),
      {astype(a, dtype, s)});
}

array squeeze_impl(
    const array& a,
    std::vector<int> axes,
    StreamOrDevice s /* = {} */) {
  for (auto& ax : axes) {
    auto new_ax = ax < 0 ? ax + a.ndim() : ax;
    if (new_ax < 0 || new_ax >= a.ndim()) {
      std::ostringstream msg;
      msg << "[squeeze] Invalid axes " << ax << " for array with " << a.ndim()
          << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if (a.shape(new_ax) != 1) {
      std::ostringstream msg;
      msg << "[squeeze] Cannot squeeze axis " << ax << " with size "
          << a.shape(ax) << " which is not equal to 1.";
      throw std::invalid_argument(msg.str());
    }
    ax = new_ax;
  }
  auto shape = Squeeze::output_shape(a, axes);
  return array(
      std::move(shape),
      a.dtype(),
      std::make_shared<Squeeze>(to_stream(s), std::move(axes)),
      {a});
}

array squeeze(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  if (axes.empty()) {
    return a;
  }
  std::set<int> unique_axes;
  for (auto ax : axes) {
    unique_axes.insert(ax < 0 ? ax + a.ndim() : ax);
  }
  if (unique_axes.size() != axes.size()) {
    throw std::invalid_argument("[squeeze] Received duplicate axes.");
  }
  std::vector<int> sorted_axes(unique_axes.begin(), unique_axes.end());
  return squeeze_impl(a, std::move(sorted_axes), s);
}

array squeeze(const array& a, int axis, StreamOrDevice s /* = {} */) {
  return squeeze_impl(a, {axis}, s);
}

array squeeze(const array& a, StreamOrDevice s /* = {} */) {
  std::vector<int> axes;
  for (int i = 0; i < a.ndim(); ++i) {
    if (a.shape(i) == 1) {
      axes.push_back(i);
    }
  }
  return squeeze_impl(a, std::move(axes), s);
}

array expand_dims_impl(
    const array& a,
    std::vector<int> axes,
    StreamOrDevice s /* = {} */) {
  auto out_ndim = a.ndim() + axes.size();
  for (auto& ax : axes) {
    auto new_ax = ax < 0 ? ax + out_ndim : ax;
    if (new_ax < 0 || new_ax >= out_ndim) {
      std::ostringstream msg;
      msg << "[expand_dims] Invalid axis " << ax << " for output array with "
          << a.ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    ax = new_ax;
  }
  auto shape = ExpandDims::output_shape(a, axes);
  return array(
      std::move(shape),
      a.dtype(),
      std::make_shared<ExpandDims>(to_stream(s), std::move(axes)),
      {a});
}

array expand_dims(const array& a, int axis, StreamOrDevice s /* = {} */) {
  return expand_dims_impl(a, {axis}, s);
}

array expand_dims(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  if (axes.empty()) {
    return a;
  }
  { // Check for repeats
    std::set<int> unique_axes(axes.begin(), axes.end());
    if (unique_axes.size() != axes.size()) {
      throw std::invalid_argument("[expand_dims] Received duplicate axes.");
    }
  }
  // Check for repeats again
  auto out_ndim = a.ndim() + axes.size();
  std::set<int> unique_axes;
  for (auto ax : axes) {
    unique_axes.insert(ax < 0 ? ax + out_ndim : ax);
  }
  if (unique_axes.size() != axes.size()) {
    throw std::invalid_argument("[expand_dims] Received duplicate axes.");
  }
  std::vector<int> sorted_axes(unique_axes.begin(), unique_axes.end());
  return expand_dims_impl(a, std::move(sorted_axes), s);
}

// Slice helper
namespace {

inline auto
normalize_slice(const Shape& shape, Shape& start, Shape stop, Shape& strides) {
  // - Start indices are normalized
  // - End indices are unchanged as -1 means something different
  //   pre-normalization (the end of the axis) versus post normalization (the
  //   position left of 0).
  // - Any strides corresponding to singleton dimension are set to 1

  Shape out_shape(shape.size());
  bool has_neg_strides = false;

  for (int i = 0; i < shape.size(); ++i) {
    // Following numpy docs
    //  Negative i and j are interpreted as n + i and n + j where n is
    //  the number of elements in the corresponding dimension. Negative
    //  k makes stepping go towards smaller indices

    auto n = shape[i];
    auto s = start[i];
    s = s < 0 ? s + n : s;
    auto e = stop[i];
    e = e < 0 ? e + n : e;

    // Note: -ve strides require start >= stop
    if (strides[i] < 0) {
      has_neg_strides = true;

      // Clamp to bounds
      auto st = std::min(s, n - 1);
      auto ed = e > -1 ? e : -1;

      start[i] = st;
      ed = ed > st ? st : ed;

      auto str = -strides[i];
      out_shape[i] = (start[i] - ed + str - 1) / str;

    } else {
      // Clamp to bounds
      auto st = std::max(static_cast<ShapeElem>(0), std::min(s, n));
      auto ed = std::max(static_cast<ShapeElem>(0), std::min(e, n));

      start[i] = st;
      ed = ed < st ? st : ed;

      out_shape[i] = (ed - start[i] + strides[i] - 1) / strides[i];
    }
    // Simplify the stride if it's unused
    if (out_shape[i] == 1) {
      strides[i] = 1;
    }
  }

  return std::make_pair(has_neg_strides, out_shape);
}

void normalize_dynamic_slice_inputs(
    const array& a,
    const array& start,
    std::vector<int>& axes,
    std::string_view prefix) {
  if (start.size() > a.ndim()) {
    std::ostringstream msg;
    msg << prefix << " Invalid number of starting positions for "
        << "array with dimension " << a.ndim() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (start.ndim() > 1) {
    std::ostringstream msg;
    msg << prefix << " array of starting indices "
        << "must be zero or one dimensional but has dimension " << start.ndim()
        << ".";
    throw std::invalid_argument(msg.str());
  }
  if (start.size() != axes.size()) {
    std::ostringstream msg;
    msg << prefix << " Number of starting indices " << start.size()
        << " does not match number of axes " << axes.size() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(start.dtype(), integer)) {
    std::ostringstream msg;
    msg << prefix << " Start indices must be integers, got type "
        << start.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  for (auto& ax : axes) {
    auto new_ax = ax < 0 ? ax + a.ndim() : ax;
    if (new_ax < 0 || new_ax >= a.ndim()) {
      std::ostringstream msg;
      msg << prefix << " Invalid axis " << ax << " for array with dimension "
          << a.ndim() << ".";
      throw std::invalid_argument(msg.str());
    }
    ax = new_ax;
  }
  std::set dims(axes.begin(), axes.end());
  if (dims.size() != axes.size()) {
    std::ostringstream msg;
    msg << prefix << " Repeat axes not allowed.";
    throw std::invalid_argument(msg.str());
  }
}

} // namespace

array slice(
    const array& a,
    Shape start,
    Shape stop,
    Shape strides,
    StreamOrDevice s /* = {} */) {
  if (start.size() != a.ndim() || stop.size() != a.ndim() ||
      strides.size() != a.ndim()) {
    std::ostringstream msg;
    msg << "[slice] Invalid number of indices or strides for "
        << "array with dimension " << a.ndim() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto [has_neg_strides, out_shape] =
      normalize_slice(a.shape(), start, stop, strides);

  if (!has_neg_strides && out_shape == a.shape()) {
    return a;
  }

  return array(
      out_shape,
      a.dtype(),
      std::make_shared<Slice>(
          to_stream(s), std::move(start), std::move(stop), std::move(strides)),
      {a});
}

array slice(
    const array& a,
    Shape start,
    Shape stop,
    StreamOrDevice s /* = {} */) {
  return slice(
      a, std::move(start), std::move(stop), Shape(a.ndim(), 1), to_stream(s));
}

array slice(
    const array& a,
    const array& start,
    std::vector<int> axes,
    Shape slice_size,
    StreamOrDevice s /* = {} */) {
  normalize_dynamic_slice_inputs(a, start, axes, "[slice]");

  // Check the slice_size
  if (slice_size.size() != a.ndim()) {
    std::ostringstream msg;
    msg << "[slice] Invalid slice size for array with " << a.ndim()
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  for (int i = 0; i < a.ndim(); ++i) {
    if (slice_size[i] > a.shape(i)) {
      std::ostringstream msg;
      msg << "[slice] Invalid slice size " << slice_size
          << " for array with shape " << a.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
  }
  auto out_shape = slice_size;
  return array(
      std::move(out_shape),
      a.dtype(),
      std::make_shared<DynamicSlice>(
          to_stream(s), std::move(axes), std::move(slice_size)),
      {a, start});
}

/** Update a slice from the source array */
array slice_update(
    const array& src,
    const array& update,
    Shape start,
    Shape stop,
    Shape strides,
    StreamOrDevice s /* = {} */) {
  // Check dimensions
  if (start.size() != src.ndim() || stop.size() != src.ndim() ||
      strides.size() != src.ndim()) {
    std::ostringstream msg;
    msg << "[slice_update] Invalid number of indices or strides for "
        << "array with dimension " << src.ndim() << ".";
    throw std::invalid_argument(msg.str());
  }

  // Process slice dimensions
  auto [has_neg_strides, upd_shape] =
      normalize_slice(src.shape(), start, stop, strides);

  // Cast update to src type and broadcast update shape to slice shape
  auto upd = broadcast_to(astype(update, src.dtype(), s), upd_shape, s);

  // If the entire src is the slice, just return the update
  if (!has_neg_strides && upd_shape == src.shape()) {
    return upd;
  }
  return array(
      src.shape(),
      src.dtype(),
      std::make_shared<SliceUpdate>(
          to_stream(s), std::move(start), std::move(stop), std::move(strides)),
      {src, upd});
}

/** Update a slice from the source array with stride 1 in each dimension */
array slice_update(
    const array& src,
    const array& update,
    Shape start,
    Shape stop,
    StreamOrDevice s /* = {} */) {
  return slice_update(
      src, update, std::move(start), std::move(stop), Shape(src.ndim(), 1), s);
}

/** Update a slice from the source array */
array slice_update(
    const array& src,
    const array& update,
    const array& start,
    std::vector<int> axes,
    StreamOrDevice s /* = {} */) {
  normalize_dynamic_slice_inputs(src, start, axes, "[slice_update]");

  // Broadcast update with unspecified axes
  auto up_shape = update.shape();
  auto dim_diff = std::max(src.ndim() - update.ndim(), size_t(0));
  up_shape.insert(
      up_shape.begin(), src.shape().begin(), src.shape().begin() + dim_diff);
  for (int d = dim_diff; d < src.ndim(); ++d) {
    up_shape[d] = std::min(up_shape[d], src.shape(d));
  }
  for (auto ax : axes) {
    if (ax < dim_diff) {
      up_shape[ax] = 1;
    }
  }
  auto upd = broadcast_to(astype(update, src.dtype(), s), up_shape, s);
  return array(
      src.shape(),
      src.dtype(),
      std::make_shared<DynamicSliceUpdate>(to_stream(s), std::move(axes)),
      {src, upd, start});
}

std::vector<array> split(
    const array& a,
    const Shape& indices,
    int axis,
    StreamOrDevice s /* = {} */) {
  auto ax = axis < 0 ? axis + a.ndim() : axis;
  if (ax < 0 || ax >= a.ndim()) {
    std::ostringstream msg;
    msg << "Invalid axis (" << axis << ") passed to split"
        << " for array with shape " << a.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  if (indices.empty()) {
    return {a};
  }

  if (indices.size() < 10 &&
      std::is_sorted(indices.begin(), indices.end(), std::less<>{}) &&
      indices[0] > 0 && indices.back() < a.shape(ax)) {
    std::vector<Dtype> dtypes(indices.size() + 1, a.dtype());
    std::vector<Shape> shapes(indices.size() + 1, a.shape());
    shapes[0][ax] = indices[0];
    for (int i = 1; i < indices.size(); i++) {
      shapes[i][ax] = indices[i] - indices[i - 1];
    }
    shapes.back()[ax] = a.shape(ax) - indices.back();

    return array::make_arrays(
        std::move(shapes),
        dtypes,
        std::make_shared<Split>(to_stream(s), indices, ax),
        {a});
  }

  std::vector<array> res;
  auto start_indices = Shape(a.ndim(), 0);
  auto stop_indices = a.shape();
  for (int i = 0; i < indices.size() + 1; ++i) {
    stop_indices[ax] = i < indices.size() ? indices[i] : a.shape(ax);
    res.push_back(slice(a, start_indices, stop_indices, to_stream(s)));
    start_indices[ax] = stop_indices[ax];
  }
  return res;
}

std::vector<array>
split(const array& a, const Shape& indices, StreamOrDevice s /* = {} */) {
  return split(a, indices, 0, s);
}

std::vector<array>
split(const array& a, int num_splits, int axis, StreamOrDevice s /* = {} */) {
  auto ax = axis < 0 ? axis + a.ndim() : axis;
  if (ax < 0 || ax >= a.ndim()) {
    std::ostringstream msg;
    msg << "Invalid axis " << axis << " passed to split"
        << " for array with shape " << a.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  auto q_and_r = std::ldiv(a.shape(axis), num_splits);
  if (q_and_r.rem) {
    std::ostringstream msg;
    msg << "Array split does not result in sub arrays with equal size:"
        << " attempting " << num_splits << " splits along axis " << axis
        << " for shape " << a.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  auto split_size = q_and_r.quot;
  Shape indices(num_splits - 1);
  for (int i = 0; i < indices.size(); ++i) {
    indices[i] = (i + 1) * split_size;
  }
  return split(a, indices, axis, s);
}

std::vector<array>
split(const array& a, int num_splits, StreamOrDevice s /* = {} */) {
  return split(a, num_splits, 0, to_stream(s));
}

std::vector<array> meshgrid(
    const std::vector<array>& arrays,
    bool sparse /* = false */,
    const std::string& indexing /* = "xy" */,
    StreamOrDevice s /* = {} */) {
  if (indexing != "xy" && indexing != "ij") {
    throw std::invalid_argument(
        "[meshgrid] Invalid indexing value. Valid values are 'xy' and 'ij'.");
  }

  auto ndim = arrays.size();
  std::vector<array> outputs;
  for (int i = 0; i < ndim; ++i) {
    Shape shape(ndim, 1);
    shape[i] = -1;
    outputs.push_back(reshape(arrays[i], std::move(shape), s));
  }

  if (indexing == "xy" && ndim > 1) {
    Shape shape(ndim, 1);

    shape[1] = arrays[0].size();
    outputs[0] = reshape(arrays[0], shape, s);
    shape[1] = 1;
    shape[0] = arrays[1].size();
    outputs[1] = reshape(arrays[1], std::move(shape), s);
  }

  if (!sparse) {
    outputs = broadcast_arrays(outputs, s);
  }

  return outputs;
}

array clip(
    const array& a,
    const std::optional<array>& a_min,
    const std::optional<array>& a_max,
    StreamOrDevice s /* = {} */) {
  if (!a_min.has_value() && !a_max.has_value()) {
    throw std::invalid_argument("At most one of a_min and a_max may be None");
  }
  array result = a;
  if (a_min.has_value()) {
    result = maximum(result, a_min.value(), s);
  }
  if (a_max.has_value()) {
    result = minimum(result, a_max.value(), s);
  }
  return result;
}

array concatenate(
    std::vector<array> arrays,
    int axis,
    StreamOrDevice s /* = {} */) {
  if (arrays.size() == 0) {
    throw std::invalid_argument(
        "[concatenate] No arrays provided for concatenation");
  }
  if (arrays.size() == 1) {
    return arrays[0];
  }

  auto ax = normalize_axis_index(axis, arrays[0].ndim(), "[concatenate] ");

  auto throw_invalid_shapes = [&]() {
    std::ostringstream msg;
    msg << "[concatenate] All the input array dimensions must match exactly "
        << "except for the concatenation axis. However, the provided shapes are ";
    for (auto& a : arrays) {
      msg << a.shape() << ", ";
    }
    msg << "and the concatenation axis is " << axis << ".";
    throw std::invalid_argument(msg.str());
  };

  auto shape = arrays[0].shape();
  shape[ax] = 0;
  // Make the output shape and validate that all arrays have the same shape
  // except for the concatenation axis.
  for (auto& a : arrays) {
    if (a.ndim() != shape.size()) {
      std::ostringstream msg;
      msg << "[concatenate] All the input arrays must have the same number of "
          << "dimensions. However, got arrays with dimensions " << shape.size()
          << " and " << a.ndim() << ".";
      throw std::invalid_argument(msg.str());
    }
    for (int i = 0; i < a.ndim(); i++) {
      if (i == ax) {
        continue;
      }
      if (a.shape(i) != shape[i]) {
        throw_invalid_shapes();
      }
    }
    shape[ax] += a.shape(ax);
  }

  // Promote all the arrays to the same type
  auto dtype = result_type(arrays);
  for (auto& a : arrays) {
    a = astype(a, dtype, s);
  }

  return array(
      std::move(shape),
      dtype,
      std::make_shared<Concatenate>(to_stream(s), ax),
      std::move(arrays));
}

array concatenate(std::vector<array> arrays, StreamOrDevice s /* = {} */) {
  for (auto& a : arrays) {
    a = flatten(a, s);
  }
  return concatenate(std::move(arrays), 0, s);
}

/** Stack arrays along a new axis */
array stack(
    const std::vector<array>& arrays,
    int axis,
    StreamOrDevice s /* = {} */) {
  if (arrays.empty()) {
    throw std::invalid_argument("[stack] No arrays provided for stacking");
  }
  if (!std::all_of(arrays.begin(), arrays.end(), [&](const auto& a) {
        return arrays[0].shape() == a.shape();
      })) {
    throw std::invalid_argument("[stack] All arrays must have the same shape");
  }
  auto normalized_axis =
      normalize_axis_index(axis, arrays[0].ndim() + 1, "[stack] ");
  std::vector<array> new_arrays;
  new_arrays.reserve(arrays.size());
  for (auto& a : arrays) {
    new_arrays.emplace_back(expand_dims(a, normalized_axis, s));
  }
  return concatenate(new_arrays, axis, s);
}

array stack(const std::vector<array>& arrays, StreamOrDevice s /* = {} */) {
  return stack(arrays, 0, s);
}

/** array repeat with axis */
array repeat(const array& arr, int repeats, int axis, StreamOrDevice s) {
  axis = normalize_axis_index(axis, arr.ndim(), "[repeat] ");

  if (repeats < 0) {
    throw std::invalid_argument(
        "[repeat] Number of repeats cannot be negative");
  }

  if (repeats == 0) {
    return array({}, arr.dtype());
  }

  if (repeats == 1) {
    return arr;
  }

  // Broadcast to (S_1, S_2, ..., S_axis, repeats, S_axis+1, ...)
  auto shape = arr.shape();
  shape.insert(shape.begin() + axis + 1, repeats);
  array out = expand_dims(arr, axis + 1, s);
  out = broadcast_to(out, shape, s);

  // Reshape back into a contiguous array where S_axis is now S_axis * repeats
  shape.erase(shape.begin() + axis + 1);
  shape[axis] *= repeats;
  out = reshape(out, shape, s);

  return out;
}

array repeat(const array& arr, int repeats, StreamOrDevice s) {
  return repeat(flatten(arr, s), repeats, 0, s);
}

array tile(
    const array& arr,
    std::vector<int> reps,
    StreamOrDevice s /* = {} */) {
  auto shape = arr.shape();
  if (reps.size() < shape.size()) {
    reps.insert(reps.begin(), shape.size() - reps.size(), 1);
  }
  if (reps.size() > shape.size()) {
    shape.insert(shape.begin(), reps.size() - shape.size(), 1);
  }

  Shape expand_shape;
  Shape broad_shape;
  Shape final_shape;
  for (int i = 0; i < shape.size(); i++) {
    if (reps[i] != 1) {
      expand_shape.push_back(1);
      broad_shape.push_back(reps[i]);
    }
    expand_shape.push_back(shape[i]);
    broad_shape.push_back(shape[i]);
    final_shape.push_back(reps[i] * shape[i]);
  }

  auto x = reshape(arr, std::move(expand_shape), s);
  x = broadcast_to(x, std::move(broad_shape), s);
  return reshape(x, std::move(final_shape), s);
}

array edge_pad(
    const array& a,
    const std::vector<int>& axes,
    const Shape& low_pad_size,
    const Shape& high_pad_size,
    const Shape& out_shape,
    StreamOrDevice s /* = {}*/) {
  array out = zeros(out_shape, a.dtype(), s);
  auto stops = a.shape();
  for (int i = 0; i < stops.size(); i++) {
    stops[i] += low_pad_size[i];
  }
  // Copy over values from the unpadded array
  array padded = slice_update(out, a, low_pad_size, stops, s);

  for (int axis = 0; axis < a.ndim(); axis++) {
    if (low_pad_size[axis] > 0) {
      Shape starts(a.ndim(), 0);
      starts[axis] = low_pad_size[axis];
      auto stops = out.shape();
      stops[axis] = low_pad_size[axis] + 1;
      // Fetch edge values
      array edge_value = slice(padded, starts, stops, s);

      starts[axis] = 0;
      stops[axis] = low_pad_size[axis];
      // Update edge values in the padded array
      padded = slice_update(padded, edge_value, starts, stops, s);
    }

    if (high_pad_size[axis] > 0) {
      Shape starts(a.ndim(), 0);
      starts[axis] = -high_pad_size[axis] - 1;
      auto stops = out.shape();
      stops[axis] = -high_pad_size[axis];
      array edge_value = slice(padded, starts, stops, s);

      starts[axis] = -high_pad_size[axis];
      stops[axis] = out.shape(axis);
      padded = slice_update(padded, edge_value, starts, stops, s);
    }
  }
  return padded;
}

/** Pad an array with a constant value */
array pad(
    const array& a,
    const std::vector<int>& axes,
    const Shape& low_pad_size,
    const Shape& high_pad_size,
    const array& pad_value /*= array(0)*/,
    const std::string& mode /*= "constant"*/,
    StreamOrDevice s /* = {}*/) {
  if (axes.size() != low_pad_size.size() ||
      axes.size() != high_pad_size.size()) {
    std::ostringstream msg;
    msg << "Invalid number of padding sizes passed to pad "
        << "with axes of size " << axes.size();
    throw std::invalid_argument(msg.str());
  }

  auto out_shape = a.shape();

  for (int i = 0; i < axes.size(); i++) {
    if (low_pad_size[i] < 0) {
      std::ostringstream msg;
      msg << "Invalid low padding size (" << low_pad_size[i]
          << ") passed to pad for axis " << i
          << ". Padding sizes must be non-negative";
      throw std::invalid_argument(msg.str());
    }
    if (high_pad_size[i] < 0) {
      std::ostringstream msg;
      msg << "Invalid high padding size (" << high_pad_size[i]
          << ") passed to pad for axis " << i
          << ". Padding sizes must be non-negative";
      throw std::invalid_argument(msg.str());
    }

    auto ax = axes[i] < 0 ? a.ndim() + axes[i] : axes[i];
    out_shape[ax] += low_pad_size[i] + high_pad_size[i];
  }

  if (mode == "constant") {
    return array(
        std::move(out_shape),
        a.dtype(),
        std::make_shared<Pad>(to_stream(s), axes, low_pad_size, high_pad_size),
        {a, astype(pad_value, a.dtype(), s)});
  } else if (mode == "edge") {
    return edge_pad(a, axes, low_pad_size, high_pad_size, out_shape, s);
  } else {
    std::ostringstream msg;
    msg << "Invalid padding mode (" << mode << ") passed to pad";
    throw std::invalid_argument(msg.str());
  }
}

/** Pad an array with a constant value along all axes */
array pad(
    const array& a,
    const std::vector<std::pair<int, int>>& pad_width,
    const array& pad_value /*= array(0)*/,
    const std::string& mode /*= "constant"*/,
    StreamOrDevice s /*= {}*/) {
  std::vector<int> axes(a.ndim(), 0);
  std::iota(axes.begin(), axes.end(), 0);

  Shape lows;
  Shape highs;

  for (auto& pads : pad_width) {
    lows.push_back(pads.first);
    highs.push_back(pads.second);
  }

  return pad(a, axes, lows, highs, pad_value, mode, s);
}

array pad(
    const array& a,
    const std::pair<int, int>& pad_width,
    const array& pad_value /*= array(0)*/,
    const std::string& mode /*= "constant"*/,
    StreamOrDevice s /*= {}*/) {
  return pad(
      a,
      std::vector<std::pair<int, int>>(a.ndim(), pad_width),
      pad_value,
      mode,
      s);
}

array pad(
    const array& a,
    int pad_width,
    const array& pad_value /*= array(0)*/,
    const std::string& mode /*= "constant"*/,
    StreamOrDevice s /*= {}*/) {
  return pad(
      a,
      std::vector<std::pair<int, int>>(a.ndim(), {pad_width, pad_width}),
      pad_value,
      mode,
      s);
}

array moveaxis(
    const array& a,
    int source,
    int destination,
    StreamOrDevice s /* = {} */) {
  auto check_ax = [&a](int ax) {
    auto ndim = static_cast<int>(a.ndim());
    if (ax < -ndim || ax >= ndim) {
      std::ostringstream msg;
      msg << "[moveaxis] Invalid axis " << ax << " for array with " << ndim
          << " dimensions.";
      throw std::out_of_range(msg.str());
    }
    return ax < 0 ? ax + ndim : ax;
  };
  source = check_ax(source);
  destination = check_ax(destination);
  if (source == destination) {
    return a;
  }
  std::vector<int> reorder(a.ndim());
  std::iota(reorder.begin(), reorder.end(), 0);
  reorder.erase(reorder.begin() + source);
  reorder.insert(reorder.begin() + destination, source);
  return transpose(a, reorder, s);
}

array swapaxes(
    const array& a,
    int axis1,
    int axis2,
    StreamOrDevice s /* = {} */) {
  auto check_ax = [&a](int ax) {
    auto ndim = static_cast<int>(a.ndim());
    if (ax < -ndim || ax >= ndim) {
      std::ostringstream msg;
      msg << "[swapaxes] Invalid axis " << ax << " for array with " << ndim
          << " dimensions.";
      throw std::out_of_range(msg.str());
    }
    return ax < 0 ? ax + ndim : ax;
  };
  axis1 = check_ax(axis1);
  axis2 = check_ax(axis2);
  std::vector<int> reorder(a.ndim());
  std::iota(reorder.begin(), reorder.end(), 0);
  std::swap(reorder[axis1], reorder[axis2]);
  return transpose(a, std::move(reorder), s);
}

array transpose(
    const array& a,
    std::vector<int> axes,
    StreamOrDevice s /* = {} */) {
  for (auto& ax : axes) {
    ax = ax < 0 ? ax + a.ndim() : ax;
  }
  if (axes.size() != a.ndim()) {
    std::ostringstream msg;
    msg << "[transpose] Recived " << axes.size() << " axes for array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  // Check in bounds and for duplicates
  Shape shape(axes.size(), 0);
  for (auto& ax : axes) {
    if (ax < 0 || ax >= a.ndim()) {
      std::ostringstream msg;
      msg << "[transpose] Invalid axis (" << ax << ") for array with "
          << a.ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if (shape[ax] != 0) {
      throw std::invalid_argument("[transpose] Repeat axes not allowed.");
    }
    shape[ax] = 1;
  }

  for (int i = 0; i < axes.size(); ++i) {
    shape[i] = a.shape()[axes[i]];
  }
  return array(
      std::move(shape),
      a.dtype(),
      std::make_shared<Transpose>(to_stream(s), std::move(axes)),
      {a});
}

array transpose(const array& a, StreamOrDevice s /* = {} */) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.rbegin(), axes.rend(), 0);
  return transpose(a, std::move(axes), to_stream(s));
}

array broadcast_to(
    const array& a,
    const Shape& shape,
    StreamOrDevice s /* = {} */) {
  if (a.shape() == shape) {
    return a;
  }

  // Make sure the shapes are broadcastable
  auto bxshape = broadcast_shapes(a.shape(), shape);
  if (bxshape != shape) {
    std::ostringstream msg;
    msg << "Cannot broadcast array of shape " << a.shape() << " into shape "
        << shape << ".";
    throw std::invalid_argument(msg.str());
  }
  return array(
      std::move(bxshape),
      a.dtype(),
      std::make_shared<Broadcast>(to_stream(s), shape),
      {a});
}

/** Broadcast the input arrays against one another while ignoring the
 * axes specified in `ignore_axes`. Note, this API is internal only.
 * The `ignore_axes` should be:
 * - negative values indicating axes from the end
 * - sorted in increasing order
 */
std::vector<array> broadcast_arrays(
    const std::vector<array>& inputs,
    std::vector<int> ignore_axes,
    StreamOrDevice s) {
  if (inputs.size() <= 1) {
    return inputs;
  }

  std::vector<array> outputs;
  auto shape = BroadcastAxes::output_shape(inputs, ignore_axes);
  auto check_and_get_shape = [&shape, &ignore_axes](const array& in) {
    auto out_shape = shape;
    for (int i = 0; i < ignore_axes.size(); ++i) {
      auto ax = ignore_axes[i];
      auto pos_ax = in.ndim() + ax;
      if (pos_ax < 0 || pos_ax > in.ndim() ||
          (i > 0 && ax <= ignore_axes[i - 1])) {
        throw std::invalid_argument(
            "[broadcast_arrays] Received invalid axes to ignore.");
      }
      out_shape[out_shape.size() + ax] = in.shape(ax);
    }
    return out_shape;
  };

  if (!detail::in_dynamic_tracing()) {
    for (auto& in : inputs) {
      auto out_shape = check_and_get_shape(in);
      if (in.shape() == out_shape) {
        outputs.push_back(in);
      } else {
        outputs.push_back(array(
            std::move(out_shape),
            in.dtype(),
            std::make_shared<Broadcast>(to_stream(s), out_shape),
            {in}));
      }
    }
    return outputs;
  }

  std::vector<array> stop_grad_inputs;
  for (auto& in : inputs) {
    stop_grad_inputs.push_back(stop_gradient(in, s));
  }

  for (int i = 0; i < inputs.size(); ++i) {
    auto& in = inputs[i];
    auto out_shape = check_and_get_shape(in);
    if (in.shape() == out_shape) {
      outputs.push_back(in);
    } else {
      // broadcasted array goes first followed by other stopgrad inputs
      std::vector<array> p_inputs = {in};
      for (int j = 0; j < inputs.size(); ++j) {
        if (j == i) {
          continue;
        }
        p_inputs.push_back(stop_grad_inputs[j]);
      }
      outputs.push_back(array(
          std::move(out_shape),
          in.dtype(),
          std::make_shared<BroadcastAxes>(to_stream(s), ignore_axes),
          std::move(p_inputs)));
    }
  }
  return outputs;
}

std::vector<array> broadcast_arrays(
    const std::vector<array>& inputs,
    StreamOrDevice s /* = {} */) {
  if (inputs.size() <= 1) {
    return inputs;
  }
  auto shape = Broadcast::output_shape(inputs);
  std::vector<array> outputs;

  if (!detail::in_dynamic_tracing()) {
    for (auto& in : inputs) {
      if (in.shape() == shape) {
        outputs.push_back(in);
      } else {
        outputs.push_back(array(
            shape,
            in.dtype(),
            std::make_shared<Broadcast>(to_stream(s), shape),
            {in}));
      }
    }
    return outputs;
  }

  std::vector<array> stop_grad_inputs;
  for (auto& in : inputs) {
    stop_grad_inputs.push_back(stop_gradient(in, s));
  }
  for (int i = 0; i < inputs.size(); ++i) {
    auto& in = inputs[i];
    if (in.shape() == shape) {
      outputs.push_back(in);
    } else {
      // broadcasted array goes first followed by other stopgrad inputs
      std::vector<array> p_inputs = {in};
      for (int j = 0; j < inputs.size(); ++j) {
        if (j == i) {
          continue;
        }
        p_inputs.push_back(stop_grad_inputs[j]);
      }
      outputs.push_back(array(
          shape,
          in.dtype(),
          std::make_shared<Broadcast>(to_stream(s), shape),
          std::move(p_inputs)));
    }
  }
  return outputs;
}

std::pair<array, array>
broadcast_arrays(const array& a, const array& b, StreamOrDevice s) {
  auto out = broadcast_arrays({a, b}, s);
  return {out[0], out[1]};
}

std::pair<array, array> broadcast_arrays(
    const array& a,
    const array& b,
    std::vector<int> ignore_axes,
    StreamOrDevice s) {
  auto out = broadcast_arrays({a, b}, std::move(ignore_axes), s);
  return {out[0], out[1]};
}

array equal(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  auto inputs = broadcast_arrays({astype(a, dtype, s), astype(b, dtype, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape, bool_, std::make_shared<Equal>(to_stream(s)), std::move(inputs));
}

array not_equal(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  auto inputs = broadcast_arrays({astype(a, dtype, s), astype(b, dtype, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape,
      bool_,
      std::make_shared<NotEqual>(to_stream(s)),
      std::move(inputs));
}

array greater(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  auto inputs = broadcast_arrays({astype(a, dtype, s), astype(b, dtype, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape, bool_, std::make_shared<Greater>(to_stream(s)), std::move(inputs));
}

array greater_equal(
    const array& a,
    const array& b,
    StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  auto inputs = broadcast_arrays({astype(a, dtype, s), astype(b, dtype, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape,
      bool_,
      std::make_shared<GreaterEqual>(to_stream(s)),
      std::move(inputs));
}

array less(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  auto inputs = broadcast_arrays({astype(a, dtype, s), astype(b, dtype, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape, bool_, std::make_shared<Less>(to_stream(s)), std::move(inputs));
}

array less_equal(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  auto inputs = broadcast_arrays({astype(a, dtype, s), astype(b, dtype, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape,
      bool_,
      std::make_shared<LessEqual>(to_stream(s)),
      std::move(inputs));
}

array array_equal(
    const array& a,
    const array& b,
    bool equal_nan,
    StreamOrDevice s /* = {} */) {
  if (a.shape() != b.shape()) {
    return array(false);
  } else {
    auto dtype = promote_types(a.dtype(), b.dtype());
    equal_nan &= issubdtype(dtype, inexact);
    return all(
        array(
            a.shape(),
            bool_,
            std::make_shared<Equal>(to_stream(s), equal_nan),
            {astype(a, dtype, s), astype(b, dtype, s)}),
        false,
        s);
  }
}

array isnan(const array& a, StreamOrDevice s /* = {} */) {
  if (issubdtype(a.dtype(), integer) || a.dtype() == bool_) {
    return full(a.shape(), false, bool_, s);
  }
  return not_equal(a, a, s);
}

array isinf(const array& a, StreamOrDevice s /* = {} */) {
  if (issubdtype(a.dtype(), integer) || a.dtype() == bool_) {
    return full(a.shape(), false, bool_, s);
  }
  return logical_or(isposinf(a, s), isneginf(a, s), s);
}

array isfinite(const array& a, StreamOrDevice s /* = {} */) {
  if (issubdtype(a.dtype(), integer) || a.dtype() == bool_) {
    return full(a.shape(), true, bool_, s);
  }
  return logical_not(logical_or(isinf(a, s), isnan(a, s), s), s);
}

array isposinf(const array& a, StreamOrDevice s /* = {} */) {
  if (issubdtype(a.dtype(), integer) || a.dtype() == bool_) {
    return full(a.shape(), false, bool_, s);
  }
  return equal(a, array(std::numeric_limits<float>::infinity(), a.dtype()), s);
}

array isneginf(const array& a, StreamOrDevice s /* = {} */) {
  if (issubdtype(a.dtype(), integer) || a.dtype() == bool_) {
    return full(a.shape(), false, bool_, s);
  }
  return equal(a, array(-std::numeric_limits<float>::infinity(), a.dtype()), s);
}

array where(
    const array& a,
    const array& b,
    const array& c,
    StreamOrDevice s /* = {} */) {
  auto condition = astype(a, bool_, s);
  Dtype out_dtype = promote_types(b.dtype(), c.dtype());
  auto inputs = broadcast_arrays(
      {condition, astype(b, out_dtype, s), astype(c, out_dtype, s)}, s);

  return array(
      inputs[0].shape(),
      out_dtype,
      std::make_shared<Select>(to_stream(s)),
      inputs);
}

array nan_to_num(
    const array& a,
    float nan /* = 0.0f */,
    const std::optional<float> posinf_ /* = std::nullopt */,
    const std::optional<float> neginf_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  Dtype dtype = a.dtype();
  if (!issubdtype(dtype, inexact)) {
    return a;
  }

  auto type_to_max = [](const auto& dtype) -> float {
    if (dtype == float32) {
      return std::numeric_limits<float>::max();
    } else if (dtype == bfloat16) {
      return std::numeric_limits<bfloat16_t>::max();
    } else if (dtype == float16) {
      return std::numeric_limits<float16_t>::max();
    } else {
      std::ostringstream msg;
      msg << "[nan_to_num] Does not yet support given type: " << dtype << ".";
      throw std::invalid_argument(msg.str());
    }
  };

  float posinf = posinf_ ? *posinf_ : type_to_max(dtype);
  float neginf = neginf_ ? *neginf_ : -type_to_max(dtype);

  auto out = where(isnan(a, s), array(nan, dtype), a, s);
  out = where(isposinf(a, s), array(posinf, dtype), out, s);
  out = where(isneginf(a, s), array(neginf, dtype), out, s);
  return out;
}

array allclose(
    const array& a,
    const array& b,
    double rtol /* = 1e-5 */,
    double atol /* = 1e-8 */,
    bool equal_nan /* = false */,
    StreamOrDevice s /* = {}*/) {
  return all(isclose(a, b, rtol, atol, equal_nan, s), s);
}

array isclose(
    const array& a,
    const array& b,
    double rtol /* = 1e-5 */,
    double atol /* = 1e-8 */,
    bool equal_nan /* = false */,
    StreamOrDevice s /* = {}*/) {
  // |a - b| <= atol + rtol * |b|
  auto rhs = add(array(atol), multiply(array(rtol), abs(b, s), s), s);
  auto lhs = abs(subtract(a, b, s), s);
  auto out = less_equal(lhs, rhs, s);

  // Correct the result for infinite values.
  auto a_pos_inf = isposinf(a, s);
  auto b_pos_inf = isposinf(b, s);
  auto a_neg_inf = isneginf(a, s);
  auto b_neg_inf = isneginf(b, s);
  auto any_inf = logical_or(
      logical_or(a_pos_inf, a_neg_inf, s),
      logical_or(b_pos_inf, b_neg_inf, s),
      s);
  auto both_inf = logical_or(
      logical_and(a_pos_inf, b_pos_inf, s),
      logical_and(a_neg_inf, b_neg_inf, s),
      s);

  // Convert all elements where either value is infinite to False.
  out = logical_and(out, logical_not(any_inf, s), s);

  // Convert all the elements where both values are infinite and of the same
  // sign to True.
  out = logical_or(out, both_inf, s);

  if (equal_nan) {
    auto both_nan = logical_and(isnan(a, s), isnan(b, s), s);
    out = logical_or(out, both_nan, s);
  }

  return out;
}

array all(const array& a, bool keepdims, StreamOrDevice s /* = {}*/) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return all(a, axes, keepdims, s);
}

array all(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {}*/) {
  auto [out_shape, sorted_axes, is_noop] =
      compute_reduce_shape(axes, a.shape());
  auto out = (is_noop)
      ? astype(a, bool_, s)
      : array(
            std::move(out_shape),
            bool_,
            std::make_shared<Reduce>(to_stream(s), Reduce::And, sorted_axes),
            {a});
  if (!keepdims) {
    out = squeeze(out, sorted_axes, s);
  }
  return out;
}

array all(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  return all(a, std::vector<int>{axis}, keepdims, s);
}

array any(const array& a, bool keepdims, StreamOrDevice s /* = {}*/) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return any(a, axes, keepdims, s);
}

array any(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {}*/) {
  auto [out_shape, sorted_axes, is_noop] =
      compute_reduce_shape(axes, a.shape());
  auto out = (is_noop)
      ? astype(a, bool_, s)
      : array(
            std::move(out_shape),
            bool_,
            std::make_shared<Reduce>(to_stream(s), Reduce::Or, sorted_axes),
            {a});
  if (!keepdims) {
    out = squeeze(out, sorted_axes, s);
  }
  return out;
}

array any(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  return any(a, std::vector<int>{axis}, keepdims, s);
}

array sum(const array& a, bool keepdims, StreamOrDevice s /* = {}*/) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return sum(a, axes, keepdims, s);
}

array sum(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {}*/) {
  if (axes.empty()) {
    return a;
  }
  auto [out_shape, sorted_axes, is_noop] =
      compute_reduce_shape(axes, a.shape());
  Dtype out_type = a.dtype();
  if (issubdtype(a.dtype(), signedinteger)) {
    out_type = a.dtype().size() <= 4 ? int32 : int64;
  } else if (issubdtype(a.dtype(), unsignedinteger)) {
    out_type = a.dtype().size() <= 4 ? uint32 : uint64;
  } else if (a.dtype() == bool_) {
    out_type = int32;
  }
  auto out = (is_noop)
      ? astype(a, out_type, s)
      : array(
            std::move(out_shape),
            out_type,
            std::make_shared<Reduce>(to_stream(s), Reduce::Sum, sorted_axes),
            {a});
  if (!keepdims) {
    out = squeeze(out, sorted_axes, s);
  }
  return out;
}

array sum(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  return sum(a, std::vector<int>{axis}, keepdims, s);
}

array mean(const array& a, bool keepdims, StreamOrDevice s /* = {}*/) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return mean(a, axes, keepdims, to_stream(s));
}

array mean(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {}*/) {
  int ndim = a.ndim();
  for (int axis : axes) {
    if (axis < -ndim || axis >= ndim) {
      std::ostringstream msg;
      msg << "[mean] axis " << axis << " is out of bounds for array with "
          << ndim << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
  }
  auto dtype = at_least_float(a.dtype());
  auto normalizer = number_of_elements(a, axes, true, dtype, s);
  return multiply(sum(a, axes, keepdims, s), normalizer, s);
}

array mean(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  return mean(a, std::vector<int>{axis}, keepdims, to_stream(s));
}

array median(const array& a, bool keepdims, StreamOrDevice s /* = {}*/) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return median(a, axes, keepdims, to_stream(s));
}

array median(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {}*/) {
  int ndim = a.ndim();
  std::set<int> set_axes;
  for (int axis : axes) {
    if (axis < -ndim || axis >= ndim) {
      std::ostringstream msg;
      msg << "[median] axis " << axis << " is out of bounds for array with "
          << ndim << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    set_axes.insert(axis < 0 ? axis + ndim : axis);
  }
  if (set_axes.size() != axes.size()) {
    throw std::invalid_argument("[median] Received duplicate axis.");
  }
  std::vector<int> sorted_axes(set_axes.begin(), set_axes.end());
  auto dtype = at_least_float(a.dtype());
  std::vector<int> transpose_axes;
  for (int i = 0, j = 0; i < a.ndim(); ++i) {
    if (j < sorted_axes.size() && i == sorted_axes[j]) {
      j++;
      continue;
    }
    transpose_axes.push_back(i);
  }
  int flat_start = transpose_axes.size();
  transpose_axes.insert(
      transpose_axes.end(), sorted_axes.begin(), sorted_axes.end());

  // Move all the median axes to the back and flatten
  auto flat_a =
      flatten(transpose(a, transpose_axes, s), flat_start, a.ndim(), s);
  int flat_size = flat_a.shape(-1);
  if (flat_size == 0) {
    throw std::invalid_argument(
        "[median] Cannot take median along empty axis.");
  }

  // Sort the last axis
  auto sorted_a = sort(flat_a, -1, s);

  // Take the midpoint
  auto mp = flat_size / 2;
  auto start = Shape(sorted_a.ndim(), 0);
  auto stop = sorted_a.shape();
  start.back() = mp;
  stop.back() = mp + 1;
  auto median_a = astype(slice(sorted_a, start, stop, s), dtype, s);
  if (flat_size % 2 == 0) {
    start.back() = mp - 1;
    stop.back() = mp;
    median_a = multiply(
        add(median_a, astype(slice(sorted_a, start, stop, s), dtype, s), s),
        array(0.5, dtype),
        s);
  }
  median_a = squeeze(median_a, -1, s);
  if (keepdims) {
    median_a = expand_dims(median_a, sorted_axes, s);
  }
  return median_a;
}

array median(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  return median(a, std::vector<int>{axis}, keepdims, to_stream(s));
}

array var(
    const array& a,
    bool keepdims,
    int ddof /* = 0*/,
    StreamOrDevice s /* = {}*/) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return var(a, axes, keepdims, ddof, to_stream(s));
}

array var(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims /* = false */,
    int ddof /* = 0*/,
    StreamOrDevice s /* = {}*/) {
  auto dtype = at_least_float(a.dtype());
  auto mu = mean(a, axes, /* keepdims= */ true, s);
  auto v = sum(square(subtract(a, mu, s), s), axes, keepdims, s);

  if (ddof != 0) {
    auto normalizer = maximum(
        subtract(
            number_of_elements(a, axes, false, dtype, s),
            array(ddof, dtype),
            s),
        array(0, dtype),
        s);
    v = divide(v, normalizer, s);
  } else {
    auto normalizer = number_of_elements(a, axes, true, dtype, s);
    v = multiply(v, normalizer, s);
  }

  return v;
}

array var(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    int ddof /* = 0*/,
    StreamOrDevice s /* = {} */) {
  return var(a, std::vector<int>{axis}, keepdims, ddof, to_stream(s));
}

array std(
    const array& a,
    bool keepdims,
    int ddof /* = 0*/,
    StreamOrDevice s /* = {}*/) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return std(a, axes, keepdims, ddof, to_stream(s));
}

array std(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims /* = false */,
    int ddof /* = 0*/,
    StreamOrDevice s /* = {}*/) {
  return sqrt(var(a, axes, keepdims, ddof, s), s);
}

array std(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    int ddof /* = 0*/,
    StreamOrDevice s /* = {} */) {
  return std(a, std::vector<int>{axis}, keepdims, ddof, to_stream(s));
}

array prod(const array& a, bool keepdims, StreamOrDevice s /* = {}*/) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return prod(a, axes, keepdims, s);
}

array prod(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {}*/) {
  if (axes.empty()) {
    return a;
  }
  auto [out_shape, sorted_axes, is_noop] =
      compute_reduce_shape(axes, a.shape());
  Dtype out_type = a.dtype();
  if (issubdtype(a.dtype(), signedinteger)) {
    out_type = a.dtype().size() <= 4 ? int32 : int64;
  } else if (issubdtype(a.dtype(), unsignedinteger)) {
    out_type = a.dtype().size() <= 4 ? uint32 : uint64;
  } else if (a.dtype() == bool_) {
    out_type = int32;
  }
  auto out = (is_noop)
      ? a
      : array(
            std::move(out_shape),
            out_type,
            std::make_shared<Reduce>(to_stream(s), Reduce::Prod, sorted_axes),
            {a});
  if (!keepdims) {
    out = squeeze(out, sorted_axes, s);
  }
  return out;
}

array prod(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  return prod(a, std::vector<int>{axis}, keepdims, s);
}

array max(const array& a, bool keepdims, StreamOrDevice s /* = {}*/) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return max(a, axes, keepdims, s);
}

array max(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {}*/) {
  if (a.size() == 0) {
    throw std::invalid_argument("[max] Cannot max reduce zero size array.");
  }
  auto [out_shape, sorted_axes, is_noop] =
      compute_reduce_shape(axes, a.shape());
  auto out = (is_noop)
      ? a
      : array(
            std::move(out_shape),
            a.dtype(),
            std::make_shared<Reduce>(to_stream(s), Reduce::Max, sorted_axes),
            {a});
  if (!keepdims) {
    out = squeeze(out, sorted_axes, s);
  }
  return out;
}

array max(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  return max(a, std::vector<int>{axis}, keepdims, s);
}

array min(const array& a, bool keepdims, StreamOrDevice s /* = {}*/) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return min(a, axes, keepdims, s);
}

array min(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {}*/) {
  if (a.size() == 0) {
    throw std::invalid_argument("[min] Cannot min reduce zero size array.");
  }
  if (axes.empty()) {
    return a;
  }
  auto [out_shape, sorted_axes, is_noop] =
      compute_reduce_shape(axes, a.shape());
  auto out = (is_noop)
      ? a
      : array(
            std::move(out_shape),
            a.dtype(),
            std::make_shared<Reduce>(to_stream(s), Reduce::Min, sorted_axes),
            {a});
  if (!keepdims) {
    out = squeeze(out, sorted_axes, s);
  }
  return out;
}

array min(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  return min(a, std::vector<int>{axis}, keepdims, s);
}

array argmin(const array& a, bool keepdims, StreamOrDevice s /* = {} */) {
  auto result = argmin(flatten(a, s), 0, true, s);
  if (keepdims) {
    std::vector<int> axes(a.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    result = expand_dims(result, axes, s);
  } else {
    result = squeeze(result, s);
  }
  return result;
}

array argmin(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  if (a.size() == 0) {
    throw std::invalid_argument(
        "[argmin] Cannot argmin reduce zero size array.");
  }
  auto [out_shape, sorted_axes, is_noop] =
      compute_reduce_shape({axis}, a.shape());
  auto out = (is_noop)
      ? zeros(out_shape, uint32, s)
      : array(
            std::move(out_shape),
            uint32,
            std::make_shared<ArgReduce>(
                to_stream(s), ArgReduce::ArgMin, sorted_axes[0]),
            {a});
  if (!keepdims) {
    out = squeeze(out, sorted_axes[0], s);
  }
  return out;
}

array argmax(const array& a, bool keepdims, StreamOrDevice s /* = {} */) {
  auto result = argmax(flatten(a, s), 0, true, s);
  if (keepdims) {
    std::vector<int> axes(a.ndim() - 1);
    std::iota(axes.begin(), axes.end(), 0);
    result = expand_dims(result, axes, s);
  } else {
    result = squeeze(result, s);
  }
  return result;
}

array argmax(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  if (a.size() == 0) {
    throw std::invalid_argument(
        "[argmax] Cannot argmax reduce zero size array.");
  }
  auto [out_shape, sorted_axes, is_noop] =
      compute_reduce_shape({axis}, a.shape());
  auto out = (is_noop)
      ? zeros(out_shape, uint32, s)
      : array(
            std::move(out_shape),
            uint32,
            std::make_shared<ArgReduce>(
                to_stream(s), ArgReduce::ArgMax, sorted_axes[0]),
            {a});
  if (!keepdims) {
    out = squeeze(out, sorted_axes[0], s);
  }
  return out;
}

array hanning(int M, StreamOrDevice s /* = {} */) {
  if (M < 1) {
    return array({});
  }
  if (M == 1) {
    return ones({1}, float32, s);
  }

  auto n = arange(0, M, float32, s);
  array factor(M_PI / (M - 1), float32);
  return square(sin(multiply(factor, n, s), s), s);
}

array hamming(int M, StreamOrDevice s /* = {} */) {
  if (M < 1) {
    return array({});
  }
  if (M == 1) {
    return ones({1}, float32, s);
  }

  auto n = arange(0, M, float32, s);
  float factor_val = (2.0 * M_PI) / (M - 1);
  auto factor = array(factor_val, float32);

  auto arg = multiply(factor, n, s);
  auto cos_vals = cos(arg, s);

  auto left_coef = array(0.54f, float32);
  auto right_coef = array(0.46f, float32);

  return subtract(left_coef, multiply(right_coef, cos_vals, s), s);
}

/** Returns a sorted copy of the flattened array. */
array sort(const array& a, StreamOrDevice s /* = {} */) {
  int size = a.size();
  return sort(reshape(a, {size}, s), 0, s);
}

/** Returns a sorted copy of the array along a given axis. */
array sort(const array& a, int axis, StreamOrDevice s /* = {} */) {
  // Check for valid axis
  if (axis + static_cast<int>(a.ndim()) < 0 ||
      axis >= static_cast<int>(a.ndim())) {
    std::ostringstream msg;
    msg << "[sort] Received invalid axis " << axis << " for array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  return array(
      a.shape(), a.dtype(), std::make_shared<Sort>(to_stream(s), axis), {a});
}

/** Returns indices that sort the flattened array. */
array argsort(const array& a, StreamOrDevice s /* = {} */) {
  int size = a.size();
  return argsort(reshape(a, {size}, s), 0, s);
}

/** Returns indices that sort the array along a given axis. */
array argsort(const array& a, int axis, StreamOrDevice s /* = {} */) {
  // Check for valid axis
  if (axis + static_cast<int>(a.ndim()) < 0 ||
      axis >= static_cast<int>(a.ndim())) {
    std::ostringstream msg;
    msg << "[argsort] Received invalid axis " << axis << " for array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  return array(
      a.shape(), uint32, std::make_shared<ArgSort>(to_stream(s), axis), {a});
}

/**
 * Returns a partitioned copy of the flattened array
 * such that the smaller kth elements are first.
 **/
array partition(const array& a, int kth, StreamOrDevice s /* = {} */) {
  int size = a.size();
  return partition(reshape(a, {size}, s), kth, 0, s);
}

/**
 * Returns a partitioned copy of the array along a given axis
 * such that the smaller kth elements are first.
 **/
array partition(
    const array& a,
    int kth,
    int axis,
    StreamOrDevice s /* = {} */) {
  // Check for valid axis
  if (axis + static_cast<int>(a.ndim()) < 0 ||
      axis >= static_cast<int>(a.ndim())) {
    std::ostringstream msg;
    msg << "[partition] Received invalid axis " << axis << " for array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  int axis_ = axis < 0 ? axis + a.ndim() : axis;
  int kth_ = kth < 0 ? kth + a.shape(axis) : kth;
  if (kth_ < 0 || kth_ >= a.shape(axis_)) {
    std::ostringstream msg;
    msg << "[partition] Received invalid kth " << kth << "along axis " << axis
        << " for array with shape: " << a.shape();
    throw std::invalid_argument(msg.str());
  }
  return array(
      a.shape(),
      a.dtype(),
      std::make_shared<Partition>(to_stream(s), kth_, axis_),
      {a});
}

/**
 * Returns indices that partition the flattened array
 * such that the smaller kth elements are first.
 **/
array argpartition(const array& a, int kth, StreamOrDevice s /* = {} */) {
  int size = a.size();
  return argpartition(reshape(a, {size}, s), kth, 0, s);
}

/**
 * Returns indices that partition the array along a given axis
 * such that the smaller kth elements are first.
 **/
array argpartition(
    const array& a,
    int kth,
    int axis,
    StreamOrDevice s /* = {} */) {
  // Check for valid axis
  if (axis + static_cast<int>(a.ndim()) < 0 ||
      axis >= static_cast<int>(a.ndim())) {
    std::ostringstream msg;
    msg << "[argpartition] Received invalid axis " << axis << " for array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  int axis_ = axis < 0 ? axis + a.ndim() : axis;
  int kth_ = kth < 0 ? kth + a.shape(axis) : kth;
  if (kth_ < 0 || kth_ >= a.shape(axis_)) {
    std::ostringstream msg;
    msg << "[argpartition] Received invalid kth " << kth << " along axis "
        << axis << " for array with shape: " << a.shape();
    throw std::invalid_argument(msg.str());
  }
  return array(
      a.shape(),
      uint32,
      std::make_shared<ArgPartition>(to_stream(s), kth_, axis_),
      {a});
}

/** Returns topk elements of the flattened array. */
array topk(const array& a, int k, StreamOrDevice s /* = {}*/) {
  int size = a.size();
  return topk(reshape(a, {size}, s), k, 0, s);
}

/** Returns topk elements of the array along a given axis. */
array topk(const array& a, int k, int axis, StreamOrDevice s /* = {}*/) {
  // Check for valid axis
  int axis_ = axis < 0 ? axis + a.ndim() : axis;
  if (axis_ < 0 || axis_ >= static_cast<int>(a.ndim())) {
    std::ostringstream msg;
    msg << "[topk] Received invalid axis " << axis << " for array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (k < 0 || k > a.shape(axis_)) {
    std::ostringstream msg;
    msg << "[topk] Received invalid k=" << k << " along axis " << axis
        << " for array with shape: " << a.shape();
    throw std::invalid_argument(msg.str());
  }

  // Return early if the whole input was requested.
  if (k == a.shape(axis_)) {
    return a;
  }

  array a_partitioned = partition(a, -k, axis_, s);
  Shape slice_starts(a.ndim(), 0);
  auto slice_ends = a.shape();
  slice_starts[axis_] = a.shape(axis_) - k;
  return slice(a_partitioned, slice_starts, slice_ends, s);
}

array logsumexp(const array& a, bool keepdims, StreamOrDevice s /* = {}*/) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return logsumexp(a, axes, keepdims, s);
}

array logsumexp(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {}*/) {
  if (a.size() == 0) {
    throw std::invalid_argument("[logsumexp] Received empty array.");
  }
  if (a.ndim() == 0 && !axes.empty()) {
    throw std::invalid_argument(
        "[logsumexp] Received non-empty axes for array with 0 dimensions.");
  }
  bool reduce_last_dim =
      !axes.empty() && (axes.back() == a.ndim() - 1 || axes.back() == -1);
  if (reduce_last_dim) {
    // For more than 2 axes check if axes is [0, 1, ..., NDIM - 1] and shape
    // is [1, 1, ..., N].
    for (int i = axes.size() - 2; i >= 0; --i) {
      if ((axes[i] + 1 != axes[i + 1]) || (a.shape(axes[i]) != 1)) {
        reduce_last_dim = false;
        break;
      }
    }
  }
  bool is_complex = issubdtype(a.dtype(), complexfloating);
  if (!is_complex && reduce_last_dim) {
    auto dtype = at_least_float(a.dtype());
    auto out_shape = a.shape();
    out_shape.back() = 1;
    auto out = array(
        std::move(out_shape),
        dtype,
        std::make_shared<LogSumExp>(to_stream(s)),
        {astype(a, dtype, s)});
    if (!keepdims) {
      out = squeeze(out, -1, s);
    }
    return out;
  }
  auto maxval = stop_gradient(max(a, axes, true, s), s);
  auto out = log(sum(exp(subtract(a, maxval, s), s), axes, keepdims, s), s);
  out = add(out, reshape(maxval, out.shape(), s), s);
  if (!keepdims) {
    maxval = squeeze(maxval, axes, s);
  }
  return where(isinf(maxval, s), maxval, out, s);
}

array logsumexp(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  return logsumexp(a, std::vector<int>{axis}, keepdims, s);
}

array abs(const array& a, StreamOrDevice s /* = {} */) {
  auto out =
      array(a.shape(), a.dtype(), std::make_shared<Abs>(to_stream(s)), {a});
  if (a.dtype() == complex64) {
    out = astype(out, float32, s);
  }
  return out;
}

array negative(const array& a, StreamOrDevice s /* = {} */) {
  if (a.dtype() == bool_) {
    auto msg = "[negative] Not supported for bool, use logical_not instead.";
    throw std::invalid_argument(msg);
  }
  return array(
      a.shape(), a.dtype(), std::make_shared<Negative>(to_stream(s)), {a});
}
array operator-(const array& a) {
  return negative(a);
}

array sign(const array& a, StreamOrDevice s /* = {} */) {
  return array(a.shape(), a.dtype(), std::make_shared<Sign>(to_stream(s)), {a});
}

array logical_not(const array& a, StreamOrDevice s /* = {} */) {
  return array(
      a.shape(),
      bool_,
      std::make_shared<LogicalNot>(to_stream(s)),
      {astype(a, bool_, s)});
}

array logical_and(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  // Broadcast arrays to a common shape
  auto inputs = broadcast_arrays({astype(a, bool_, s), astype(b, bool_, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape,
      bool_,
      std::make_shared<LogicalAnd>(to_stream(s)),
      std::move(inputs));
}
array operator&&(const array& a, const array& b) {
  return logical_and(a, b);
}

array logical_or(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  // Broadcast arrays to a common shape
  auto inputs = broadcast_arrays({astype(a, bool_, s), astype(b, bool_, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape,
      bool_,
      std::make_shared<LogicalOr>(to_stream(s)),
      std::move(inputs));
}
array operator||(const array& a, const array& b) {
  return logical_or(a, b);
}

array reciprocal(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  return divide(array(1.0f, dtype), a, to_stream(s));
}

array add(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto out_type = promote_types(a.dtype(), b.dtype());
  auto inputs =
      broadcast_arrays({astype(a, out_type, s), astype(b, out_type, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape, out_type, std::make_shared<Add>(to_stream(s)), std::move(inputs));
}

array operator+(const array& a, const array& b) {
  return add(a, b);
}

array subtract(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto out_type = promote_types(a.dtype(), b.dtype());
  auto inputs =
      broadcast_arrays({astype(a, out_type, s), astype(b, out_type, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape,
      out_type,
      std::make_shared<Subtract>(to_stream(s)),
      std::move(inputs));
}

array operator-(const array& a, const array& b) {
  return subtract(a, b);
}

array multiply(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto out_type = promote_types(a.dtype(), b.dtype());
  auto inputs =
      broadcast_arrays({astype(a, out_type, s), astype(b, out_type, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape,
      out_type,
      std::make_shared<Multiply>(to_stream(s)),
      std::move(inputs));
}

array operator*(const array& a, const array& b) {
  return multiply(a, b);
}

array divide(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(promote_types(a.dtype(), b.dtype()));
  auto inputs = broadcast_arrays(
      {astype(a, dtype, s), astype(b, dtype, to_stream(s))}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape, dtype, std::make_shared<Divide>(to_stream(s)), std::move(inputs));
}
array operator/(const array& a, const array& b) {
  return divide(a, b);
}
array operator/(double a, const array& b) {
  return divide(array(a), b);
}
array operator/(const array& a, double b) {
  return divide(a, array(b));
}

array floor_divide(
    const array& a,
    const array& b,
    StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  if (issubdtype(dtype, inexact)) {
    return floor(divide(a, b, s), s);
  }

  auto inputs = broadcast_arrays({astype(a, dtype, s), astype(b, dtype, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape, dtype, std::make_shared<Divide>(to_stream(s)), std::move(inputs));
}

array remainder(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  auto inputs = broadcast_arrays(
      {astype(a, dtype, s), astype(b, dtype, to_stream(s))}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape,
      dtype,
      std::make_shared<Remainder>(to_stream(s)),
      std::move(inputs));
}
array operator%(const array& a, const array& b) {
  return remainder(a, b);
}

std::vector<array>
divmod(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  if (issubdtype(dtype, complexfloating)) {
    throw std::invalid_argument("[divmod] Complex type not supported.");
  }
  auto inputs = broadcast_arrays(
      {astype(a, dtype, s), astype(b, dtype, to_stream(s))}, s);
  return array::make_arrays(
      {inputs[0].shape(), inputs[0].shape()},
      {inputs[0].dtype(), inputs[0].dtype()},
      std::make_shared<DivMod>(to_stream(s)),
      inputs);
}

array maximum(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto out_type = promote_types(a.dtype(), b.dtype());
  auto inputs =
      broadcast_arrays({astype(a, out_type, s), astype(b, out_type, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape,
      out_type,
      std::make_shared<Maximum>(to_stream(s)),
      std::move(inputs));
}

array minimum(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto out_type = promote_types(a.dtype(), b.dtype());
  auto inputs =
      broadcast_arrays({astype(a, out_type, s), astype(b, out_type, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape,
      out_type,
      std::make_shared<Minimum>(to_stream(s)),
      std::move(inputs));
}

array floor(const array& a, StreamOrDevice s /* = {} */) {
  if (a.dtype() == complex64) {
    throw std::invalid_argument("[floor] Not supported for complex64.");
  }
  return array(
      a.shape(), a.dtype(), std::make_shared<Floor>(to_stream(s)), {a});
}

array ceil(const array& a, StreamOrDevice s /* = {} */) {
  if (a.dtype() == complex64) {
    throw std::invalid_argument("[floor] Not supported for complex64.");
  }
  return array(a.shape(), a.dtype(), std::make_shared<Ceil>(to_stream(s)), {a});
}

array square(const array& a, StreamOrDevice s /* = {} */) {
  return array(
      a.shape(), a.dtype(), std::make_shared<Square>(to_stream(s)), {a});
}

array exp(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(a.shape(), dtype, std::make_shared<Exp>(to_stream(s)), {input});
}

array expm1(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_shared<Expm1>(to_stream(s)), {input});
}

array sin(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(a.shape(), dtype, std::make_shared<Sin>(to_stream(s)), {input});
}

array cos(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(a.shape(), dtype, std::make_shared<Cos>(to_stream(s)), {input});
}

array tan(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(a.shape(), dtype, std::make_shared<Tan>(to_stream(s)), {input});
}

array arcsin(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_shared<ArcSin>(to_stream(s)), {input});
}

array arccos(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_shared<ArcCos>(to_stream(s)), {input});
}

array arctan(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_shared<ArcTan>(to_stream(s)), {input});
}

array arctan2(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(promote_types(a.dtype(), b.dtype()));
  auto inputs = broadcast_arrays({astype(a, dtype, s), astype(b, dtype, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape, dtype, std::make_shared<ArcTan2>(to_stream(s)), std::move(inputs));
}

array sinh(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(a.shape(), dtype, std::make_shared<Sinh>(to_stream(s)), {input});
}

array cosh(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(a.shape(), dtype, std::make_shared<Cosh>(to_stream(s)), {input});
}

array tanh(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(a.shape(), dtype, std::make_shared<Tanh>(to_stream(s)), {input});
}

array arcsinh(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_shared<ArcSinh>(to_stream(s)), {input});
}

array arccosh(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_shared<ArcCosh>(to_stream(s)), {input});
}

array arctanh(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_shared<ArcTanh>(to_stream(s)), {input});
}

array degrees(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  return multiply(a, array(180.0 / M_PI, dtype), s);
}

array radians(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  return multiply(a, array(M_PI / 180.0, dtype), s);
}

array log(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(),
      dtype,
      std::make_shared<Log>(to_stream(s), Log::Base::e),
      {input});
}

array log2(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(),
      dtype,
      std::make_shared<Log>(to_stream(s), Log::Base::two),
      {input});
}

array log10(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(),
      dtype,
      std::make_shared<Log>(to_stream(s), Log::Base::ten),
      {input});
}

array log1p(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_shared<Log1p>(to_stream(s)), {input});
}

array logaddexp(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  // Make sure out type is floating point
  auto out_type = at_least_float(promote_types(a.dtype(), b.dtype()));
  auto inputs =
      broadcast_arrays({astype(a, out_type, s), astype(b, out_type, s)}, s);
  auto& shape = inputs[0].shape();
  return array(
      shape,
      out_type,
      std::make_shared<LogAddExp>(to_stream(s)),
      std::move(inputs));
}

array sigmoid(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_shared<Sigmoid>(to_stream(s)), {input});
}

array erf(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  return array(
      a.shape(),
      dtype,
      std::make_shared<Erf>(to_stream(s)),
      {astype(a, dtype, s)});
}

array erfinv(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  return array(
      a.shape(),
      dtype,
      std::make_shared<ErfInv>(to_stream(s)),
      {astype(a, dtype, s)});
}

array stop_gradient(const array& a, StreamOrDevice s /* = {} */) {
  return array(
      a.shape(), a.dtype(), std::make_shared<StopGradient>(to_stream(s)), {a});
}

array round(const array& a, int decimals, StreamOrDevice s /* = {} */) {
  if (decimals == 0) {
    return array(
        a.shape(), a.dtype(), std::make_shared<Round>(to_stream(s)), {a});
  }

  auto dtype = at_least_float(a.dtype());
  float scale = std::pow(10, decimals);
  auto result = multiply(a, array(scale, dtype), s);
  result = round(result, 0, s);
  result = multiply(result, array(1 / scale, dtype), s);

  return astype(result, a.dtype(), s);
}

array matmul(
    const array& in_a,
    const array& in_b,
    StreamOrDevice s /* = {} */) {
  auto a = in_a;
  auto b = in_b;
  if (a.ndim() == 0 || b.ndim() == 0) {
    throw std::invalid_argument(
        "[matmul] Got 0 dimension input. Inputs must "
        "have at least one dimension.");
  }

  if (a.ndim() == 1) {
    // Insert a singleton dim in the beginning
    a = expand_dims(a, 0, s);
  }
  if (b.ndim() == 1) {
    // Insert a singleton dim at the end
    b = expand_dims(b, 1, s);
  }
  if (a.shape(-1) != b.shape(-2)) {
    std::ostringstream msg;
    msg << "[matmul] Last dimension of first input with shape " << a.shape()
        << " must match second to last dimension of"
        << " second input with shape " << b.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  // Type promotion
  auto out_type = promote_types(a.dtype(), b.dtype());

  if (!issubdtype(out_type, inexact)) {
    std::ostringstream msg;
    msg << "[matmul] Only inexact types are supported but " << a.dtype()
        << " and " << b.dtype() << " were provided which results"
        << " in " << out_type << ", which is not a floating point type.";
    throw std::invalid_argument(msg.str());
  }
  if (a.dtype() != out_type) {
    a = astype(a, out_type, s);
  }
  if (b.dtype() != out_type) {
    b = astype(b, out_type, s);
  }

  // We can batch the multiplication by reshaping a
  if (in_a.ndim() > 2 && in_b.ndim() <= 2) {
    a = flatten(a, 0, -2, s);
  } else if (in_b.ndim() > 2) {
    std::tie(a, b) = broadcast_arrays(a, b, {-2, -1}, s);
  }

  auto out_shape = a.shape();
  out_shape.back() = b.shape(-1);

  auto out = array(
      std::move(out_shape),
      out_type,
      std::make_shared<Matmul>(to_stream(s)),
      {a, b});
  if (in_a.ndim() > 2 && in_b.ndim() <= 2) {
    auto orig_shape = in_a.shape();
    orig_shape.pop_back();
    out = unflatten(out, 0, std::move(orig_shape), s);
  }

  // Remove the possibly inserted singleton dimensions
  std::vector<int> axes;
  if (in_a.ndim() == 1) {
    axes.push_back(out.ndim() - 2);
  }
  if (in_b.ndim() == 1) {
    axes.push_back(out.ndim() - 1);
  }
  return axes.empty() ? out : squeeze(out, axes, s);
}

array gather(
    const array& a,
    const std::vector<array>& indices,
    const std::vector<int>& axes,
    const Shape& slice_sizes,
    StreamOrDevice s /* = {} */) {
  // Checks that indices, dimensions, and slice_sizes are all valid
  if (indices.size() > a.ndim()) {
    std::ostringstream msg;
    msg << "[gather] Too many index arrays. Got " << indices.size()
        << " index arrays for input with " << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  std::set dims(axes.begin(), axes.end());
  if (dims.size() != axes.size()) {
    throw std::invalid_argument("[gather] Repeat axes not allowed in gather.");
  }
  if (!dims.empty() && (*dims.begin() < 0 || *dims.rbegin() >= a.ndim())) {
    throw std::invalid_argument("[gather] Axes don't match array dimensions.");
  }
  if (indices.size() != axes.size()) {
    throw std::invalid_argument(
        "[gather] Number of index arrays does not match number of axes.");
  }
  for (auto& x : indices) {
    if (x.dtype() == bool_) {
      throw std::invalid_argument("[Gather] Boolean indices not supported.");
    }
  }

  if (slice_sizes.size() != a.ndim()) {
    std::ostringstream msg;
    msg << "[gather] Got slice_sizes with size " << slice_sizes.size()
        << " for array with " << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  // Promote indices to the same type
  auto dtype = result_type(indices);
  if (issubdtype(dtype, inexact)) {
    throw std::invalid_argument(
        "[gather] Got indices with invalid dtype. Indices must be integral.");
  }

  // Broadcast and cast indices if necessary
  auto inputs = broadcast_arrays(indices);
  for (auto& idx : inputs) {
    idx = astype(idx, dtype, s);
  }

  if (a.size() == 0) {
    // Empty input, either the total slice size is 0 or the indices are empty
    auto total_slice = std::accumulate(
        slice_sizes.begin(), slice_sizes.end(), 1, std::multiplies<int64_t>{});
    auto idx_size = !inputs.empty() ? inputs[0].size() : 1;
    if (idx_size != 0 && total_slice != 0) {
      std::ostringstream msg;
      msg << "[gather] If the input is empty, either the indices must be"
          << " empty or the total slice size must be 0.";
      throw std::invalid_argument(msg.str());
    }
  } else {
    // Non-empty input, check slice sizes are valid
    for (int i = 0; i < a.ndim(); ++i) {
      if (slice_sizes[i] < 0 || slice_sizes[i] > a.shape(i)) {
        std::ostringstream msg;
        msg << "[gather] Slice sizes must be in [0, a.shape(i)]. Got "
            << slice_sizes << " for array with shape " << a.shape() << ".";
        throw std::invalid_argument(msg.str());
      }
    }
  }

  Shape out_shape;
  if (!inputs.empty()) {
    out_shape = inputs[0].shape();
  }
  out_shape.insert(out_shape.end(), slice_sizes.begin(), slice_sizes.end());

  inputs.insert(inputs.begin(), a);
  return array(
      std::move(out_shape),
      a.dtype(),
      std::make_shared<Gather>(
          to_stream(s), std::move(axes), std::move(slice_sizes)),
      inputs);
}

array kron(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  if (a.size() == 0 || b.size() == 0) {
    throw std::invalid_argument("[kron] Input arrays cannot be empty.");
  }

  int ndim = std::max(a.ndim(), b.ndim());
  Shape a_shape(2 * ndim, 1);
  Shape b_shape(2 * ndim, 1);
  Shape out_shape(ndim, 1);

  for (int i = ndim - 1, j = a.ndim() - 1; j >= 0; j--, i--) {
    a_shape[2 * i] = a.shape(j);
    out_shape[i] *= a.shape(j);
  }
  for (int i = ndim - 1, j = b.ndim() - 1; j >= 0; j--, i--) {
    b_shape[2 * i + 1] = b.shape(j);
    out_shape[i] *= b.shape(j);
  }

  return reshape(
      multiply(
          reshape(a, std::move(a_shape), s),
          reshape(b, std::move(b_shape), s),
          s),
      std::move(out_shape),
      s);
}

array take(
    const array& a,
    const array& indices,
    int axis,
    StreamOrDevice s /* = {} */) {
  // Check for valid axis
  if (axis + static_cast<int>(a.ndim()) < 0 ||
      axis >= static_cast<int>(a.ndim())) {
    std::ostringstream msg;
    msg << "[take] Received invalid axis " << axis << " for array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  // Check for valid take
  if (a.shape(axis) == 0 && indices.size() != 0) {
    throw std::invalid_argument(
        "[take] Cannot do a non-empty take from an empty axis.");
  }

  // Handle negative axis
  axis = axis < 0 ? a.ndim() + axis : axis;

  // Make slice sizes to pass to gather
  Shape slice_sizes = a.shape();
  slice_sizes[axis] = 1;

  auto out = gather(a, indices, axis, slice_sizes, s);

  // Transpose indices dimensions to axis dimension
  if (axis != 0) {
    std::vector<int> t_axes(out.ndim());
    std::iota(t_axes.begin(), t_axes.begin() + axis, indices.ndim());
    std::iota(t_axes.begin() + axis, t_axes.begin() + axis + indices.ndim(), 0);
    std::iota(
        t_axes.begin() + axis + indices.ndim(),
        t_axes.end(),
        indices.ndim() + axis);
    out = transpose(out, t_axes, s);
  }

  // Squeeze the axis we take over
  return squeeze(out, indices.ndim() + axis, s);
}

array take(const array& a, const array& indices, StreamOrDevice s /* = {} */) {
  return take(flatten(a, s), indices, 0, s);
}

array take(const array& a, int index, int axis, StreamOrDevice s /* = {} */) {
  // Check for valid axis
  if (axis + static_cast<int>(a.ndim()) < 0 ||
      axis >= static_cast<int>(a.ndim())) {
    std::ostringstream msg;
    msg << "[take] Received invalid axis " << axis << " for array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  // Check for valid take
  if (a.size() == 0) {
    throw std::invalid_argument(
        "[take] Cannot do a non-empty take from an array with zero elements.");
  }

  // Handle negative axis
  axis = axis < 0 ? a.ndim() + axis : axis;

  Shape starts(a.ndim(), 0);
  Shape stops = a.shape();
  starts[axis] = index;
  stops[axis] = index + 1;
  return squeeze(slice(a, std::move(starts), std::move(stops), s), axis, s);
}

array take(const array& a, int index, StreamOrDevice s /* = {} */) {
  return take(flatten(a, s), index, 0, s);
}

array take_along_axis(
    const array& a,
    const array& indices,
    int axis,
    StreamOrDevice s /* = {} */) {
  if (axis + a.ndim() < 0 || axis >= static_cast<int>(a.ndim())) {
    std::ostringstream msg;
    msg << "[take_along_axis] Received invalid axis for array with " << a.ndim()
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  if (indices.ndim() != a.ndim()) {
    std::ostringstream msg;
    msg << "[take_along_axis] Indices of dimension " << indices.ndim()
        << " does not match array of dimension " << a.ndim() << ".";
    throw std::invalid_argument(msg.str());
  }

  // Allow negative axis
  axis = axis < 0 ? a.ndim() + axis : axis;

  // Broadcast indices and input ignoring the take axis
  auto inputs =
      broadcast_arrays({a, indices}, std::vector<int>{axis - int(a.ndim())}, s);

  auto out_shape = inputs[1].shape();
  return array(
      std::move(out_shape),
      a.dtype(),
      std::make_shared<GatherAxis>(to_stream(s), axis),
      std::move(inputs));
}

array scatter_axis(
    const array& a,
    const array& indices,
    const array& values,
    int axis,
    ScatterAxis::ReduceType mode,
    StreamOrDevice s) {
  std::string prefix =
      (mode == ScatterAxis::None) ? "[put_along_axis]" : "[scatter_add_axis]";
  if (axis + a.ndim() < 0 || axis >= static_cast<int>(a.ndim())) {
    std::ostringstream msg;
    msg << prefix << " Received invalid axis for array with " << a.ndim()
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  if (indices.ndim() != a.ndim()) {
    std::ostringstream msg;
    msg << prefix << " Indices of dimension " << indices.ndim()
        << " does not match array of dimension " << a.ndim() << ".";
    throw std::invalid_argument(msg.str());
  }

  if (a.size() == 0) {
    return a;
  }

  auto upd = astype(values, a.dtype(), s);

  // Squeeze leading singletons out of update
  if (upd.ndim() > indices.ndim()) {
    std::vector<int> sq_ax(upd.ndim() - indices.ndim());
    std::iota(sq_ax.begin(), sq_ax.end(), 0);
    upd = squeeze(upd, sq_ax, s);
  }

  auto inputs = broadcast_arrays({indices, upd}, s);
  inputs.insert(inputs.begin(), a);

  // Allow negative axis
  axis = axis < 0 ? a.ndim() + axis : axis;

  // Broadcast src, indices, values while ignoring the take axis
  inputs = broadcast_arrays(inputs, {axis - int(a.ndim())}, s);

  auto out_shape = inputs[0].shape();
  return array(
      std::move(out_shape),
      a.dtype(),
      std::make_shared<ScatterAxis>(to_stream(s), mode, axis),
      std::move(inputs));
}

array put_along_axis(
    const array& a,
    const array& indices,
    const array& values,
    int axis,
    StreamOrDevice s /* = {} */) {
  return scatter_axis(a, indices, values, axis, ScatterAxis::None, s);
}

array scatter_add_axis(
    const array& a,
    const array& indices,
    const array& values,
    int axis,
    StreamOrDevice s /* = {} */) {
  return scatter_axis(a, indices, values, axis, ScatterAxis::Sum, s);
}

/** Scatter updates to given indices */
array scatter(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    Scatter::ReduceType mode,
    StreamOrDevice s) {
  // Checks that indices, dimensions, and slice_sizes are all valid
  if (indices.size() > a.ndim()) {
    std::ostringstream msg;
    msg << "[scatter] Too many index arrays. Got " << indices.size()
        << " index arrays for input with " << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  for (auto& x : indices) {
    if (x.dtype() == bool_) {
      throw("[scatter] Boolean indices not supported.");
    }
  }

  std::set dims(axes.begin(), axes.end());
  if (dims.size() != axes.size()) {
    throw std::invalid_argument(
        "[scatter] Repeat axes not allowed in scatter.");
  }
  if (!dims.empty() && (*dims.begin() < 0 || *dims.rbegin() >= a.ndim())) {
    throw std::invalid_argument("[scatter] Axes don't match array dimensions.");
  }
  if (indices.size() != axes.size()) {
    throw std::invalid_argument(
        "[scatter] Number of index arrays does not match number of axes.");
  }

  // Broadcast and cast indices if necessary
  auto inputs = broadcast_arrays(indices);

  Shape idx_shape;
  if (!inputs.empty()) {
    idx_shape = inputs[0].shape();
  }

  if (updates.ndim() != (a.ndim() + idx_shape.size())) {
    std::ostringstream msg;
    msg << "[scatter] Updates with " << updates.ndim()
        << " dimensions does not match the sum of the array (" << a.ndim()
        << ") and indices (" << idx_shape.size() << ") dimensions.";
    throw std::invalid_argument(msg.str());
  }
  for (int i = 0; i < idx_shape.size(); ++i) {
    if (updates.shape(i) != idx_shape[i]) {
      std::ostringstream msg;
      msg << "[scatter] Update shape " << updates.shape()
          << " is not valid for broadcasted index shape " << idx_shape << ".";
      throw std::invalid_argument(msg.str());
    }
  }
  for (int i = 0; i < a.ndim(); ++i) {
    auto up_shape = updates.shape(i + idx_shape.size());
    if (up_shape > a.shape(i)) {
      std::ostringstream msg;
      msg << "[scatter] Updates with shape " << updates.shape()
          << " are too large for array with shape " << a.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
  }

  // Promote indices to the same type
  auto dtype = result_type(indices);
  if (issubdtype(dtype, inexact)) {
    throw std::invalid_argument(
        "[scatter] Got indices with invalid dtype. Indices must be integral.");
  }
  for (auto& idx : inputs) {
    idx = astype(idx, dtype, s);
  }

  // TODO, remove when scatter supports 64-bit outputs
  if (to_stream(s).device == Device::gpu && size_of(a.dtype()) == 8) {
    std::ostringstream msg;
    msg << "[scatter] GPU scatter does not yet support " << a.dtype()
        << " for the input or updates.";
    throw std::invalid_argument(msg.str());
  }

  inputs.insert(inputs.begin(), a);
  inputs.push_back(astype(updates, a.dtype(), s));

  return array(
      a.shape(),
      a.dtype(),
      std::make_shared<Scatter>(to_stream(s), mode, axes),
      std::move(inputs));
}

array scatter(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s /*= {}*/) {
  return scatter(a, indices, updates, axes, Scatter::None, s);
}

array scatter_add(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s /*= {}*/) {
  return scatter(a, indices, updates, axes, Scatter::Sum, s);
}

array scatter_prod(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s /*= {}*/) {
  return scatter(a, indices, updates, axes, Scatter::Prod, s);
}

array scatter_max(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s /*= {}*/) {
  return scatter(a, indices, updates, axes, Scatter::Max, s);
}

array scatter_min(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s /*= {}*/) {
  return scatter(a, indices, updates, axes, Scatter::Min, s);
}

array masked_scatter(
    const array& a,
    const array& mask,
    const array& value,
    StreamOrDevice s /* =  {} */) {
  if (mask.dtype() != bool_) {
    throw std::invalid_argument("[masked_scatter] The mask has to be boolean.");
  }

  if (mask.ndim() > a.ndim()) {
    throw std::invalid_argument(
        "[masked_scatter] The mask cannot have more dimensions than the target.");
  }

  int unmasked_dims = a.ndim() - mask.ndim();

  if (value.ndim() > unmasked_dims + 1) {
    std::ostringstream msg;
    msg << "[masked_scatter] Value array shape must be broadcastable with the last "
        << unmasked_dims << " dimensions of the input.";
    throw std::invalid_argument(msg.str());
  }

  // Check if the start of the mask is compatible
  if (!std::equal(
          mask.shape().begin(), mask.shape().end(), a.shape().begin())) {
    std::ostringstream msg;
    msg << "[masked_scatter] The boolean mask should have the same shape as the "
        << "beginning of the indexed array but the mask has shape "
        << mask.shape() << " and the array has shape " << a.shape();
    throw std::invalid_argument(msg.str());
  }

  array expanded_mask = mask;
  array expanded_value = astype(value, a.dtype(), s);

  // Broadcast both the mask with the last unmasked_dims of a
  if (unmasked_dims > 0) {
    auto mask_shape = mask.shape();
    while (mask_shape.size() < a.ndim()) {
      mask_shape.push_back(1);
    }
    expanded_mask = broadcast_to(reshape(mask, mask_shape, s), a.shape(), s);
  }

  // Broadcast the value with the unmasked dims plus one extra dimension of
  // size mask.size(). If that dim is already provided leave it as is.
  if (value.ndim() < unmasked_dims + 1) {
    Shape value_shape(unmasked_dims + 1 - value.ndim(), 1);
    value_shape.insert(
        value_shape.end(), value.shape().begin(), value.shape().end());
    expanded_value = reshape(expanded_value, value_shape, s);

    value_shape[0] = mask.size();
    for (int i = 1; i < unmasked_dims + 1; i++) {
      value_shape[i] = a.shape(i - unmasked_dims - 1);
    }
    expanded_value = broadcast_to(expanded_value, value_shape, s);
  } else if (!std::equal(
                 value.shape().begin() + 1,
                 value.shape().end(),
                 a.shape().end() - unmasked_dims)) {
    auto value_shape = value.shape();
    for (int i = 1; i < unmasked_dims + 1; i++) {
      value_shape[i] = a.shape(i - unmasked_dims - 1);
    }
    expanded_value = broadcast_to(expanded_value, value_shape, s);
  }

  array expanded_a = expand_dims(a, 0, s);
  expanded_mask = expand_dims(expanded_mask, 0, s);
  expanded_value = expand_dims(expanded_value, 0, s);

  return squeeze(
      array(
          expanded_a.shape(),
          expanded_a.dtype(),
          std::make_shared<MaskedScatter>(to_stream(s)),
          {expanded_a, expanded_mask, expanded_value}),
      0,
      s);
}

array sqrt(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  return array(
      a.shape(),
      dtype,
      std::make_shared<Sqrt>(to_stream(s)),
      {astype(a, dtype, s)});
}

array rsqrt(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  return array(
      a.shape(),
      dtype,
      std::make_shared<Sqrt>(to_stream(s), true),
      {astype(a, dtype, s)});
}

array softmax(
    const array& a,
    const std::vector<int>& axes,
    bool precise /* = false */,
    StreamOrDevice s /* = {}*/) {
  if (a.size() == 0) {
    return a;
  }
  if (a.ndim() == 0 && !axes.empty()) {
    throw std::invalid_argument(
        "[softmax] Received non-empty axes for array with 0 dimensions.");
  }
  bool reduce_last_dim =
      !axes.empty() && (axes.back() == a.ndim() - 1 || axes.back() == -1);
  if (reduce_last_dim) {
    // For more than 2 axes check if axes is [0, 1, ..., NDIM - 1] and shape
    // is [1, 1, ..., N].
    for (int i = axes.size() - 2; i >= 0; --i) {
      if ((axes[i] + 1 != axes[i + 1]) || (a.shape(axes[i]) != 1)) {
        reduce_last_dim = false;
        break;
      }
    }
  }
  bool is_complex = issubdtype(a.dtype(), complexfloating);
  if (!is_complex && reduce_last_dim) {
    auto dtype = at_least_float(a.dtype());
    return array(
        a.shape(),
        dtype,
        std::make_shared<Softmax>(to_stream(s), precise),
        {astype(a, dtype, s)});
  } else {
    auto in = a;
    if (precise && !is_complex) {
      in = astype(a, float32, s);
    }
    auto a_max = stop_gradient(max(in, axes, /*keepdims = */ true, s), s);
    auto ex = exp(subtract(in, a_max, s), s);
    return astype(
        divide(ex, sum(ex, axes, /*keepdims = */ true, s), s), a.dtype(), s);
  }
}

array softmax(
    const array& a,
    bool precise /* = false */,
    StreamOrDevice s /* = {}*/) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return softmax(a, axes, precise, s);
}

array power(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  std::vector<array> inputs = {astype(a, dtype, s), astype(b, dtype, s)};
  if (a.shape() != b.shape()) {
    inputs = broadcast_arrays(inputs, s);
  }
  return array(
      inputs[0].shape(), dtype, std::make_shared<Power>(to_stream(s)), inputs);
}

array cumsum(
    const array& a,
    int axis,
    bool reverse /* = false*/,
    bool inclusive /* = true*/,
    StreamOrDevice s /* = {}*/) {
  int ndim = a.ndim();
  if (axis >= ndim || axis < -ndim) {
    std::ostringstream msg;
    msg << "[cumsum] Axis " << axis << " is out of bounds for array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  axis = (axis + a.ndim()) % a.ndim();
  auto out_type = a.dtype() == bool_ ? int32 : a.dtype();
  return array(
      a.shape(),
      out_type,
      std::make_shared<Scan>(
          to_stream(s), Scan::ReduceType::Sum, axis, reverse, inclusive),
      {a});
}

array cumsum(
    const array& a,
    bool reverse /* = false*/,
    bool inclusive /* = true*/,
    StreamOrDevice s /* = {}*/) {
  return cumsum(flatten(a, to_stream(s)), 0, reverse, inclusive, to_stream(s));
}

array cumprod(
    const array& a,
    int axis,
    bool reverse /* = false*/,
    bool inclusive /* = true*/,
    StreamOrDevice s /* = {}*/) {
  int ndim = a.ndim();
  if (axis >= ndim || axis < -ndim) {
    std::ostringstream msg;
    msg << "[cumprod] Axis " << axis << " is out of bounds for array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  axis = (axis + a.ndim()) % a.ndim();
  return array(
      a.shape(),
      a.dtype(),
      std::make_shared<Scan>(
          to_stream(s), Scan::ReduceType::Prod, axis, reverse, inclusive),
      {a});
}

array cumprod(
    const array& a,
    bool reverse /* = false*/,
    bool inclusive /* = true*/,
    StreamOrDevice s /* = {}*/) {
  return cumprod(flatten(a, s), 0, reverse, inclusive, s);
}

array cummax(
    const array& a,
    int axis,
    bool reverse /* = false*/,
    bool inclusive /* = true*/,
    StreamOrDevice s /* = {}*/) {
  int ndim = a.ndim();
  if (axis >= ndim || axis < -ndim) {
    std::ostringstream msg;
    msg << "[cummax] Axis " << axis << " is out of bounds for array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  axis = (axis + a.ndim()) % a.ndim();
  return array(
      a.shape(),
      a.dtype(),
      std::make_shared<Scan>(
          to_stream(s), Scan::ReduceType::Max, axis, reverse, inclusive),
      {a});
}

array cummax(
    const array& a,
    bool reverse /* = false*/,
    bool inclusive /* = true*/,
    StreamOrDevice s /* = {}*/) {
  return cummax(flatten(a, s), 0, reverse, inclusive, s);
}

array cummin(
    const array& a,
    int axis,
    bool reverse /* = false*/,
    bool inclusive /* = true*/,
    StreamOrDevice s /* = {}*/) {
  int ndim = a.ndim();
  if (axis >= ndim || axis < -ndim) {
    std::ostringstream msg;
    msg << "[cummin] Axis " << axis << " is out of bounds for array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  axis = (axis + a.ndim()) % a.ndim();
  return array(
      a.shape(),
      a.dtype(),
      std::make_shared<Scan>(
          to_stream(s), Scan::ReduceType::Min, axis, reverse, inclusive),
      {a});
}

array cummin(
    const array& a,
    bool reverse /* = false*/,
    bool inclusive /* = true*/,
    StreamOrDevice s /* = {}*/) {
  return cummin(flatten(a, s), 0, reverse, inclusive, s);
}

array logcumsumexp(
    const array& a,
    int axis,
    bool reverse /* = false*/,
    bool inclusive /* = true*/,
    StreamOrDevice s /* = {}*/) {
  int ndim = a.ndim();
  if (axis >= ndim || axis < -ndim) {
    std::ostringstream msg;
    msg << "[logcumsumexp] Axis " << axis << " is out of bounds for array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  axis = (axis + a.ndim()) % a.ndim();
  return array(
      a.shape(),
      a.dtype(),
      std::make_shared<Scan>(
          to_stream(s), Scan::ReduceType::LogAddExp, axis, reverse, inclusive),
      {a});
}

array logcumsumexp(
    const array& a,
    bool reverse /* = false*/,
    bool inclusive /* = true*/,
    StreamOrDevice s /* = {}*/) {
  return logcumsumexp(
      flatten(a, to_stream(s)), 0, reverse, inclusive, to_stream(s));
}

/** Convolution operations */

namespace {

inline void
run_conv_checks(const array& in, const array& wt, int n_dim, int groups) {
  if (!issubdtype(in.dtype(), floating)) {
    std::ostringstream msg;
    msg << "[conv] Invalid input array with type " << in.dtype() << "."
        << " Convolution currently only supports floating point types";
    throw std::invalid_argument(msg.str());
  }

  if (in.ndim() != n_dim + 2) {
    std::ostringstream msg;
    msg << "[conv] Invalid input array with " << in.ndim() << " dimensions for "
        << n_dim << "D convolution. Expected an array with " << n_dim + 2
        << " dimensions following the format [N, ..., C_in].";
    throw std::invalid_argument(msg.str());
  }

  if (wt.ndim() != n_dim + 2) {
    std::ostringstream msg;
    msg << "[conv] Invalid weight array with " << wt.ndim()
        << " dimensions for " << n_dim << "D convolution."
        << " Expected an array with " << n_dim + 2
        << " dimensions following the format [C_out, ..., C_in].";
    throw std::invalid_argument(msg.str());
  }

  if (in.shape(n_dim + 1) % groups != 0) {
    std::ostringstream msg;
    msg << "[conv] The input channels must be divisible by the number"
        << " of groups. Got input with shape " << in.shape() << " and "
        << groups << " groups.";
    throw std::invalid_argument(msg.str());
  }

  if (groups > 1 && wt.shape(0) % groups != 0) {
    std::ostringstream msg;
    msg << "[conv] If groups > 1, the output channels must be divisible by the number"
        << " of groups. Got " << wt.shape(0) << " output channels and "
        << groups << " groups.";
    throw std::invalid_argument(msg.str());
  }

  if (in.shape(n_dim + 1) != (groups * wt.shape(n_dim + 1))) {
    std::ostringstream msg;
    if (groups == 1) {
      msg << "[conv] Expect the input channels in the input"
          << " and weight array to match but got shapes -"
          << " input: " << in.shape() << " and weight: " << wt.shape();

    } else {
      msg << "Given groups=" << groups << " and weights of shape " << wt.shape()
          << ", expected to have " << (groups * wt.shape(n_dim + 1))
          << " input channels but got " << in.shape(n_dim + 1)
          << " input channels instead.";
    }
    throw std::invalid_argument(msg.str());
  }
}

} // namespace

/** 1D convolution with a filter */
array conv1d(
    const array& in_,
    const array& wt_,
    int stride /* = 1 */,
    int padding /* = 0 */,
    int dilation /* = 1 */,
    int groups /* = 1 */,
    StreamOrDevice s /* = {} */) {
  return conv_general(
      /* const array& input = */ in_,
      /* const array& weight = */ wt_,
      /* std::vector<int> stride = */ {stride},
      /* std::vector<int> padding = */ {padding},
      /* std::vector<int> kernel_dilation = */ {dilation},
      /* std::vector<int> input_dilation = */ {1},
      /* int groups = */ groups,
      /* bool flip = */ false,
      s);
}

/** 2D convolution with a filter */
array conv2d(
    const array& in_,
    const array& wt_,
    const std::pair<int, int>& stride /* = {1, 1} */,
    const std::pair<int, int>& padding /* = {0, 0} */,
    const std::pair<int, int>& dilation /* = {1, 1} */,
    int groups /* = 1 */,
    StreamOrDevice s /* = {} */) {
  return conv_general(
      /* const array& input = */ in_,
      /* const array& weight = */ wt_,
      /* std::vector<int> stride = */ {stride.first, stride.second},
      /* std::vector<int> padding = */ {padding.first, padding.second},
      /* std::vector<int> kernel_dilation = */
      {dilation.first, dilation.second},
      /* std::vector<int> input_dilation = */ {1, 1},
      /* int groups = */ groups,
      /* bool flip = */ false,
      s);
}

/** 3D convolution with a filter */
array conv3d(
    const array& in_,
    const array& wt_,
    const std::tuple<int, int, int>& stride /* = {1, 1, 1} */,
    const std::tuple<int, int, int>& padding /* = {0, 0, 0} */,
    const std::tuple<int, int, int>& dilation /* = {1, 1, 1} */,
    int groups /* = 1 */,
    StreamOrDevice s /* = {} */) {
  return conv_general(
      /* const array& input = */ in_,
      /* const array& weight = */ wt_,
      /* std::vector<int> stride = */
      {std::get<0>(stride), std::get<1>(stride), std::get<2>(stride)},
      /* std::vector<int> padding = */
      {std::get<0>(padding), std::get<1>(padding), std::get<2>(padding)},
      /* std::vector<int> kernel_dilation = */
      {std::get<0>(dilation), std::get<1>(dilation), std::get<2>(dilation)},
      /* std::vector<int> input_dilation = */ {1, 1, 1},
      /* int groups = */ groups,
      /* bool flip = */ false,
      s);
}

// Helper function for transposed convolutions
array conv_transpose_general(
    const array& input,
    const array& weight,
    std::vector<int> stride,
    std::vector<int> padding,
    std::vector<int> dilation,
    std::vector<int> output_padding,
    int groups,
    StreamOrDevice s) {
  std::vector<int> padding_lo(padding.size());
  std::vector<int> padding_hi(padding.size());
  for (int i = 0; i < padding.size(); ++i) {
    int wt_size = 1 + dilation[i] * (weight.shape(1 + i) - 1);
    padding_lo[i] = wt_size - padding[i] - 1;

    int conv_output_shape = (input.shape(i + 1) - 1) * stride[i] -
        2 * padding[i] + dilation[i] * (weight.shape(i + 1) - 1) + 1;

    int in_size = 1 + (conv_output_shape - 1);
    int out_size = 1 + stride[i] * (input.shape(1 + i) - 1);
    padding_hi[i] = in_size - out_size + padding[i] +
        output_padding[i]; // Adjust with output_padding
  }

  return conv_general(
      /* const array& input = */ input,
      /* const array& weight = */ weight,
      /* std::vector<int> stride = */ std::vector(stride.size(), 1),
      /* std::vector<int> padding_lo = */ std::move(padding_lo),
      /* std::vector<int> padding_hi = */ std::move(padding_hi),
      /* std::vector<int> kernel_dilation = */ std::move(dilation),
      /* std::vector<int> input_dilation = */ std::move(stride),
      /* int groups = */ groups,
      /* bool flip = */ true,
      s);
}

/** 1D transposed convolution with a filter */
array conv_transpose1d(
    const array& in_,
    const array& wt_,
    int stride /* = 1 */,
    int padding /* = 0 */,
    int dilation /* = 1 */,
    int output_padding /* = 0 */,
    int groups /* = 1 */,
    StreamOrDevice s /* = {} */) {
  return conv_transpose_general(
      in_, wt_, {stride}, {padding}, {dilation}, {output_padding}, groups, s);
}

/** 2D transposed convolution with a filter */
array conv_transpose2d(
    const array& in_,
    const array& wt_,
    const std::pair<int, int>& stride /* = {1, 1} */,
    const std::pair<int, int>& padding /* = {0, 0} */,
    const std::pair<int, int>& dilation /* = {1, 1} */,
    const std::pair<int, int>& output_padding /* = {0, 0} */,
    int groups /* = 1 */,
    StreamOrDevice s /* = {} */) {
  return conv_transpose_general(
      in_,
      wt_,
      {stride.first, stride.second},
      {padding.first, padding.second},
      {dilation.first, dilation.second},
      {output_padding.first, output_padding.second},
      groups,
      s);
}

/** 3D transposed convolution with a filter */
array conv_transpose3d(
    const array& in_,
    const array& wt_,
    const std::tuple<int, int, int>& stride /* = {1, 1, 1} */,
    const std::tuple<int, int, int>& padding /* = {0, 0, 0} */,
    const std::tuple<int, int, int>& dilation /* = {1, 1, 1} */,
    const std::tuple<int, int, int>& output_padding /* = {0, 0, 0} */,
    int groups /* = 1 */,
    StreamOrDevice s /* = {} */) {
  return conv_transpose_general(
      in_,
      wt_,
      {std::get<0>(stride), std::get<1>(stride), std::get<2>(stride)},
      {std::get<0>(padding), std::get<1>(padding), std::get<2>(padding)},
      {std::get<0>(dilation), std::get<1>(dilation), std::get<2>(dilation)},
      {std::get<0>(output_padding),
       std::get<1>(output_padding),
       std::get<2>(output_padding)},
      groups,
      s);
}

/** General convolution with a filter */
array conv_general(
    array in,
    array wt,
    std::vector<int> stride /* = {} */,
    std::vector<int> padding_lo /* = {} */,
    std::vector<int> padding_hi /* = {} */,
    std::vector<int> kernel_dilation /* = {} */,
    std::vector<int> input_dilation /* = {} */,
    int groups /* = 1 */,
    bool flip /* = false */,
    StreamOrDevice s /* = {} */) {
  // Run checks
  if (groups != 1 && in.ndim() != 3 && in.ndim() != 4) {
    throw std::invalid_argument(
        "[conv] Can only handle groups != 1 in 1D or 2D convolutions.");
  }

  int spatial_dims = in.ndim() - 2;

  if (spatial_dims < 1 || spatial_dims > 3) {
    throw std::invalid_argument(
        "[conv] Only works for inputs with 1-3 spatial dimensions."
        " The inputs must be in the format [N, ..., C_in]");
  }

  // Run checks
  run_conv_checks(in, wt, spatial_dims, groups);

  // Type promotion
  auto out_type = promote_types(in.dtype(), wt.dtype());
  in = astype(in, out_type, s);
  wt = astype(wt, out_type, s);

  if (stride.size() <= 1) {
    int stride_int = stride.size() ? stride[0] : 1;
    stride = std::vector<int>(spatial_dims, stride_int);
  }

  if (padding_lo.size() <= 1) {
    int padding_int = padding_lo.size() ? padding_lo[0] : 0;
    padding_lo = std::vector<int>(spatial_dims, padding_int);
  }

  if (padding_hi.size() <= 1) {
    int padding_int = padding_hi.size() ? padding_hi[0] : 0;
    padding_hi = std::vector<int>(spatial_dims, padding_int);
  }

  if (kernel_dilation.size() <= 1) {
    int kernel_dilation_int = kernel_dilation.size() ? kernel_dilation[0] : 1;
    kernel_dilation = std::vector<int>(spatial_dims, kernel_dilation_int);
  }

  if (input_dilation.size() <= 1) {
    int input_dilation_int = input_dilation.size() ? input_dilation[0] : 1;
    input_dilation = std::vector<int>(spatial_dims, input_dilation_int);
  }

  // Check for negative padding
  bool has_neg_padding = false;
  for (auto& pd : padding_lo) {
    has_neg_padding |= (pd < 0);
  }
  for (auto& pd : padding_hi) {
    has_neg_padding |= (pd < 0);
  }

  // Handle negative padding
  if (has_neg_padding) {
    Shape starts(in.ndim(), 0);
    auto stops = in.shape();

    for (int i = 0; i < spatial_dims; i++) {
      if (padding_lo[i] < 0) {
        starts[i + 1] -= padding_lo[i];
        padding_lo[i] = 0;
      }

      if (padding_hi[i] < 0) {
        stops[i + 1] += padding_hi[i];
        padding_hi[i] = 0;
      }
    }

    in = slice(in, std::move(starts), std::move(stops), s);
  }

  // Get output shapes
  auto out_shape = Convolution::conv_out_shape(
      in.shape(),
      wt.shape(),
      stride,
      padding_lo,
      padding_hi,
      kernel_dilation,
      input_dilation);

  return array(
      std::move(out_shape),
      in.dtype(),
      std::make_shared<Convolution>(
          to_stream(s),
          stride,
          padding_lo,
          padding_hi,
          kernel_dilation,
          input_dilation,
          groups,
          flip),
      {in, wt});
}

std::pair<int, int> quantization_params_from_mode(
    QuantizationMode mode,
    std::optional<int> group_size_,
    std::optional<int> bits_) {
  int default_group_size;
  int default_bits;
  switch (mode) {
    case QuantizationMode::Affine:
      default_group_size = 64;
      default_bits = 4;
      break;
    case QuantizationMode::Nvfp4:
      default_group_size = 16;
      default_bits = 4;
      break;
    case QuantizationMode::Mxfp4:
      default_group_size = 32;
      default_bits = 4;
      break;
    case QuantizationMode::Mxfp8:
      default_group_size = 32;
      default_bits = 8;
      break;
  }
  return {
      group_size_.has_value() ? *group_size_ : default_group_size,
      bits_.has_value() ? *bits_ : default_bits};
}

std::pair<Dtype, QuantizationMode> validate_mode_with_type(
    std::string_view tag,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<Dtype> out_type,
    const std::string& mode) {
  auto qmode = string_to_quantization_mode(mode, tag);
  if (out_type.has_value() && !issubdtype(*out_type, floating)) {
    std::ostringstream msg;
    msg << "[" << tag << "] Only real floating types are supported but "
        << "output dtype == " << *out_type << ".";
    throw std::invalid_argument(msg.str());
  }

  if (qmode == QuantizationMode::Affine) {
    if (!biases) {
      std::ostringstream msg;
      msg << "[" << tag << "] Biases must be provided for affine quantization.";
      throw std::invalid_argument(msg.str());
    }
    auto dtype = result_type(scales, *biases);
    if (!issubdtype(dtype, floating)) {
      std::ostringstream msg;
      msg << "[" << tag << "] Only real floating types are supported but "
          << "scales.dtype() == " << scales.dtype()
          << " and biases.dtype() == " << biases->dtype() << ".";
      throw std::invalid_argument(msg.str());
    }
    if (out_type.has_value()) {
      return {*out_type, qmode};
    } else {
      return {dtype, qmode};
    }
  } else if (scales.dtype() != uint8) {
    std::ostringstream msg;
    msg << "[" << tag << "] Scale type must be uint8 but received type "
        << scales.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (biases) {
    std::ostringstream msg;
    msg << "[" << tag << "] Biases must be null for quantization mode '" << mode
        << "'.";
    throw std::invalid_argument(msg.str());
  }
  if (out_type.has_value()) {
    return {*out_type, qmode};
  } else {
    return {bfloat16, qmode};
  }
}

void validate_global_scale(
    std::string_view tag,
    QuantizationMode qmode,
    const std::optional<array>& global_scale) {
  if (global_scale.has_value()) {
    if (qmode != QuantizationMode::Nvfp4) {
      std::ostringstream msg;
      msg << "[" << tag << "] Global scale is only supported for 'nvfp4' "
          << "quantization mode.";
      throw std::invalid_argument(msg.str());
    } else {
      if (global_scale->size() != 1) {
        std::ostringstream msg;
        msg << "[" << tag << "] Global scale must be a scalar but got shape "
            << global_scale->shape() << ".";
        throw std::invalid_argument(msg.str());
      }
      // TODO: not sure if type should be restricted to float32
      if (global_scale->dtype() != float32) {
        std::ostringstream msg;
        msg << "[" << tag << "] Global scale must have dtype float32 but got "
            << global_scale->dtype() << ".";
        throw std::invalid_argument(msg.str());
      }
    }
  }
}

array quantized_matmul(
    array x,
    array w,
    array scales,
    std::optional<array> biases /* = std::nullopt */,
    bool transpose /* = true */,
    std::optional<int> group_size_ /* = std::nullopt */,
    std::optional<int> bits_ /* = std::nullopt */,
    const std::string& mode /* = "affine" */,
    StreamOrDevice s /* = {} */) {
  auto [dtype, qmode] = validate_mode_with_type(
      "quantized_matmul", scales, biases, std::nullopt, mode);

  auto [group_size, bits] =
      quantization_params_from_mode(qmode, group_size_, bits_);
  // Check and extract the quantized matrix shape against x
  auto [w_inner_dims, w_outer_dims] = extract_quantized_matmul_dims(
      "quantized_matmul", x, w, scales, biases, transpose, group_size, bits);

  if (qmode == QuantizationMode::Affine) {
    dtype = promote_types(x.dtype(), dtype);
  } else {
    dtype = x.dtype();
  }

  if (!issubdtype(dtype, floating)) {
    std::ostringstream msg;
    msg << "[quantized_matmul] Only real floating types are supported but "
        << "x.dtype() == " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  std::vector<array> inputs;
  if (qmode == QuantizationMode::Affine) {
    inputs = {
        astype(x, dtype), w, astype(scales, dtype), astype(*biases, dtype)};
  } else {
    inputs = {x, w, scales};
  }

  if (x.ndim() > 2 && w.ndim() > 2) {
    inputs = broadcast_arrays(inputs, {-2, -1}, s);
  }
  auto out_shape = inputs[0].shape();
  out_shape.back() = w_outer_dims;
  return array(
      std::move(out_shape),
      dtype,
      std::make_shared<QuantizedMatmul>(
          to_stream(s), group_size, bits, qmode, transpose),
      std::move(inputs));
}

void validate_qqmm_inputs(
    array x,
    array w,
    std::optional<array> scales_w,
    int group_size,
    int bits,
    std::optional<array> global_scale_x,
    std::optional<array> global_scale_w,
    QuantizationMode qmode) {
  // check 2D (for now)
  if (x.ndim() > 2 || w.ndim() > 2) {
    std::ostringstream msg;
    msg << "[qqmm] Only 2D inputs are supported but "
        << "x.ndim() == " << x.ndim() << " and "
        << "w.ndim() == " << w.ndim() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (w.dtype() == uint32) {
    // if w is quantized, scales are provided
    if (!scales_w.has_value()) {
      std::ostringstream msg;
      throw std::invalid_argument(
          "[qqmm] Scales must be provided if second argument is quantized.");
    }
    // if scales are provided, check compatibility with quantized w
    else {
      validate_quantized_input("qqmm", w, *scales_w, group_size, bits);
    }
  }
  // if w is not quantized, dtype must be in {f16, bf16, fp32}
  else {
    if (!issubdtype(w.dtype(), floating) || w.dtype() == float64) {
      std::ostringstream msg;
      msg << "[qqmm] Only real floating types except float64 are supported but "
          << "second argument dtype == " << w.dtype() << ".";
      throw std::invalid_argument(msg.str());
    }
  }
  // x dtype must be in {f16, bf16, fp32}
  if (!issubdtype(x.dtype(), floating) || x.dtype() == float64) {
    std::ostringstream msg;
    msg << "[qqmm] Only real floating types except float64 are supported but "
        << "first argument dtype == " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  // validate global scales
  validate_global_scale("qqmm", qmode, global_scale_x);
  validate_global_scale("qqmm", qmode, global_scale_w);
  // For nvfp4 mode, both global scales must be provided together or neither
  if (qmode == QuantizationMode::Nvfp4) {
    bool has_x = global_scale_x.has_value();
    bool has_w = global_scale_w.has_value();
    if (has_x != has_w) {
      throw std::invalid_argument(
          "[qqmm] For nvfp4 mode, either both global_scale_x and "
          "global_scale_w must be provided, or neither.");
    }
  }
}

std::pair<int, int> extract_qqmm_dims(
    array x,
    array w,
    std::optional<array> scales_w,
    int group_size,
    int bits) {
  if (w.dtype() != uint32) {
    // if w is not quantized, check that last dims match
    if (x.shape(-1) != w.shape(-1)) {
      std::ostringstream msg;
      msg << "[qqmm] Last dimension of first input with shape " << x.shape()
          << " must match last dimension of"
          << " second input with shape " << w.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
    return std::make_pair(w.shape(-1), w.shape(-2));
  } else {
    // if w is quantized, extract dims from quantized w
    return extract_quantized_matmul_dims(
        "qqmm",
        x,
        w,
        *scales_w,
        std::nullopt,
        /* transpose = */ true,
        group_size,
        bits);
  }
}

array qqmm(
    array in_x,
    array w,
    std::optional<array> scales_w,
    std::optional<int> group_size_ /* = std::nullopt */,
    std::optional<int> bits_ /* = std::nullopt */,
    const std::string& mode /* = "nvfp4" */,
    const std::optional<array> global_scale_x /* = std::nullopt */,
    const std::optional<array> global_scale_w /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto stream = to_stream(s);
  auto qmode = string_to_quantization_mode(mode, "qqmm");
  // cuBLAS block scaled matmul only supports nvfp4 and mxfp8
  if (qmode != QuantizationMode::Nvfp4 && qmode != QuantizationMode::Mxfp8) {
    std::ostringstream msg;
    msg << "[qqmm] Only 'nvfp4' and 'mxfp8' quantization modes are supported but '"
        << mode << "' was provided.";
    throw std::invalid_argument(msg.str());
  }
  // we need to check 2 cases:
  // 1. w is quantized, scales is provided
  // 2. w is not quantized, scales is not provided
  auto [group_size, bits] =
      quantization_params_from_mode(qmode, group_size_, bits_);

  // Allow gemv
  auto x = in_x;
  if (x.ndim() == 1) {
    // Insert a singleton dim in the beginning
    x = expand_dims(x, 0, s);
  } else if (w.ndim() == 2 && x.ndim() > 2) {
    x = flatten(x, 0, -2, s);
  }

  // validate inputs
  validate_qqmm_inputs(
      x, w, scales_w, group_size, bits, global_scale_x, global_scale_w, qmode);
  // validate and extract shapes
  auto [w_inner_dims, w_outer_dims] =
      extract_qqmm_dims(x, w, scales_w, group_size, bits);
  std::vector<array> inputs = {
      x,
      w,
  };
  if (scales_w.has_value()) {
    inputs.push_back(*scales_w);
  }
  if (global_scale_x.has_value() && global_scale_w.has_value()) {
    inputs.push_back(*global_scale_x);
    inputs.push_back(*global_scale_w);
  }

  auto out_shape = inputs[0].shape();
  out_shape.back() = w_outer_dims;
  auto out = array(
      std::move(out_shape),
      x.dtype(), // output dtype is the same as x dtype
      std::make_shared<QQMatmul>(stream, group_size, bits, qmode),
      std::move(inputs));
  if (in_x.ndim() > 2) {
    auto orig_shape = in_x.shape();
    orig_shape.pop_back();
    out = unflatten(out, 0, std::move(orig_shape), s);
  } else if (in_x.ndim() == 1) {
    out = squeeze(out, 0, s);
  }
  return out;
}

array pack_and_quantize(
    array& packed_w,
    const array& scales,
    const array& biases,
    int bits,
    const Stream& s) {
  int el_per_int = 32 / bits;
  array zero(0, packed_w.dtype());
  array n_bins((1 << bits) - 1, packed_w.dtype()); // 2**bits - 1
  packed_w = astype(
      clip(
          round(divide(subtract(packed_w, biases, s), scales, s), s),
          zero,
          n_bins,
          s),
      uint32,
      s);
  if (is_power_of_2(bits)) {
    array shifts = power(array(2, uint32), arange(0, 32, bits, uint32, s), s);
    packed_w = reshape(packed_w, {packed_w.shape(0), -1, el_per_int}, s);
    packed_w =
        sum(multiply(packed_w, shifts, s),
            /* axis= */ 2,
            /* keepdims= */ false,
            s);
  } else {
    // This is slow but we have fast GPU/CPU versions of this function so we
    // shouldn't be here often.
    packed_w = expand_dims(packed_w, /* axis= */ -1, s);
    packed_w = bitwise_and(
        right_shift(packed_w, arange(bits, uint32, s), s),
        array({1}, uint32),
        s);
    auto new_shape = packed_w.shape();
    new_shape[new_shape.size() - 2] = -1;
    new_shape.back() = 32;
    packed_w = reshape(packed_w, new_shape, s);
    array shifts = arange(32, uint32, s);
    packed_w =
        sum(left_shift(packed_w, shifts, s),
            /* axis= */ -1,
            /* keepdims= */ false,
            s);
  }
  return packed_w;
}

std::vector<array>
affine_quantize(const array& w, int group_size, int bits, StreamOrDevice s_) {
  auto s = to_stream(s_);
  if (group_size != 32 && group_size != 64 && group_size != 128) {
    std::ostringstream msg;
    msg << "[quantize] The requested group size " << group_size
        << " is not supported. The supported group sizes are 32, 64, and 128.";
    throw std::invalid_argument(msg.str());
  }

  if (bits < 1 || bits > 8 || bits == 7) {
    std::ostringstream msg;
    msg << "[quantize] The requested number of bits " << bits
        << " is not supported. The supported bits are 1, 2, 3, 4, 5, 6 and 8.";
    throw std::invalid_argument(msg.str());
  }

  auto fallback = [group_size, bits, s](
                      const std::vector<array>& inputs) -> std::vector<array> {
    auto& w = inputs[0];
    auto wshape = w.shape();
    wshape.back() = -1;

    array zero(0, float32);
    array n_bins((1 << bits) - 1, float32); // 2**bits - 1
    array eps(1e-7, float32);

    array packed_w = reshape(w, {-1, w.shape(-1) / group_size, group_size}, s);

    array w_max = max(packed_w, /* axis= */ -1, /* keepdims= */ true, s);
    array w_min = min(packed_w, /* axis= */ -1, /* keepdims= */ true, s);
    w_max = astype(w_max, float32, s);
    w_min = astype(w_min, float32, s);

    array scales(0, float32);
    array biases(0, float32);

    if (bits == 1) {
      // Affine 1-bit: bit 0 -> w_min, bit 1 -> w_max
      scales = maximum(subtract(w_max, w_min, s), eps, s);
      biases = w_min;
    } else {
      array mask = greater(abs(w_min, s), abs(w_max, s), s);
      scales = maximum(divide(subtract(w_max, w_min, s), n_bins, s), eps, s);
      scales = where(mask, scales, negative(scales, s), s);
      array edge = where(mask, w_min, w_max, s);
      array q0 = round(divide(edge, scales, s), s);
      scales = where(not_equal(q0, zero, s), divide(edge, q0, s), scales);
      biases = where(equal(q0, zero, s), zero, edge, s);
    }

    packed_w = pack_and_quantize(packed_w, scales, biases, bits, s);

    scales = astype(scales, w.dtype(), s);
    biases = astype(biases, w.dtype(), s);
    return {
        reshape(packed_w, wshape, s),
        reshape(scales, wshape, s),
        reshape(biases, wshape, s),
    };
  };

  auto wq_shape = w.shape();
  wq_shape.back() = w.shape(-1) * bits / 32;
  auto sshape = w.shape();
  sshape.back() = w.shape(-1) / group_size;
  return array::make_arrays(
      {std::move(wq_shape), sshape, sshape},
      {uint32, w.dtype(), w.dtype()},
      std::make_shared<fast::Quantize>(
          s, fallback, group_size, bits, QuantizationMode::Affine, false),
      {w});
}

std::vector<array> fp_quantize(
    const array& w,
    int group_size,
    int bits,
    QuantizationMode mode,
    const std::optional<array>& global_scale /* = std::nullopt */,
    Stream s) {
  int expected_gs = mode == QuantizationMode::Nvfp4 ? 16 : 32;
  int expected_bits = mode == QuantizationMode::Mxfp8 ? 8 : 4;
  if (group_size != expected_gs) {
    std::ostringstream msg;
    msg << "[quantize] " << quantization_mode_to_string(mode)
        << " quantization requires group size " << expected_gs << " but got "
        << group_size << ".";
    throw std::invalid_argument(msg.str());
  }
  if (bits != expected_bits) {
    std::ostringstream msg;
    msg << "[quantize] " << quantization_mode_to_string(mode)
        << " quantization requires bits to be " << expected_bits << " but got "
        << bits << ".";
    throw std::invalid_argument(msg.str());
  }

  auto inputs = std::vector<array>{w};
  if (global_scale.has_value()) {
    inputs.push_back(global_scale.value());
  }

  auto fallback = [bits = bits, group_size = group_size, s](
                      const std::vector<array>& inputs) -> std::vector<array> {
    auto& w = inputs[0];
    float maxval = (bits == 4) ? 6.0f : 448.0f;
    auto new_shape = w.shape();
    new_shape.back() = -1;
    auto wq = reshape(w, {-1, group_size}, s);
    auto scales =
        divide(max(abs(wq, s), -1, true, s), array(maxval, w.dtype()), s);
    if (group_size == 16) {
      // convert to e4m3
      auto scale_encode = inputs.size() > 1
          ? divide(array(448.0f * 6.0f, float32), inputs[1], s)
          : array(1.0f, float32);
      scales = multiply(scales, scale_encode, s);
      scales = to_fp8(scales, s);
      wq = multiply(
          divide(wq, from_fp8(scales, w.dtype(), s), s), scale_encode, s);
    } else {
      // convert to e8m0
      auto z = array(0, scales.dtype());
      scales = where(
          equal(scales, z, s),
          z,
          astype(round(log2(scales, s), s), int32, s),
          s);

      wq = divide(wq, power(array(2.0f, w.dtype()), scales, s), s);
      scales = astype(add(scales, array(127, int32), s), uint8, s);
    }
    if (bits == 4) {
      auto lut = array({
          +0.0f,
          +0.5f,
          +1.0f,
          +1.5f,
          +2.0f,
          +3.0f,
          +4.0f,
          +6.0f,
          -0.0f,
          -0.5f,
          -1.0f,
          -1.5f,
          -2.0f,
          -3.0f,
          -4.0f,
          -6.0f,
      });
      lut = astype(lut, w.dtype(), s);
      wq = argmin(
          abs(subtract(expand_dims(wq, -1, s), lut, s), s), -1, false, s);
      auto shifts = power(array(2, uint32), arange(0, 32, 4, uint32, s), s);
      wq = reshape(wq, {-1, 4, 8}, s);
      wq = sum(multiply(wq, shifts, s), -1, false, s);
    } else {
      wq = view(to_fp8(wq, s), uint32, s);
    }
    wq = reshape(wq, new_shape, s);
    scales = reshape(scales, new_shape, s);
    return {std::move(wq), std::move(scales)};
  };

  if (s.device == Device::gpu) {
    auto wq_shape = w.shape();
    wq_shape.back() = w.shape(-1) * bits / 32;
    auto sshape = w.shape();
    sshape.back() = w.shape(-1) / group_size;
    return array::make_arrays(
        {std::move(wq_shape), std::move(sshape)},
        {uint32, uint8},
        std::make_shared<fast::Quantize>(
            s, fallback, group_size, bits, mode, false),
        inputs);
  }
  return fallback(inputs);
}

std::vector<array> quantize(
    const array& w,
    std::optional<int> group_size_ /* = std::nullopt */,
    std::optional<int> bits_ /* = std::nullopt */,
    const std::string& mode /* = "affine" */,
    const std::optional<array>& global_scale /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto qmode = string_to_quantization_mode(mode, "quantize");
  auto [group_size, bits] =
      quantization_params_from_mode(qmode, group_size_, bits_);
  if (!issubdtype(w.dtype(), floating)) {
    std::ostringstream msg;
    msg << "[quantize] Only real floating types can be quantized "
        << "but w has type " << w.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  if (w.ndim() < 2) {
    std::ostringstream msg;
    msg << "[quantize] The matrix to be quantized must have at least 2 dimension "
        << "but it has only " << w.ndim() << ".";
    throw std::invalid_argument(msg.str());
  }

  if ((w.shape(-1) % group_size) != 0) {
    std::ostringstream msg;
    msg << "[quantize] The last dimension of the matrix needs to be divisible by "
        << "the quantization group size " << group_size
        << ". However the provided "
        << " matrix has shape " << w.shape();
    throw std::invalid_argument(msg.str());
  }
  if (to_stream(s).device == Device::gpu && metal::is_available() &&
      global_scale.has_value()) {
    std::ostringstream msg;
    msg << "[quantize] Global scale is not supported on the Metal backend.";
    throw std::invalid_argument(msg.str());
  }
  validate_global_scale("quantize", qmode, global_scale);
  if (qmode == QuantizationMode::Affine) {
    return affine_quantize(w, group_size, bits, s);
  } else {
    return fp_quantize(w, group_size, bits, qmode, global_scale, to_stream(s));
  }
}

array affine_dequantize(
    const array& w,
    const array& scales,
    const array& biases,
    int group_size,
    int bits,
    StreamOrDevice s_) {
  auto wshape = w.shape();
  auto sshape = scales.shape();
  auto bshape = biases.shape();
  if (wshape.size() != sshape.size() || wshape.size() != bshape.size()) {
    throw std::invalid_argument(
        "[dequantize] Shape of scales and biases does not match the matrix");
  }
  wshape.back() = -1;
  sshape.back() = -1;
  bshape.back() = -1;

  if (wshape != sshape || wshape != bshape) {
    throw std::invalid_argument(
        "[dequantize] Shape of scales and biases does not match the matrix");
  }

  // Packing into uint32
  int out_size = w.shape(-1) * 32 / bits;

  if (out_size != scales.shape(-1) * group_size) {
    std::ostringstream msg;
    msg << "[dequantize] Shape of scales and biases does not match the matrix "
        << "given the quantization parameters. Provided matrix of shape "
        << w.shape() << " and scales/biases of shape " << scales.shape()
        << " with group_size=" << group_size << " and bits=" << bits << ".";
    throw std::invalid_argument(msg.str());
  }

  auto s = to_stream(s_);

  auto fallback =
      [wshape = std::move(wshape),
       sshape = std::move(sshape),
       group_size,
       bits,
       s](const std::vector<array>& inputs) mutable -> std::vector<array> {
    auto w = inputs[0];
    auto& scales = inputs[1];
    auto& biases = inputs[2];
    if (is_power_of_2(bits)) {
      std::vector<array> parts;
      for (int start = 0; start < 32; start += bits) {
        parts.push_back(expand_dims(
            right_shift(
                left_shift(w, array(32 - (start + bits), uint32), s),
                array(32 - bits, uint32),
                s),
            -1,
            s));
      }
      w = concatenate(parts, -1, s);
    } else {
      w = expand_dims(w, /* axis= */ -1, s);
      w = bitwise_and(
          right_shift(w, arange(32, uint32, s), s), array({1}, uint32), s);
      auto new_shape = w.shape();
      new_shape[new_shape.size() - 2] = -1;
      new_shape.back() = bits;
      w = reshape(w, new_shape, s);
      array shifts = arange(bits, uint32, s);
      w = sum(
          left_shift(w, shifts, s), /* axis= */ -1, /* keepdims= */ false, s);
    }

    // Dequantize
    wshape.push_back(group_size);
    w = reshape(w, wshape, s);
    w = multiply(w, expand_dims(scales, -1, s), s);
    w = add(w, expand_dims(biases, -1, s), s);
    w = reshape(w, sshape, s);

    return {w};
  };

  if (s.device == Device::gpu) {
    auto out_shape = w.shape();
    out_shape.back() = out_size;
    return array(
        std::move(out_shape),
        scales.dtype(),
        std::make_shared<fast::Quantize>(
            s, fallback, group_size, bits, QuantizationMode::Affine, true),
        {w, scales, biases});
  }
  return fallback({w, scales, biases})[0];
}

array fp_dequantize(
    const array& w,
    const array& scales,
    int group_size,
    int bits,
    Dtype out_type,
    QuantizationMode mode,
    const std::optional<array>& global_scale /* = std::nullopt */,
    Stream s) {
  int expected_gs = mode == QuantizationMode::Nvfp4 ? 16 : 32;
  int expected_bits = mode == QuantizationMode::Mxfp8 ? 8 : 4;
  if (group_size != expected_gs) {
    std::ostringstream msg;
    msg << "[dequantize] " << quantization_mode_to_string(mode)
        << " quantization requires group size " << expected_gs << " but got "
        << group_size << ".";
    throw std::invalid_argument(msg.str());
  }
  if (bits != expected_bits) {
    std::ostringstream msg;
    msg << "[dequantize] " << quantization_mode_to_string(mode)
        << " quantization requires bits to be " << expected_bits << " but got "
        << bits << ".";
    throw std::invalid_argument(msg.str());
  }

  auto wshape = w.shape();
  auto sshape = scales.shape();
  if (wshape.size() != sshape.size()) {
    throw std::invalid_argument(
        "[dequantize] Shape of scales does not match the matrix");
  }

  wshape.back() = -1;
  sshape.back() = -1;

  if (wshape != sshape) {
    throw std::invalid_argument(
        "[dequantize] Shape of scales does not match the matrix");
  }

  // Packing into uint32
  int out_size = w.shape(-1) * 32 / bits;

  if (out_size != scales.shape(-1) * group_size) {
    std::ostringstream msg;
    msg << "[dequantize] Shape of scales does not match the matrix "
        << "given the quantization parameters. Provided matrix of shape "
        << w.shape() << " and scales of shape " << scales.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto inputs = std::vector<array>{w, scales};
  if (global_scale.has_value()) {
    inputs.push_back(global_scale.value());
  }

  auto fallback =
      [wshape = std::move(wshape),
       sshape = std::move(sshape),
       group_size,
       bits,
       out_type,
       s](const std::vector<array>& inputs) mutable -> std::vector<array> {
    auto out = inputs[0];
    auto scales = inputs[1];
    if (bits == 4) {
      auto lut = array(
          {
              +0.0f,
              +0.5f,
              +1.0f,
              +1.5f,
              +2.0f,
              +3.0f,
              +4.0f,
              +6.0f,
              -0.0f,
              -0.5f,
              -1.0f,
              -1.5f,
              -2.0f,
              -3.0f,
              -4.0f,
              -6.0f,
          },
          out_type);
      out = view(reshape(out, {-1, 4}, s), int8, s);
      auto idx_lo = bitwise_and(out, array(0x0F, int8), s);
      auto idx_hi = right_shift(out, array(4, int8), s);
      auto lo = gather(lut, idx_lo, 0, {1}, s);
      auto hi = gather(lut, idx_hi, 0, {1}, s);
      out = concatenate({lo, hi}, -1, s);
    } else {
      out = from_fp8(view(out, uint8, s), out_type, s);
    }
    out = reshape(out, {-1, group_size}, s);
    scales = reshape(scales, {-1, 1}, s);
    if (group_size == 16) {
      array inv_scale_enc = inputs.size() > 2
          ? divide(inputs[2], array(448.0f * 6.0f, out_type), s)
          : array(1.0f, out_type);
      scales = multiply(from_fp8(scales, out_type, s), inv_scale_enc, s);
    } else {
      scales = subtract(astype(scales, out_type, s), array(127, out_type), s);
      scales = power(array(2.0f, out_type), scales, s);
    }
    return {reshape(multiply(out, scales, s), wshape, s)};
  };

  if (s.device == Device::gpu) {
    auto out_shape = w.shape();
    out_shape.back() = out_size;
    return array(
        std::move(out_shape),
        out_type,
        std::make_shared<fast::Quantize>(
            s, fallback, group_size, bits, mode, true),
        inputs);
  }
  return fallback(inputs)[0];
}

array dequantize(
    const array& w,
    const array& scales,
    const std::optional<array>& biases /* = std::nullopt */,
    std::optional<int> group_size_ /* = std::nullopt */,
    std::optional<int> bits_ /* = std::nullopt */,
    const std::string& mode /* = "affine" */,
    const std::optional<array>& global_scale /* = std::nullopt */,
    std::optional<Dtype> dtype /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  auto [out_type, qmode] =
      validate_mode_with_type("dequantize", scales, biases, dtype, mode);
  auto [group_size, bits] =
      quantization_params_from_mode(qmode, group_size_, bits_);
  if (bits <= 0) {
    std::ostringstream msg;
    msg << "[dequantize] Invalid value for bits: " << bits;
    throw std::invalid_argument(msg.str());
  }
  if (group_size <= 0) {
    std::ostringstream msg;
    msg << "[dequantize] Invalid value for group_size: " << group_size;
    throw std::invalid_argument(msg.str());
  }
  if (w.dtype() != uint32) {
    throw std::invalid_argument(
        "[dequantize] The matrix should be given as a uint32");
  }
  if (w.ndim() < 2) {
    std::ostringstream msg;
    msg << "[dequantize] The matrix to be dequantized must have at least 2 dimension "
        << "but it has only " << w.ndim() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (global_scale.has_value()) {
    if (to_stream(s).device == Device::gpu && metal::is_available()) {
      std::ostringstream msg;
      msg << "[dequantize] Global scale is not supported on the Metal backend.";
      throw std::invalid_argument(msg.str());
    }
  }
  validate_global_scale("dequantize", qmode, global_scale);

  if (qmode == QuantizationMode::Affine) {
    return astype(
        affine_dequantize(w, scales, *biases, group_size, bits, s),
        out_type,
        s);
  } else {
    return fp_dequantize(
        w,
        scales,
        group_size,
        bits,
        out_type,
        qmode,
        global_scale,
        to_stream(s));
  }
}

array from_fp8(array x, Dtype dtype, StreamOrDevice s) {
  if (x.dtype() != uint8) {
    std::ostringstream msg;
    msg << "[from_fp8] Input must have type uint8 but "
        << "x.dtype() == " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  if (!issubdtype(dtype, floating)) {
    std::ostringstream msg;
    msg << "[from_fp8] Only real floating types are supported but "
        << "dtype == " << dtype << ".";
    throw std::invalid_argument(msg.str());
  }
  return array(
      x.shape(),
      dtype,
      std::make_shared<fast::ConvertFP8>(to_stream(s), false),
      {x});
}

array to_fp8(array x, StreamOrDevice s) {
  if (!issubdtype(x.dtype(), floating)) {
    std::ostringstream msg;
    msg << "[to_fp8] Only real floating types are supported but "
        << "x.dtype() == " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }
  return array(
      x.shape(),
      uint8,
      std::make_shared<fast::ConvertFP8>(to_stream(s), true),
      {x});
}

array gather_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases /* = std::nullopt */,
    std::optional<array> lhs_indices_ /* = std::nullopt */,
    std::optional<array> rhs_indices_ /* = std::nullopt */,
    bool transpose /* = true */,
    std::optional<int> group_size_ /* = std::nullopt */,
    std::optional<int> bits_ /* = std::nullopt */,
    const std::string& mode /* = "affine" */,
    bool sorted_indices /* = false */,
    StreamOrDevice s /* = {} */) {
  if (!lhs_indices_ && !rhs_indices_) {
    return quantized_matmul(
        x, w, scales, biases, transpose, group_size_, bits_, mode, s);
  }

  auto [out_type, qmode] =
      validate_mode_with_type("gather_qmm", scales, biases, std::nullopt, mode);
  auto [group_size, bits] =
      quantization_params_from_mode(qmode, group_size_, bits_);
  auto [w_inner_dims, w_outer_dims] = extract_quantized_matmul_dims(
      "gather_qmm", x, w, scales, biases, transpose, group_size, bits);
  if (qmode == QuantizationMode::Affine) {
    out_type = promote_types(x.dtype(), out_type);
  } else {
    out_type = x.dtype();
  }

  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[gather_qmm] Only real floating types are supported but "
        << "x.dtype() == " << x.dtype() << ".";
    throw std::invalid_argument(msg.str());
  }

  // Extract indices and broadcast them
  array lhs_indices = indices_or_default(lhs_indices_, x, s);
  array rhs_indices = indices_or_default(rhs_indices_, w, s);
  std::tie(lhs_indices, rhs_indices) =
      broadcast_arrays(lhs_indices, rhs_indices, s);

  if (!issubdtype(lhs_indices.dtype(), integer)) {
    throw std::invalid_argument(
        "[gather_qmm] Got lhs_indices with invalid dtype. Indices must be integral.");
  }

  if (!issubdtype(rhs_indices.dtype(), integer)) {
    throw std::invalid_argument(
        "[gather_qmm] Got rhs_indices with invalid dtype. Indices must be integral.");
  }
  if (x.ndim() < 2) {
    std::ostringstream msg;
    msg << "[gather_qmm] Non-quantized input must have at least two"
        << " dimensions but got input with shape " << x.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  lhs_indices = astype(lhs_indices, uint32, s);
  rhs_indices = astype(rhs_indices, uint32, s);

  // Compute the full output shape
  auto out_shape = lhs_indices.shape();
  out_shape.push_back(x.shape(-2));
  out_shape.push_back(w_outer_dims);
  std::vector<array> inputs;
  if (qmode == QuantizationMode::Affine) {
    inputs = {
        astype(x, out_type, s),
        std::move(w),
        astype(scales, out_type, s),
        astype(*biases, out_type, s),
        std::move(lhs_indices),
        std::move(rhs_indices)};
  } else {
    inputs = {
        astype(x, out_type, s),
        std::move(w),
        std::move(scales),
        std::move(lhs_indices),
        std::move(rhs_indices)};
  }
  return array(
      std::move(out_shape),
      out_type,
      std::make_shared<GatherQMM>(
          to_stream(s),
          group_size,
          bits,
          qmode,
          transpose,
          sorted_indices && !rhs_indices_,
          sorted_indices && !lhs_indices_),
      std::move(inputs));
}

array tensordot(
    const array& a,
    const array& b,
    const int axis /* = 2 */,
    StreamOrDevice s /* = {} */
) {
  if (axis < 0) {
    throw std::invalid_argument(
        "[tensordot] axis must be greater or equal to 0.");
  }
  if (axis > std::min(a.ndim(), b.ndim())) {
    throw std::invalid_argument(
        "[tensordot] axis must be less than the number of dimensions of a and b.");
  }
  std::vector<int> adims;
  std::vector<int> bdims;
  for (int i = 0; i < axis; i++) {
    bdims.emplace_back(i);
    adims.emplace_back(i - axis);
  }
  return tensordot(a, b, {adims}, {bdims}, s);
}

array tensordot(
    const array& a,
    const array& b,
    const std::vector<int>& axes_a,
    const std::vector<int>& axes_b,
    StreamOrDevice s /* = {} */) {
  if (axes_a.size() != axes_b.size()) {
    throw std::invalid_argument("[tensordot] axes must have the same size.");
  }
  int csize = 1;
  auto x = a;
  auto y = b;
  for (int i = 0; i < axes_a.size(); i++) {
    if (x.shape(axes_a.at(i)) == y.shape(axes_b.at(i))) {
      csize *= x.shape(axes_a.at(i));
    } else {
      throw std::invalid_argument(
          "[tensordot] a and b must have the same shape on the contracted axes.");
    }
  }

  std::vector<bool> cdims1(x.ndim(), false);
  std::vector<bool> cdims2(y.ndim(), false);
  for (const auto n : axes_a) {
    int n_ = (n < 0) ? n + x.ndim() : n;
    cdims1[n_] = true;
  }
  for (const auto n : axes_b) {
    int n_ = (n < 0) ? n + y.ndim() : n;
    cdims2[n_] = true;
  }

  std::vector<int> t1;
  std::vector<int> t2;
  Shape rshape;
  int size1 = 1;
  int size2 = 1;
  for (int i = 0; i < a.ndim(); i++) {
    if (!cdims1[i]) {
      t1.emplace_back(i);
      size1 *= a.shape(i);
      rshape.emplace_back(a.shape(i));
    }
  }
  for (const auto x : axes_a) {
    t1.emplace_back(x);
  }
  for (const auto x : axes_b) {
    t2.emplace_back(x);
  }
  for (int i = 0; i < b.ndim(); i++) {
    if (!cdims2[i]) {
      t2.emplace_back(i);
      size2 *= b.shape(i);
      rshape.emplace_back(b.shape(i));
    }
  }
  x = reshape(transpose(x, t1, s), {size1, csize}, s);
  y = reshape(transpose(y, t2, s), {csize, size2}, s);
  return reshape(matmul(x, y, s), rshape, s);
}

array outer(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  return multiply(
      reshape(a, {static_cast<int>(a.size()), 1}, s), flatten(b, s), s);
}

array inner(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  if (a.ndim() == 0 || b.ndim() == 0) {
    return multiply(a, b, s);
  }
  if (a.shape(-1) != b.shape(-1)) {
    throw std::invalid_argument(
        "[inner] a and b must have the same last dimension.");
  }

  return tensordot(a, b, {-1}, {-1}, s);
}

/** Compute D = beta * C + alpha * (A @ B) */
array addmm(
    array c,
    array a,
    array b,
    const float& alpha /* = 1.f */,
    const float& beta /* = 1.f */,
    StreamOrDevice s /* = {} */) {
  int in_a_ndim = a.ndim();
  int in_b_ndim = b.ndim();

  if (a.ndim() == 0 || b.ndim() == 0) {
    throw std::invalid_argument(
        "[addmm] Got 0 dimension input. Inputs must "
        "have at least one dimension.");
  }

  // Type promotion
  auto out_type = result_type(a, b, c);

  if (out_type == complex64) {
    return add(
        multiply(matmul(a, b, s), array(alpha), s),
        multiply(array(beta), c, s),
        s);
  }

  if (a.ndim() == 1) {
    // Insert a singleton dim in the beginning
    a = expand_dims(a, 0, s);
  }
  if (b.ndim() == 1) {
    // Insert a singleton dim at the end
    b = expand_dims(b, 1, s);
  }

  if (a.shape(-1) != b.shape(-2)) {
    std::ostringstream msg;
    msg << "[addmm] Last dimension of first input with shape " << a.shape()
        << " must match second to last dimension of"
        << " second input with shape " << b.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[addmm] Only real floating point types are supported but "
        << c.dtype() << ", " << a.dtype() << " and " << b.dtype()
        << " were provided which results in " << out_type
        << ", which is not a real floating point type.";
    throw std::invalid_argument(msg.str());
  }

  a = astype(a, out_type, s);
  b = astype(b, out_type, s);
  c = astype(c, out_type, s);

  // We can batch the multiplication by reshaping a
  if (a.ndim() > 2 && b.ndim() == 2 && c.ndim() <= 1) {
    auto out_shape = a.shape();
    a = reshape(a, {-1, out_shape.back()}, s);
    out_shape.back() = b.shape(-1);

    if (in_b_ndim == 1) {
      out_shape.pop_back();
    }

    c = broadcast_to(c, {a.shape(0), b.shape(1)}, s);

    auto out = array(
        {a.shape(0), b.shape(1)},
        out_type,
        std::make_shared<AddMM>(to_stream(s), alpha, beta),
        {a, b, c});
    return reshape(out, out_shape, s);
  }

  if (a.ndim() > 2 || b.ndim() > 2) {
    Shape bsx_a(a.shape().begin(), a.shape().end() - 2);
    Shape bsx_b(b.shape().begin(), b.shape().end() - 2);
    auto inner_shape = broadcast_shapes(bsx_a, bsx_b);

    // Broadcast a
    inner_shape.push_back(a.shape(-2));
    inner_shape.push_back(a.shape(-1));
    a = broadcast_to(a, inner_shape, s);

    // Broadcast b
    *(inner_shape.end() - 2) = b.shape(-2);
    *(inner_shape.end() - 1) = b.shape(-1);
    b = broadcast_to(b, inner_shape, s);
  }

  auto out_shape = a.shape();
  out_shape.back() = b.shape(-1);

  auto out_shape_adjusted = out_shape;

  if (in_a_ndim == 1 || in_b_ndim == 1) {
    out_shape_adjusted.erase(
        out_shape_adjusted.end() - ((in_a_ndim == 1) ? 2 : 1),
        out_shape_adjusted.end() - ((in_b_ndim == 1) ? 0 : 1));
  }

  auto c_broadcast_shape = broadcast_shapes(c.shape(), out_shape_adjusted);
  c = broadcast_to(c, c_broadcast_shape, s);

  if (in_a_ndim == 1 || in_b_ndim == 1) {
    auto c_reshape = c.shape();
    if (in_b_ndim == 1) {
      c_reshape.push_back(1);
    }

    if (in_a_ndim == 1) {
      c_reshape.push_back(c_reshape.back());
      c_reshape[c_reshape.size() - 2] = 1;
    }

    c = reshape(c, c_reshape, s);
  }
  if (c.shape() != out_shape) {
    throw std::invalid_argument(
        "[addmm] input c must broadcast to the output shape");
  }

  auto out = array(
      std::move(out_shape),
      out_type,
      std::make_shared<AddMM>(to_stream(s), alpha, beta),
      {a, b, c});

  // Remove the possibly inserted singleton dimensions
  std::vector<int> axes;
  if (in_a_ndim == 1) {
    axes.push_back(out.ndim() - 2);
  }
  if (in_b_ndim == 1) {
    axes.push_back(out.ndim() - 1);
  }
  return axes.empty() ? out : squeeze(out, axes, s);
}

/** Compute matrix product with tile-level masking */
array block_masked_mm(
    array a,
    array b,
    int block_size,
    std::optional<array> mask_out /* = std::nullopt */,
    std::optional<array> mask_lhs /* = std::nullopt */,
    std::optional<array> mask_rhs /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  // If no masks, just perform regular matmul
  if (!mask_out && !mask_lhs && !mask_rhs) {
    return matmul(a, b, s);
  }

  bool has_out_mask = mask_out.has_value();
  bool has_operand_mask = mask_lhs.has_value() || mask_rhs.has_value();

  // Check valid tile sizes
  // TODO: Add support for 16x16 tile
  if (block_size != 32 && block_size != 64) {
    std::ostringstream msg;
    msg << "[block_masked_mm] Only block_sizes 32, 64 are supported."
        << "Got block size " << block_size << ".";
    throw std::invalid_argument(msg.str());
  }

  // Do shape checks for operands
  int in_a_ndim = a.ndim();
  int in_b_ndim = b.ndim();

  if (a.ndim() == 0 || b.ndim() == 0) {
    throw std::invalid_argument(
        "[block_masked_mm] Got 0 dimension input. Inputs must "
        "have at least one dimension.");
  }

  if (a.ndim() == 1) {
    // Insert a singleton dim in the beginning
    a = expand_dims(a, 0, s);
  }
  if (b.ndim() == 1) {
    // Insert a singleton dim at the end
    b = expand_dims(b, 1, s);
  }

  if (a.shape(-1) != b.shape(-2)) {
    std::ostringstream msg;
    msg << "[block_masked_mm] Last dimension of first input with shape "
        << a.shape() << " must match second to last dimension of"
        << " second input with shape " << b.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  // Type promotion
  auto out_type = result_type(a, b);
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[block_masked_mm] Only real floating point types are supported but "
        << a.dtype() << " and " << b.dtype()
        << " were provided which results in " << out_type
        << ", which is not a real floating point type.";
    throw std::invalid_argument(msg.str());
  }

  a = astype(a, out_type, s);
  b = astype(b, out_type, s);

  // Handle broadcasting
  Shape bsx_a(a.shape().begin(), a.shape().end() - 2);
  Shape bsx_b(b.shape().begin(), b.shape().end() - 2);

  auto bsx_shape = broadcast_shapes(bsx_a, bsx_b);

  bsx_shape.push_back(1);
  bsx_shape.push_back(1);
  int nd = bsx_shape.size();

  int M = a.shape(-2);
  int N = b.shape(-1);
  int K = a.shape(-1);

  // Prepare A
  bsx_shape[nd - 2] = M;
  bsx_shape[nd - 1] = K;
  a = broadcast_to(a, bsx_shape, s);

  // Prepare B
  bsx_shape[nd - 2] = K;
  bsx_shape[nd - 1] = N;
  b = broadcast_to(b, bsx_shape, s);

  // Get output shape
  auto out_shape = bsx_shape;
  out_shape[nd - 2] = M;
  out_shape[nd - 1] = N;

  // Determine mask shape requirments
  int tm = (M + block_size - 1) / block_size;
  int tn = (N + block_size - 1) / block_size;
  int tk = (K + block_size - 1) / block_size;

  std::vector<array> inputs = {a, b};

  // Broadcast and astype mask
  auto broadcast_mask = [](array mask,
                           Shape& bs_shape,
                           int y,
                           int x,
                           Dtype mask_dtype,
                           StreamOrDevice s) {
    int nd_bsx = bs_shape.size();
    bs_shape[nd_bsx - 2] = y;
    bs_shape[nd_bsx - 1] = x;
    mask = astype(mask, mask_dtype, s);
    return broadcast_to(mask, bs_shape, s);
  };

  // Out mask
  if (has_out_mask) {
    array mask_out_p = mask_out.value_or(array({true}));
    if (in_a_ndim == 1 || in_b_ndim == 1) {
      std::vector<int> ex_dims;
      if (in_a_ndim == 1)
        ex_dims.push_back(-2);
      if (in_b_ndim == 1)
        ex_dims.push_back(-1);
      mask_out_p = expand_dims(mask_out_p, ex_dims, s);
    }
    auto maskout_dtype = mask_out_p.dtype() == bool_ ? bool_ : out_type;
    mask_out_p =
        broadcast_mask(mask_out_p, bsx_shape, tm, tn, maskout_dtype, s);

    inputs.push_back(mask_out_p);
  }

  // Operand masks
  if (has_operand_mask) {
    // Pull masks
    array mask_lhs_p = mask_lhs.value_or(array({true}));
    array mask_rhs_p = mask_rhs.value_or(array({true}));
    auto mask_dtype =
        (mask_lhs_p.dtype() == bool_ && mask_rhs_p.dtype() == bool_) ? bool_
                                                                     : out_type;

    // LHS mask
    if (in_a_ndim == 1) {
      mask_lhs_p = expand_dims(mask_lhs_p, -2, s);
    }
    mask_lhs_p = broadcast_mask(mask_lhs_p, bsx_shape, tm, tk, mask_dtype, s);

    // RHS mask
    if (in_b_ndim == 1) {
      mask_rhs_p = expand_dims(mask_rhs_p, -1, s);
    }
    mask_rhs_p = broadcast_mask(mask_rhs_p, bsx_shape, tk, tn, mask_dtype, s);

    inputs.push_back(mask_lhs_p);
    inputs.push_back(mask_rhs_p);
  }

  // Caculate array
  auto out = array(
      std::move(out_shape),
      out_type,
      std::make_shared<BlockMaskedMM>(to_stream(s), block_size),
      std::move(inputs));
  // Remove the possibly inserted singleton dimensions
  std::vector<int> axes;
  if (in_a_ndim == 1) {
    axes.push_back(out.ndim() - 2);
  }
  if (in_b_ndim == 1) {
    axes.push_back(out.ndim() - 1);
  }
  return axes.empty() ? out : squeeze(out, axes, s);
}

/** Compute matrix product with matrix-level gather */
array gather_mm(
    array a,
    array b,
    std::optional<array> lhs_indices_ /* = std::nullopt */,
    std::optional<array> rhs_indices_ /* = std::nullopt */,
    bool sorted_indices /* = false */,
    StreamOrDevice s /* = {} */) {
  // If no indices, fall back to full matmul
  if (!lhs_indices_ && !rhs_indices_) {
    return matmul(a, b, s);
  }

  // Do shape checks for operands
  int in_a_ndim = a.ndim();
  int in_b_ndim = b.ndim();

  if (a.ndim() == 0 || b.ndim() == 0) {
    throw std::invalid_argument(
        "[gather_mm] Got 0 dimension input. Inputs must "
        "have at least one dimension.");
  }

  if (a.ndim() == 1) {
    // Insert a singleton dim in the beginning
    a = expand_dims(a, 0, s);
  }
  if (b.ndim() == 1) {
    // Insert a singleton dim at the end
    b = expand_dims(b, 1, s);
  }

  if (a.shape(-1) != b.shape(-2)) {
    std::ostringstream msg;
    msg << "[gather_mm] Last dimension of first input with shape " << a.shape()
        << " must match second to last dimension of"
        << " second input with shape " << b.shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  // Type promotion
  auto out_type = result_type(a, b);
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[gather_mm] Only real floating point types are supported but "
        << a.dtype() << " and " << b.dtype()
        << " were provided which results in " << out_type
        << ", which is not a real floating point type.";
    throw std::invalid_argument(msg.str());
  }

  a = astype(a, out_type, s);
  b = astype(b, out_type, s);

  // Handle broadcasting
  array lhs_indices = indices_or_default(lhs_indices_, a, s);
  array rhs_indices = indices_or_default(rhs_indices_, b, s);

  if (!issubdtype(lhs_indices.dtype(), integer)) {
    throw std::invalid_argument(
        "[gather_mm] Got lhs_indices with invalid dtype. Indices must be integral.");
  }

  if (!issubdtype(rhs_indices.dtype(), integer)) {
    throw std::invalid_argument(
        "[gather_mm] Got rhs_indices with invalid dtype. Indices must be integral.");
  }

  lhs_indices = astype(lhs_indices, uint32, s);
  rhs_indices = astype(rhs_indices, uint32, s);

  int M = a.shape(-2);
  int N = b.shape(-1);

  std::tie(lhs_indices, rhs_indices) =
      broadcast_arrays(lhs_indices, rhs_indices, s);

  auto out_shape = lhs_indices.shape();
  out_shape.push_back(M);
  out_shape.push_back(N);

  // Make the output array
  auto out = array(
      std::move(out_shape),
      out_type,
      std::make_shared<GatherMM>(
          to_stream(s),
          sorted_indices && !rhs_indices_,
          sorted_indices && !lhs_indices_),
      {std::move(a),
       std::move(b),
       std::move(lhs_indices),
       std::move(rhs_indices)});

  // Remove the possibly inserted singleton dimensions
  std::vector<int> axes;
  if (in_a_ndim == 1) {
    axes.push_back(out.ndim() - 2);
  }
  if (in_b_ndim == 1) {
    axes.push_back(out.ndim() - 1);
  }
  return axes.empty() ? out : squeeze(out, axes, s);
}

array segmented_mm(
    array a,
    array b,
    array segments,
    StreamOrDevice s /* = {} */) {
  if (a.ndim() != 2 || b.ndim() != 2) {
    throw std::invalid_argument("[segmented_mm] Batched matmul not supported");
  }

  if (segments.ndim() < 1 || segments.shape().back() != 2) {
    std::ostringstream msg;
    msg << "[segmented_mm] The segments should have shape (..., 2) but "
        << segments.shape() << " was provided.";
    throw std::invalid_argument(msg.str());
  }

  // Type promotion
  auto out_type = result_type(a, b);
  if (!issubdtype(out_type, floating)) {
    std::ostringstream msg;
    msg << "[segmented_mm] Only real floating point types are supported but "
        << a.dtype() << " and " << b.dtype()
        << " were provided which results in " << out_type
        << ", which is not a real floating point type.";
    throw std::invalid_argument(msg.str());
  }

  if (!issubdtype(segments.dtype(), integer)) {
    throw std::invalid_argument(
        "[segmented_mm] Got segments with invalid dtype. Segments must be integral.");
  }

  a = astype(a, out_type, s);
  b = astype(b, out_type, s);
  segments = astype(segments, uint32, s);

  Shape out_shape = segments.shape();
  out_shape.pop_back();
  out_shape.push_back(a.shape(0));
  out_shape.push_back(b.shape(1));

  return array(
      std::move(out_shape),
      out_type,
      std::make_shared<SegmentedMM>(to_stream(s)),
      {std::move(a), std::move(b), std::move(segments)});
}

array diagonal(
    const array& a,
    int offset /* = 0 */,
    int axis1 /* = 0 */,
    int axis2 /* = 1 */,
    StreamOrDevice s /* = {} */
) {
  int ndim = a.ndim();
  if (ndim < 2) {
    std::ostringstream msg;
    msg << "[diagonal] Array must have at least two dimensions, but got "
        << ndim << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  auto ax1 = (axis1 < 0) ? axis1 + ndim : axis1;
  if (ax1 < 0 || ax1 >= ndim) {
    std::ostringstream msg;
    msg << "[diagonal] Invalid axis1 " << axis1 << " for array with " << ndim
        << " dimensions.";
    throw std::out_of_range(msg.str());
  }

  auto ax2 = (axis2 < 0) ? axis2 + ndim : axis2;
  if (ax2 < 0 || ax2 >= ndim) {
    std::ostringstream msg;
    msg << "[diagonal] Invalid axis2 " << axis2 << " for array with " << ndim
        << " dimensions.";
    throw std::out_of_range(msg.str());
  }

  if (ax1 == ax2) {
    throw std::invalid_argument(
        "[diagonal] axis1 and axis2 cannot be the same axis");
  }

  ShapeElem off1 = std::max(-offset, 0);
  ShapeElem off2 = std::max(offset, 0);

  auto diag_size = std::min(a.shape(ax1) - off1, a.shape(ax2) - off2);
  diag_size = diag_size < 0 ? 0 : diag_size;

  std::vector<array> indices = {
      arange(off1, off1 + diag_size, s), arange(off2, off2 + diag_size, s)};

  Shape slice_sizes = a.shape();
  slice_sizes[ax1] = 1;
  slice_sizes[ax2] = 1;

  auto out = gather(a, indices, {ax1, ax2}, slice_sizes, s);
  return moveaxis(squeeze(out, {ax1 + 1, ax2 + 1}, s), 0, -1, s);
}

array diag(const array& a, int k /* = 0 */, StreamOrDevice s /* = {} */) {
  if (a.ndim() == 1) {
    int a_size = a.size();
    int n = a_size + std::abs(k);
    auto res = zeros({n, n}, a.dtype(), s);

    std::vector<array> indices;
    auto s1 = std::max(0, -k);
    auto s2 = std::max(0, k);
    indices.push_back(arange(s1, a_size + s1, uint32, s));
    indices.push_back(arange(s2, a_size + s2, uint32, s));

    return scatter(res, indices, reshape(a, {a_size, 1, 1}, s), {0, 1}, s);
  } else if (a.ndim() == 2) {
    return diagonal(a, k, 0, 1, s);
  } else {
    std::ostringstream msg;
    msg << "[diag] array must be 1-D or 2-D, got array with " << a.ndim()
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
}

array trace(
    const array& a,
    int offset,
    int axis1,
    int axis2,
    Dtype dtype,
    StreamOrDevice s /* = {} */) {
  int ndim = a.ndim();
  if (ndim < 2) {
    std::ostringstream msg;
    msg << "[trace] Array must have at least two dimensions, but got " << ndim
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  auto ax1 = (axis1 < 0) ? axis1 + ndim : axis1;
  if (ax1 < 0 || ax1 >= ndim) {
    std::ostringstream msg;
    msg << "[trace] Invalid axis1 " << axis1 << " for array with " << ndim
        << " dimensions.";
    throw std::out_of_range(msg.str());
  }

  auto ax2 = (axis2 < 0) ? axis2 + ndim : axis2;
  if (ax2 < 0 || ax2 >= ndim) {
    std::ostringstream msg;
    msg << "[trace] Invalid axis2 " << axis2 << " for array with " << ndim
        << " dimensions.";
    throw std::out_of_range(msg.str());
  }

  if (ax1 == ax2) {
    throw std::invalid_argument(
        "[trace] axis1 and axis2 cannot be the same axis");
  }

  return sum(
      astype(diagonal(a, offset, axis1, axis2, s), dtype, s),
      /* axis = */ -1,
      /* keepdims = */ false,
      s);
}
array trace(
    const array& a,
    int offset,
    int axis1,
    int axis2,
    StreamOrDevice s /* = {} */) {
  auto dtype = a.dtype();
  return trace(a, offset, axis1, axis2, dtype, s);
}
array trace(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = a.dtype();
  return trace(a, 0, 0, 1, dtype, s);
}

std::vector<array> depends(
    const std::vector<array>& inputs,
    const std::vector<array>& dependencies) {
  std::vector<array> all_inputs = inputs;
  all_inputs.insert(all_inputs.end(), dependencies.begin(), dependencies.end());

  // Compute the stream. Maybe do it in a smarter way at some point in the
  // future.
  Stream s = (inputs[0].has_primitive()) ? inputs[0].primitive().stream()
                                         : to_stream({});
  // Make the output info
  std::vector<Shape> shapes;
  std::vector<Dtype> dtypes;
  for (const auto& in : inputs) {
    shapes.emplace_back(in.shape());
    dtypes.emplace_back(in.dtype());
  }

  return array::make_arrays(
      std::move(shapes),
      dtypes,
      std::make_shared<Depends>(to_stream(s)),
      all_inputs);
}

array atleast_1d(const array& a, StreamOrDevice s /* = {} */) {
  if (a.ndim() == 0) {
    return reshape(a, {1}, s);
  }
  return a;
}

std::vector<array> atleast_1d(
    const std::vector<array>& arrays,
    StreamOrDevice s /* = {} */) {
  std::vector<array> out;
  out.reserve(arrays.size());
  for (const auto& a : arrays) {
    out.push_back(atleast_1d(a, s));
  }
  return out;
}

array atleast_2d(const array& a, StreamOrDevice s /* = {} */) {
  switch (a.ndim()) {
    case 0:
      return reshape(a, {1, 1}, s);
    case 1:
      return reshape(a, {1, a.shape(0)}, s);
    default:
      return a;
  }
}

std::vector<array> atleast_2d(
    const std::vector<array>& arrays,
    StreamOrDevice s /* = {} */) {
  std::vector<array> out;
  out.reserve(arrays.size());
  for (const auto& a : arrays) {
    out.push_back(atleast_2d(a, s));
  }
  return out;
}

array atleast_3d(const array& a, StreamOrDevice s /* = {} */) {
  switch (a.ndim()) {
    case 0:
      return reshape(a, {1, 1, 1}, s);
    case 1:
      return reshape(a, {1, a.shape(0), 1}, s);
    case 2:
      return reshape(a, {a.shape(0), a.shape(1), 1}, s);
    default:
      return a;
  }
}

std::vector<array> atleast_3d(
    const std::vector<array>& arrays,
    StreamOrDevice s /* = {} */) {
  std::vector<array> out;
  out.reserve(arrays.size());
  for (const auto& a : arrays) {
    out.push_back(atleast_3d(a, s));
  }
  return out;
}

array number_of_elements(
    const array& a,
    std::vector<int> axes,
    bool inverted,
    Dtype dtype /* = int32 */,
    StreamOrDevice s /* = {} */) {
  for (auto& ax : axes) {
    int normal_axis = (ax + a.ndim()) % a.ndim();
    if (normal_axis >= a.ndim() || normal_axis < 0) {
      std::ostringstream msg;
      msg << "[number_of_elements] Can't get the shape for axis " << ax
          << " from an array with " << a.ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    ax = normal_axis;
  }

  if (!detail::in_dynamic_tracing()) {
    double numel = 1;
    for (auto ax : axes) {
      numel *= a.shape(ax);
    }
    return array(inverted ? 1.0 / numel : numel, dtype);
  }
  return stop_gradient(array(
      Shape{},
      dtype,
      std::make_shared<NumberOfElements>(
          to_stream(s), std::move(axes), inverted, dtype),
      {a}));
}

array conjugate(const array& a, StreamOrDevice s /* = {} */) {
  // Mirror NumPy's behaviour for real input
  if (a.dtype() != complex64) {
    return a;
  }
  return array(
      a.shape(), a.dtype(), std::make_shared<Conjugate>(to_stream(s)), {a});
}

array bitwise_impl(
    const array& a,
    const array& b,
    BitwiseBinary::Op op,
    const std::string& op_name,
    const StreamOrDevice& s,
    std::optional<Dtype> out_type_ = std::nullopt) {
  auto out_type = out_type_ ? *out_type_ : promote_types(a.dtype(), b.dtype());
  if (!(issubdtype(out_type, integer) || out_type == bool_)) {
    std::ostringstream msg;
    msg << "[" << op_name
        << "] Only allowed on integer or boolean types "
           "but got types "
        << a.dtype() << " and " << b.dtype() << ".";
    throw std::runtime_error(msg.str());
  }
  auto inputs =
      broadcast_arrays({astype(a, out_type, s), astype(b, out_type, s)}, s);
  auto& out_shape = inputs[0].shape();
  return array(
      out_shape,
      out_type,
      std::make_shared<BitwiseBinary>(to_stream(s), op),
      std::move(inputs));
}

array bitwise_and(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  return bitwise_impl(a, b, BitwiseBinary::Op::And, "bitwise_and", s);
}
array operator&(const array& a, const array& b) {
  return bitwise_and(a, b);
}

array bitwise_or(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  return bitwise_impl(a, b, BitwiseBinary::Op::Or, "bitwise_or", s);
}
array operator|(const array& a, const array& b) {
  return bitwise_or(a, b);
}

array bitwise_xor(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  return bitwise_impl(a, b, BitwiseBinary::Op::Xor, "bitwise_xor", s);
}
array operator^(const array& a, const array& b) {
  return bitwise_xor(a, b);
}

array left_shift(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto t = result_type(a, b);
  if (t == bool_) {
    t = uint8;
  }
  return bitwise_impl(a, b, BitwiseBinary::Op::LeftShift, "left_shift", s, t);
}
array operator<<(const array& a, const array& b) {
  return left_shift(a, b);
}

array right_shift(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto t = result_type(a, b);
  if (t == bool_) {
    t = uint8;
  }
  return bitwise_impl(
      astype(a, t, s),
      astype(b, t, s),
      BitwiseBinary::Op::RightShift,
      "right_shift",
      s,
      t);
}
array operator>>(const array& a, const array& b) {
  return right_shift(a, b);
}

array bitwise_invert(const array& a, StreamOrDevice s /* = {} */) {
  if (issubdtype(a.dtype(), inexact)) {
    throw std::invalid_argument(
        "[bitwise_invert] Bitwise inverse only allowed on integer types.");
  } else if (a.dtype() == bool_) {
    return logical_not(a, s);
  }
  return array(
      a.shape(), a.dtype(), std::make_shared<BitwiseInvert>(to_stream(s)), {a});
}

array operator~(const array& a) {
  return bitwise_invert(a);
}

array view(const array& a, const Dtype& dtype, StreamOrDevice s /* = {} */) {
  if (a.dtype() == dtype) {
    return a;
  }
  auto out_shape = a.shape();
  auto ibytes = size_of(a.dtype());
  auto obytes = size_of(dtype);
  if (a.ndim() == 0 && ibytes != obytes) {
    throw std::invalid_argument(
        "[view] Changing the type of a scalar is only allowed"
        " for types with the same size.");
  } else {
    if (ibytes < obytes) {
      if (out_shape.back() % (obytes / ibytes) != 0) {
        throw std::invalid_argument(
            "[view] When viewing as a larger dtype, the size in bytes of the last"
            " axis must be a multiple of the requested type size.");
      }
      out_shape.back() /= (obytes / ibytes);
    } else if (ibytes > obytes) {
      // Type size ratios are always integers
      out_shape.back() *= (ibytes / obytes);
    }
  }
  return array(
      out_shape, dtype, std::make_shared<View>(to_stream(s), dtype), {a});
}

array roll(
    const array& a,
    const Shape& shift,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  if (axes.empty()) {
    return a;
  }

  if (shift.size() < axes.size()) {
    std::ostringstream msg;
    msg << "[roll] At least one shift value per axis is required, "
        << shift.size() << " provided for " << axes.size() << " axes.";
    throw std::invalid_argument(msg.str());
  }

  array result = a;
  for (int i = 0; i < axes.size(); i++) {
    int ax = axes[i];
    if (ax < 0) {
      ax += a.ndim();
    }
    if (ax < 0 || ax >= a.ndim()) {
      std::ostringstream msg;
      msg << "[roll] Invalid axis " << axes[i] << " for array with " << a.ndim()
          << " dimensions.";
      throw std::invalid_argument(msg.str());
    }

    auto sh = shift[i];
    auto size = a.shape(ax);
    if (size == 0) {
      continue; // skip rolling this axis if it has size 0
    }
    auto split_index = (sh < 0) ? (-sh) % size : size - sh % size;

    auto parts = split(result, Shape{split_index}, ax, s);
    std::swap(parts[0], parts[1]);
    result = concatenate(parts, ax, s);
  }

  return result;
}

array roll(const array& a, int shift, StreamOrDevice s /* = {} */) {
  auto shape = a.shape();
  return reshape(
      roll(flatten(a, s), Shape{shift}, std::vector<int>{0}, s),
      std::move(shape),
      s);
}

array roll(const array& a, const Shape& shift, StreamOrDevice s /* = {} */) {
  int total_shift = 0;
  for (auto& s : shift) {
    total_shift += s;
  }
  return roll(a, total_shift, s);
}

array roll(const array& a, int shift, int axis, StreamOrDevice s /* = {} */) {
  return roll(a, Shape{shift}, std::vector<int>{axis}, s);
}

array roll(
    const array& a,
    int shift,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  Shape shifts(axes.size(), shift);
  return roll(a, shifts, axes, s);
}

array roll(
    const array& a,
    const Shape& shift,
    int axis,
    StreamOrDevice s /* = {} */) {
  int total_shift = 0;
  for (auto& s : shift) {
    total_shift += s;
  }
  return roll(a, Shape{total_shift}, std::vector<int>{axis}, s);
}

array real(const array& a, StreamOrDevice s /* = {} */) {
  if (!issubdtype(a.dtype(), complexfloating)) {
    return a;
  }
  return array(a.shape(), float32, std::make_shared<Real>(to_stream(s)), {a});
}

array imag(const array& a, StreamOrDevice s /* = {} */) {
  if (!issubdtype(a.dtype(), complexfloating)) {
    return zeros_like(a);
  }
  return array(a.shape(), float32, std::make_shared<Imag>(to_stream(s)), {a});
}

array contiguous(
    const array& a,
    bool allow_col_major /* = false */,
    StreamOrDevice s /* = {} */) {
  return array(
      a.shape(),
      a.dtype(),
      std::make_shared<Contiguous>(to_stream(s), allow_col_major),
      {a});
}

} // namespace mlx::core