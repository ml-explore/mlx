// Copyright © 2023 Apple Inc.

#include <cmath>
#include <numeric>
#include <set>
#include <sstream>

#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core {

namespace {

std::pair<std::vector<int>, std::vector<int>> compute_reduce_shape(
    const std::vector<int>& axes,
    const std::vector<int>& shape,
    bool keepdims) {
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
  std::vector<int> out_shape;
  for (int i = 0; i < ndim; ++i) {
    if (axes_set.count(i) == 0) {
      out_shape.push_back(shape[i]);
    } else if (keepdims) {
      out_shape.push_back(1);
    }
  }
  std::vector<int> sorted_axes(axes_set.begin(), axes_set.end());
  return {out_shape, sorted_axes};
}

int compute_number_of_elements(const array& a, const std::vector<int>& axes) {
  int nelements = 1;
  for (auto axis : axes) {
    nelements *= a.shape(axis);
  }
  return nelements;
}

Dtype at_least_float(const Dtype& d) {
  return is_floating_point(d) ? d : promote_types(d, float32);
}

} // namespace

Stream to_stream(StreamOrDevice s) {
  if (std::holds_alternative<std::monostate>(s)) {
    return default_stream(default_device());
  } else if (std::holds_alternative<Device>(s)) {
    return default_stream(std::get<Device>(s));
  } else {
    return std::get<Stream>(s);
  }
}

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
  int size = std::max(static_cast<int>(std::ceil((stop - start) / step)), 0);
  return array(
      {size},
      dtype,
      std::make_unique<Arange>(to_stream(s), start, stop, step),
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
  array sequence = arange(0, num, float32, to_stream(s));
  float step = (stop - start) / (num - 1);
  return astype(
      add(multiply(sequence, array(step), to_stream(s)),
          array(start),
          to_stream(s)),
      dtype,
      to_stream(s));
}

array astype(const array& a, Dtype dtype, StreamOrDevice s /* = {} */) {
  if (dtype == a.dtype()) {
    return a;
  }
  return array(
      a.shape(), dtype, std::make_unique<AsType>(to_stream(s), dtype), {a});
}

array as_strided(
    const array& a,
    std::vector<int> shape,
    std::vector<size_t> strides,
    size_t offset,
    StreamOrDevice s /* = {} */) {
  // Force the input array to be contiguous
  auto x = reshape(a, {-1}, s);
  return array(
      shape,
      a.dtype(),
      std::make_unique<AsStrided>(to_stream(s), shape, strides, offset),
      {x});
}

array copy(const array& a, StreamOrDevice s /* = {} */) {
  return array(a.shape(), a.dtype(), std::make_unique<Copy>(to_stream(s)), {a});
}

array full(
    const std::vector<int>& shape,
    const array& vals,
    Dtype dtype,
    StreamOrDevice s /* = {} */) {
  auto in = broadcast_to(astype(vals, dtype, s), shape, s);
  return array(shape, dtype, std::make_unique<Full>(to_stream(s)), {in});
}

array full(
    const std::vector<int>& shape,
    const array& vals,
    StreamOrDevice s /* = {} */) {
  return full(shape, vals, vals.dtype(), to_stream(s));
}

array zeros(
    const std::vector<int>& shape,
    Dtype dtype,
    StreamOrDevice s /* = {} */) {
  return full(shape, array(0, dtype), to_stream(s));
}

array zeros_like(const array& a, StreamOrDevice s /* = {} */) {
  return zeros(a.shape(), a.dtype(), to_stream(s));
}

array ones(
    const std::vector<int>& shape,
    Dtype dtype,
    StreamOrDevice s /* = {} */) {
  return full(shape, array(1, dtype), to_stream(s));
}

array ones_like(const array& a, StreamOrDevice s /* = {} */) {
  return ones(a.shape(), a.dtype(), to_stream(s));
}

array eye(int n, int m, int k, Dtype dtype, StreamOrDevice s /* = {} */) {
  if (n <= 0 || m <= 0) {
    throw std::invalid_argument("N and M must be positive integers.");
  }
  array result = zeros({n * m}, dtype, s);
  if (k >= m || -k >= n) {
    return reshape(result, {n, m}, s);
  }

  int diagonal_length = k >= 0 ? std::min(n, m - k) : std::min(n + k, m);
  int start_index = (k >= 0) ? k : -k * m;

  array diag_indices_array = arange(
      start_index, start_index + diagonal_length * (m + 1), m + 1, int32, s);
  array ones_array = ones({diagonal_length, 1}, dtype, s);
  result = scatter(result, diag_indices_array, ones_array, 0, s);

  return reshape(result, {n, m}, s);
}

array identity(int n, Dtype dtype, StreamOrDevice s /* = {} */) {
  return eye(n, n, 0, dtype, s);
}

array tri(int n, int m, int k, Dtype type, StreamOrDevice s /* = {} */) {
  auto l = expand_dims(arange(n, s), 1, s);
  auto r = expand_dims(arange(-k, m - k, s), 0, s);
  return astype(greater_equal(l, r, s), type, s);
}

array tril(array x, int k, StreamOrDevice s /* = {} */) {
  if (x.ndim() < 2) {
    throw std::invalid_argument("[tril] array must be atleast 2-D");
  }
  auto mask = tri(x.shape(-2), x.shape(-1), k, x.dtype(), s);
  return where(mask, x, zeros_like(x, s), s);
}

array triu(array x, int k, StreamOrDevice s /* = {} */) {
  if (x.ndim() < 2) {
    throw std::invalid_argument("[triu] array must be atleast 2-D");
  }
  auto mask = tri(x.shape(-2), x.shape(-1), k - 1, x.dtype(), s);
  return where(mask, zeros_like(x, s), x, s);
}

array reshape(
    const array& a,
    std::vector<int> shape,
    StreamOrDevice s /* = {} */) {
  if (a.shape() == shape) {
    return a;
  }

  size_t size = 1;
  int infer_idx = -1;
  for (int i = 0; i < shape.size(); ++i) {
    if (shape[i] == -1) {
      if (infer_idx >= 0) {
        throw std::invalid_argument("Reshape can only infer one dimension.");
      }
      infer_idx = i;
    } else {
      size *= shape[i];
    }
  }
  if (size > 0) {
    auto q_and_r = std::ldiv(a.size(), size);
    if (infer_idx >= 0) {
      shape[infer_idx] = q_and_r.quot;
      size *= q_and_r.quot;
    }
  }
  if (a.size() != size) {
    std::ostringstream msg;
    msg << "Cannot reshape array of size " << a.size() << " into shape "
        << shape << ".";
    throw std::invalid_argument(msg.str());
  }
  return array(
      shape, a.dtype(), std::make_unique<Reshape>(to_stream(s), shape), {a});
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
  std::vector<int> new_shape(a.shape().begin(), a.shape().begin() + start_ax);
  new_shape.push_back(-1);
  new_shape.insert(
      new_shape.end(), a.shape().begin() + end_ax + 1, a.shape().end());
  return reshape(a, new_shape, s);
}

array flatten(const array& a, StreamOrDevice s /* = {} */) {
  return flatten(a, 0, a.ndim() - 1, s);
}

array squeeze(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  std::set<int> unique_axes;
  for (auto ax : axes) {
    ax = ax < 0 ? ax + a.ndim() : ax;
    if (ax < 0 || ax >= a.ndim()) {
      std::ostringstream msg;
      msg << "[squeeze] Invalid axies " << ax << " for array with " << a.ndim()
          << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    if (a.shape(ax) != 1) {
      std::ostringstream msg;
      msg << "[squeeze] Cannot squeeze axis " << ax << " with size "
          << a.shape(ax) << " which is not equal to 1.";
      throw std::invalid_argument(msg.str());
    }
    unique_axes.insert(ax);
  }

  if (unique_axes.size() != axes.size()) {
    throw std::invalid_argument("[squeeze] Received duplicate axes.");
  }
  std::vector<int> sorted_axes(unique_axes.begin(), unique_axes.end());
  std::vector<int> shape;
  for (int i = 0, j = 0; i < a.ndim(); ++i) {
    if (j < sorted_axes.size() && i == sorted_axes[j]) {
      j++;
    } else {
      shape.push_back(a.shape(i));
    }
  }
  return reshape(a, shape, s);
}

array squeeze(const array& a, StreamOrDevice s /* = {} */) {
  std::vector<int> axes;
  for (int i = 0; i < a.ndim(); ++i) {
    if (a.shape(i) == 1) {
      axes.push_back(i);
    }
  }
  return squeeze(a, axes, s);
}

array expand_dims(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  { // Check for repeats
    std::set<int> unique_axes(axes.begin(), axes.end());
    if (unique_axes.size() != axes.size()) {
      throw std::invalid_argument("[expand_dims] Received duplicate axes.");
    }
  }

  int out_ndim = axes.size() + a.ndim();
  std::vector<int> canonical_axes = axes;
  for (auto& ax : canonical_axes) {
    ax = ax < 0 ? ax + out_ndim : ax;
    if (ax < 0 || ax >= out_ndim) {
      std::ostringstream msg;
      msg << "[squeeze] Invalid axies " << ax << " for output array with "
          << a.ndim() << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
  }

  // Check for repeats again
  std::set<int> unique_axes(canonical_axes.begin(), canonical_axes.end());
  if (unique_axes.size() != axes.size()) {
    throw std::invalid_argument("[expand_dims] Received duplicate axes.");
  }

  std::vector<int> sorted_axes(unique_axes.begin(), unique_axes.end());
  auto out_shape = a.shape();
  for (int i = 0; i < sorted_axes.size(); ++i) {
    out_shape.insert(out_shape.begin() + sorted_axes[i], 1);
  }
  return reshape(a, out_shape, s);
}

array slice(
    const array& a,
    std::vector<int> start,
    std::vector<int> stop,
    std::vector<int> strides,
    StreamOrDevice s /* = {} */) {
  if (start.size() != a.ndim() || stop.size() != a.ndim() ||
      strides.size() != a.ndim()) {
    std::ostringstream msg;
    msg << "[slice] Invalid number of indices or strides for "
        << "array with dimension " << a.ndim() << ".";
    throw std::invalid_argument(msg.str());
  }

  std::vector<int> negatively_strided_axes;
  std::vector<std::vector<int>> negatively_strided_slices;
  std::vector<int> out_shape(a.ndim());
  for (int i = 0; i < a.ndim(); ++i) {
    // Following numpy docs
    //  Negative i and j are interpreted as n + i and n + j where n is
    //  the number of elements in the corresponding dimension. Negative
    //  k makes stepping go towards smaller indices

    auto n = a.shape(i);
    auto s = start[i];
    s = s < 0 ? s + n : s;
    auto e = stop[i];
    e = e < 0 ? e + n : e;

    // Note: We pass positive strides to the primitive and then flip
    //       the axes later as needed
    if (strides[i] < 0) {
      negatively_strided_axes.push_back(i);
      auto st = std::min(s, n - 1);
      auto ed = std::max(e, -1);
      negatively_strided_slices.push_back({st, ed, strides[i]});
      start[i] = 0;
      stop[i] = n;
      strides[i] = 1;
    } else {
      start[i] = s;
      stop[i] = e < s ? s : e;
    }

    // Clamp to bounds
    start[i] = std::max(0, std::min(start[i], n));
    stop[i] = std::max(0, std::min(stop[i], n));

    out_shape[i] = (stop[i] - start[i] + strides[i] - 1) / strides[i];
  }

  // If strides are negative, slice and then make a copy with axes flipped
  if (negatively_strided_axes.size() > 0) {
    // First, take the slice of the positvely strided axes
    auto out = array(
        out_shape,
        a.dtype(),
        std::make_unique<Slice>(
            to_stream(s),
            std::move(start),
            std::move(stop),
            std::move(strides)),
        {a});

    std::vector<array> indices;
    std::vector<int> slice_sizes = out.shape();
    std::vector<int> t_axes(out.ndim(), -1);
    std::vector<int> out_reshape(out.ndim(), -1);

    int n_axes = negatively_strided_axes.size();
    for (int i = 0; i < n_axes; i++) {
      // Get axis and corresponding slice
      auto ax = negatively_strided_axes[i];
      auto sl = negatively_strided_slices[i];

      // Get indices for the slice
      auto ax_idx = arange(sl[0], sl[1], sl[2], s);

      // Reshape indices for broadcast as needed
      std::vector<int> ax_idx_shape(n_axes, 1);
      ax_idx_shape[i] = ax_idx.size();
      ax_idx = reshape(ax_idx, ax_idx_shape, s);

      // Add indices to list
      indices.push_back(ax_idx);

      // Set slice size for axis
      slice_sizes[ax] = 1;

      // Gather moves the axis up, remainder needs to be squeezed
      out_reshape[i] = indices[i].size();

      // Gather moves the axis up, needs to be tranposed
      t_axes[ax] = i;
    }

    // Prepare out_reshape to squeeze gathered dims
    // Prepare to transpose dims as needed
    int j = n_axes;
    for (int i = 0; j < out.ndim() && i < out.ndim(); i++) {
      if (t_axes[i] < 0) {
        t_axes[i] = j;
        out_reshape[j] = out_shape[i];
        j++;
      }
    }

    // Gather
    out = gather(out, indices, negatively_strided_axes, slice_sizes, s);

    // Squeeze dims
    out = reshape(out, out_reshape, s);

    // Transpose dims
    out = transpose(out, t_axes, s);

    return out;
  }
  if (out_shape == a.shape()) {
    return a;
  }
  return array(
      out_shape,
      a.dtype(),
      std::make_unique<Slice>(
          to_stream(s), std::move(start), std::move(stop), std::move(strides)),
      {a});
}

array slice(
    const array& a,
    const std::vector<int>& start,
    const std::vector<int>& stop,
    StreamOrDevice s /* = {} */) {
  return slice(a, start, stop, std::vector<int>(a.ndim(), 1), to_stream(s));
}

std::vector<array> split(
    const array& a,
    const std::vector<int>& indices,
    int axis,
    StreamOrDevice s /* = {} */) {
  auto ax = axis < 0 ? axis + a.ndim() : axis;
  if (ax < 0 || ax >= a.ndim()) {
    std::ostringstream msg;
    msg << "Invalid axis (" << axis << ") passed to split"
        << " for array with shape " << a.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  std::vector<array> res;
  auto out_shape = a.shape();
  auto start_indices = std::vector<int>(a.ndim(), 0);
  auto stop_indices = a.shape();
  for (int i = 0; i < indices.size() + 1; ++i) {
    stop_indices[ax] = i < indices.size() ? indices[i] : a.shape(ax);
    res.push_back(slice(a, start_indices, stop_indices, to_stream(s)));
    start_indices[ax] = stop_indices[ax];
  }
  return res;
}

std::vector<array> split(
    const array& a,
    const std::vector<int>& indices,
    StreamOrDevice s /* = {} */) {
  return split(a, indices, 0, s);
}

std::vector<array>
split(const array& a, int num_splits, int axis, StreamOrDevice s /* = {} */) {
  auto q_and_r = std::ldiv(a.shape(axis), num_splits);
  if (q_and_r.rem) {
    std::ostringstream msg;
    msg << "Array split does not result in sub arrays with equal size:"
        << " attempting " << num_splits << " splits along axis " << axis
        << " for shape " << a.shape() << ".";
    throw std::invalid_argument(msg.str());
  }
  auto split_size = q_and_r.quot;
  std::vector<int> indices(num_splits - 1);
  for (int i = 0; i < indices.size(); ++i) {
    indices[i] = (i + 1) * split_size;
  }
  return split(a, indices, axis, s);
}

std::vector<array>
split(const array& a, int num_splits, StreamOrDevice s /* = {} */) {
  return split(a, num_splits, 0, to_stream(s));
}

array clip(
    const array& a,
    const std::optional<array>& a_min,
    const std::optional<array>& a_max,
    StreamOrDevice s /* = {} */) {
  if (!a_min.has_value() && !a_max.has_value()) {
    throw std::invalid_argument("At most one of a_min and a_max may be None");
  }
  array result = astype(a, a.dtype(), s);
  if (a_min.has_value()) {
    result = maximum(result, a_min.value(), s);
  }
  if (a_max.has_value()) {
    result = minimum(result, a_max.value(), s);
  }
  return result;
}

array concatenate(
    const std::vector<array>& arrays,
    int axis,
    StreamOrDevice s /* = {} */) {
  if (arrays.size() == 0) {
    throw std::invalid_argument("No arrays provided for concatenation");
  }

  // Normalize the given axis
  auto ax = axis < 0 ? axis + arrays[0].ndim() : axis;
  if (ax < 0 || ax >= arrays[0].ndim()) {
    std::ostringstream msg;
    msg << "Invalid axis (" << axis << ") passed to concatenate"
        << " for array with shape " << arrays[0].shape() << ".";
    throw std::invalid_argument(msg.str());
  }

  auto throw_invalid_shapes = [&]() {
    std::ostringstream msg;
    msg << "All the input array dimensions must match exactly except"
        << " for the concatenation axis. However, the provided shapes are ";
    for (auto& a : arrays) {
      msg << a.shape() << ", ";
    }
    msg << "and the concatenation axis is " << axis;
    throw std::invalid_argument(msg.str());
  };

  std::vector<int> shape = arrays[0].shape();
  shape[ax] = 0;
  // Make the output shape and validate that all arrays have the same shape
  // except for the concatenation axis.
  for (auto& a : arrays) {
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

  return array(
      shape, dtype, std::make_unique<Concatenate>(to_stream(s), ax), arrays);
}

array concatenate(
    const std::vector<array>& arrays,
    StreamOrDevice s /* = {} */) {
  std::vector<array> flat_inputs;
  for (auto& a : arrays) {
    flat_inputs.push_back(reshape(a, {-1}, s));
  }
  return concatenate(flat_inputs, 0, s);
}

/** Stack arrays along a new axis */
array stack(
    const std::vector<array>& arrays,
    int axis,
    StreamOrDevice s /* = {} */) {
  if (arrays.empty()) {
    throw std::invalid_argument("No arrays provided for stacking");
  }
  if (!is_same_shape(arrays)) {
    throw std::invalid_argument("All arrays must have the same shape");
  }
  int normalized_axis = normalize_axis(axis, arrays[0].ndim() + 1);
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

/** Pad an array with a constant value */
array pad(
    const array& a,
    const std::vector<int>& axes,
    const std::vector<int>& low_pad_size,
    const std::vector<int>& high_pad_size,
    const array& pad_value /*= array(0)*/,
    StreamOrDevice s /* = {}*/) {
  if (axes.size() != low_pad_size.size() ||
      axes.size() != high_pad_size.size()) {
    std::ostringstream msg;
    msg << "Invalid number of padding sizes passed to pad "
        << "with axes of size " << axes.size();
    throw std::invalid_argument(msg.str());
  }

  std::vector<int> out_shape = a.shape();

  for (int i = 0; i < axes.size(); i++) {
    if (low_pad_size[i] < 0) {
      std::ostringstream msg;
      msg << "Invalid low padding size (" << low_pad_size[i]
          << ") passed to pad"
          << " for axis " << i << ". Padding sizes must be non-negative";
      throw std::invalid_argument(msg.str());
    }
    if (high_pad_size[i] < 0) {
      std::ostringstream msg;
      msg << "Invalid high padding size (" << high_pad_size[i]
          << ") passed to pad"
          << " for axis " << i << ". Padding sizes must be non-negative";
      throw std::invalid_argument(msg.str());
    }

    auto ax = axes[i] < 0 ? a.ndim() + axes[i] : axes[i];
    out_shape[ax] += low_pad_size[i] + high_pad_size[i];
  }

  return array(
      out_shape,
      a.dtype(),
      std::make_unique<Pad>(to_stream(s), axes, low_pad_size, high_pad_size),
      {a, astype(pad_value, a.dtype(), s)});
}

/** Pad an array with a constant value along all axes */
array pad(
    const array& a,
    const std::vector<std::pair<int, int>>& pad_width,
    const array& pad_value /*= array(0)*/,
    StreamOrDevice s /*= {}*/) {
  std::vector<int> axes(a.ndim(), 0);
  std::iota(axes.begin(), axes.end(), 0);

  std::vector<int> lows;
  std::vector<int> highs;

  for (auto& pads : pad_width) {
    lows.push_back(pads.first);
    highs.push_back(pads.second);
  }

  return pad(a, axes, lows, highs, pad_value, s);
}

array pad(
    const array& a,
    const std::pair<int, int>& pad_width,
    const array& pad_value /*= array(0)*/,
    StreamOrDevice s /*= {}*/) {
  return pad(
      a, std::vector<std::pair<int, int>>(a.ndim(), pad_width), pad_value, s);
}

array pad(
    const array& a,
    int pad_width,
    const array& pad_value /*= array(0)*/,
    StreamOrDevice s /*= {}*/) {
  return pad(
      a,
      std::vector<std::pair<int, int>>(a.ndim(), {pad_width, pad_width}),
      pad_value,
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
  return transpose(a, reorder, s);
}

array transpose(
    const array& a,
    std::vector<int> axes,
    StreamOrDevice s /* = {} */) {
  for (auto& ax : axes) {
    ax = ax < 0 ? ax + a.ndim() : ax;
  }
  std::set dims(axes.begin(), axes.end());
  if (dims.size() != axes.size()) {
    throw std::invalid_argument("Repeat axes not allowed in transpose.");
  }
  if (dims.size() != a.ndim() ||
      a.ndim() > 0 &&
          (*dims.begin() != 0 || *dims.rbegin() != (a.ndim() - 1))) {
    throw std::invalid_argument("Transpose axes don't match array dimensions.");
  }
  std::vector<int> shape;
  shape.reserve(axes.size());
  for (auto ax : axes) {
    shape.push_back(a.shape()[ax]);
  }
  return array(
      shape,
      a.dtype(),
      std::make_unique<Transpose>(to_stream(s), std::move(axes)),
      {a});
}

array transpose(const array& a, StreamOrDevice s /* = {} */) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.rbegin(), axes.rend(), 0);
  return transpose(a, std::move(axes), to_stream(s));
}

array broadcast_to(
    const array& a,
    const std::vector<int>& shape,
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
      shape, a.dtype(), std::make_unique<Broadcast>(to_stream(s), shape), {a});
}

std::vector<array> broadcast_arrays(
    const std::vector<array>& inputs,
    StreamOrDevice s /* = {} */) {
  std::vector<int> shape{};
  for (const auto& in : inputs) {
    shape = broadcast_shapes(shape, in.shape());
  }
  std::vector<array> outputs;
  for (const auto& in : inputs) {
    outputs.push_back(broadcast_to(in, shape, s));
  }
  return outputs;
}

array equal(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  std::vector<array> inputs = {astype(a, dtype, s), astype(b, dtype, s)};
  if (a.shape() != b.shape()) {
    inputs = broadcast_arrays(inputs, s);
  }
  return array(
      inputs[0].shape(), bool_, std::make_unique<Equal>(to_stream(s)), inputs);
}

array not_equal(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  std::vector<array> inputs = {astype(a, dtype, s), astype(b, dtype, s)};
  if (a.shape() != b.shape()) {
    inputs = broadcast_arrays(inputs, s);
  }
  return array(
      inputs[0].shape(),
      bool_,
      std::make_unique<NotEqual>(to_stream(s)),
      inputs);
}

array greater(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  std::vector<array> inputs = {astype(a, dtype, s), astype(b, dtype, s)};
  if (a.shape() != b.shape()) {
    inputs = broadcast_arrays(inputs, s);
  }
  return array(
      inputs[0].shape(),
      bool_,
      std::make_unique<Greater>(to_stream(s)),
      inputs);
}

array greater_equal(
    const array& a,
    const array& b,
    StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  std::vector<array> inputs = {astype(a, dtype, s), astype(b, dtype, s)};
  if (a.shape() != b.shape()) {
    inputs = broadcast_arrays(inputs, s);
  }
  return array(
      inputs[0].shape(),
      bool_,
      std::make_unique<GreaterEqual>(to_stream(s)),
      inputs);
}

array less(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  std::vector<array> inputs = {astype(a, dtype, s), astype(b, dtype, s)};
  if (a.shape() != b.shape()) {
    inputs = broadcast_arrays(inputs, s);
  }
  return array(
      inputs[0].shape(), bool_, std::make_unique<Less>(to_stream(s)), inputs);
}

array less_equal(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  std::vector<array> inputs = {astype(a, dtype, s), astype(b, dtype, s)};
  if (a.shape() != b.shape()) {
    inputs = broadcast_arrays(inputs, s);
  }
  return array(
      inputs[0].shape(),
      bool_,
      std::make_unique<LessEqual>(to_stream(s)),
      inputs);
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
    equal_nan &= is_floating_point(dtype);
    return all(
        array(
            a.shape(),
            bool_,
            std::make_unique<Equal>(to_stream(s), equal_nan),
            {astype(a, dtype, s), astype(b, dtype, s)}),
        false,
        s);
  }
}

array where(
    const array& condition,
    const array& x,
    const array& y,
    StreamOrDevice s /* = {} */) {
  // TODO, fix this to handle the NaN case when x has infs
  auto mask = astype(condition, bool_, s);
  return add(multiply(x, mask, s), multiply(y, logical_not(mask, s), s), s);
}

array allclose(
    const array& a,
    const array& b,
    double rtol /* = 1e-5 */,
    double atol /* = 1e-8 */,
    StreamOrDevice s /* = {}*/) {
  // |a - b| <= atol + rtol * |b|
  auto rhs = add(array(atol), multiply(array(rtol), abs(b, s), s), s);
  auto lhs = abs(subtract(a, b, s), s);
  return all(less_equal(lhs, rhs, s), s);
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
  if (axes.empty()) {
    return astype(a, bool_, s);
  }
  auto [out_shape, sorted_axes] =
      compute_reduce_shape(axes, a.shape(), keepdims);
  return array(
      out_shape,
      bool_,
      std::make_unique<Reduce>(to_stream(s), Reduce::And, sorted_axes),
      {a});
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
  if (axes.empty()) {
    return astype(a, bool_, s);
  }
  auto [out_shape, sorted_axes] =
      compute_reduce_shape(axes, a.shape(), keepdims);
  return array(
      out_shape,
      bool_,
      std::make_unique<Reduce>(to_stream(s), Reduce::Or, sorted_axes),
      {a});
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
  auto [out_shape, sorted_axes] =
      compute_reduce_shape(axes, a.shape(), keepdims);
  auto out_type = a.dtype() == bool_ ? int32 : a.dtype();
  return array(
      out_shape,
      out_type,
      std::make_unique<Reduce>(to_stream(s), Reduce::Sum, sorted_axes),
      {a});
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
  auto nelements = compute_number_of_elements(a, axes);
  auto dtype = at_least_float(a.dtype());
  return multiply(sum(a, axes, keepdims, s), array(1.0 / nelements, dtype), s);
}

array mean(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  return mean(a, std::vector<int>{axis}, keepdims, to_stream(s));
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
  auto nelements = compute_number_of_elements(a, axes);
  auto dtype = at_least_float(a.dtype());
  auto mu = mean(a, axes, true, s);
  auto S = sum(square(subtract(a, mu, s), s), axes, keepdims, s);
  return multiply(S, array(1.0 / (nelements - ddof), dtype), s);
}

array var(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    int ddof /* = 0*/,
    StreamOrDevice s /* = {} */) {
  return var(a, std::vector<int>{axis}, keepdims, ddof, to_stream(s));
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
  auto [out_shape, sorted_axes] =
      compute_reduce_shape(axes, a.shape(), keepdims);
  return array(
      out_shape,
      a.dtype(),
      std::make_unique<Reduce>(to_stream(s), Reduce::Prod, sorted_axes),
      {a});
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
  if (axes.empty()) {
    return a;
  }
  auto [out_shape, sorted_axes] =
      compute_reduce_shape(axes, a.shape(), keepdims);
  return array(
      out_shape,
      a.dtype(),
      std::make_unique<Reduce>(to_stream(s), Reduce::Max, sorted_axes),
      {a});
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
  auto [out_shape, sorted_axes] =
      compute_reduce_shape(axes, a.shape(), keepdims);
  return array(
      out_shape,
      a.dtype(),
      std::make_unique<Reduce>(to_stream(s), Reduce::Min, sorted_axes),
      {a});
}

array min(
    const array& a,
    int axis,
    bool keepdims /* = false */,
    StreamOrDevice s /* = {} */) {
  return min(a, std::vector<int>{axis}, keepdims, s);
}

array argmin(const array& a, bool keepdims, StreamOrDevice s /* = {} */) {
  int size = a.size();
  auto result = argmin(reshape(a, {size}, s), 0, false, s);
  if (keepdims) {
    result = reshape(result, std::vector<int>(a.shape().size(), 1), s);
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
  auto [out_shape, sorted_axes] =
      compute_reduce_shape({axis}, a.shape(), keepdims);
  return array(
      out_shape,
      uint32,
      std::make_unique<ArgReduce>(
          to_stream(s), ArgReduce::ArgMin, sorted_axes[0]),
      {a});
}

array argmax(const array& a, bool keepdims, StreamOrDevice s /* = {} */) {
  int size = a.size();
  auto result = argmax(reshape(a, {size}, s), 0, false, s);
  if (keepdims) {
    result = reshape(result, std::vector<int>(a.shape().size(), 1), s);
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
  auto [out_shape, sorted_axes] =
      compute_reduce_shape({axis}, a.shape(), keepdims);
  return array(
      out_shape,
      uint32,
      std::make_unique<ArgReduce>(
          to_stream(s), ArgReduce::ArgMax, sorted_axes[0]),
      {a});
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

  // TODO: Fix GPU kernel
  if (a.shape(axis) >= (1u << 21) && to_stream(s).device.type == Device::gpu) {
    std::ostringstream msg;
    msg << "[sort] GPU sort cannot handle sort axis of >= 2M elements,"
        << " got array with sort axis size " << a.shape(axis) << "."
        << " Please place this operation on the CPU instead.";
    throw std::runtime_error(msg.str());
  }

  return array(
      a.shape(), a.dtype(), std::make_unique<Sort>(to_stream(s), axis), {a});
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

  // TODO: Fix GPU kernel
  if (a.shape(axis) >= (1u << 21) && to_stream(s).device.type == Device::gpu) {
    std::ostringstream msg;
    msg << "[argsort] GPU sort cannot handle sort axis of >= 2M elements,"
        << " got array with sort axis size " << a.shape(axis) << "."
        << " Please place this operation on the CPU instead.";
    throw std::runtime_error(msg.str());
  }

  return array(
      a.shape(), uint32, std::make_unique<ArgSort>(to_stream(s), axis), {a});
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
      std::make_unique<Partition>(to_stream(s), kth_, axis_),
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
    msg << "[argpartition] Received invalid kth " << kth << "along axis "
        << axis << " for array with shape: " << a.shape();
    throw std::invalid_argument(msg.str());
  }
  return array(
      a.shape(),
      uint32,
      std::make_unique<ArgPartition>(to_stream(s), kth_, axis_),
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
  int kth_ = k < 0 ? k + a.shape(axis) : k;
  if (axis_ < 0 || axis_ >= static_cast<int>(a.ndim())) {
    std::ostringstream msg;
    msg << "[topk] Received invalid axis " << axis << " for array with "
        << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (kth_ < 0 || kth_ >= a.shape(axis_)) {
    std::ostringstream msg;
    msg << "[topk] Received invalid k " << k << "along axis " << axis
        << " for array with shape: " << a.shape();
    throw std::invalid_argument(msg.str());
  }

  array a_partitioned = partition(a, kth_, axis_, s);
  std::vector<int> slice_starts(a.ndim(), 0);
  std::vector<int> slice_ends = a.shape();
  slice_starts[axis_] = kth_;
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
  auto maxval = stop_gradient(max(a, axes, true, s));
  auto out = log(sum(exp(subtract(a, maxval, s), s), axes, keepdims, s), s);
  return add(out, reshape(maxval, out.shape(), s), s);
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
      array(a.shape(), a.dtype(), std::make_unique<Abs>(to_stream(s)), {a});
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
      a.shape(), a.dtype(), std::make_unique<Negative>(to_stream(s)), {a});
}
array operator-(const array& a) {
  return negative(a);
}

array sign(const array& a, StreamOrDevice s /* = {} */) {
  return array(a.shape(), a.dtype(), std::make_unique<Sign>(to_stream(s)), {a});
}

array logical_not(const array& a, StreamOrDevice s /* = {} */) {
  return array(
      a.shape(),
      bool_,
      std::make_unique<LogicalNot>(to_stream(s)),
      {astype(a, bool_, s)});
}

array reciprocal(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  return divide(array(1.0f, dtype), a, to_stream(s));
}

array add(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto out_type = promote_types(a.dtype(), b.dtype());
  auto inputs =
      broadcast_arrays({astype(a, out_type, s), astype(b, out_type, s)}, s);
  return array(
      inputs[0].shape(), out_type, std::make_unique<Add>(to_stream(s)), inputs);
}

array operator+(const array& a, const array& b) {
  return add(a, b);
}

array subtract(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto out_type = promote_types(a.dtype(), b.dtype());
  auto inputs =
      broadcast_arrays({astype(a, out_type, s), astype(b, out_type, s)}, s);
  return array(
      inputs[0].shape(),
      out_type,
      std::make_unique<Subtract>(to_stream(s)),
      inputs);
}

array operator-(const array& a, const array& b) {
  return subtract(a, b);
}

array multiply(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto out_type = promote_types(a.dtype(), b.dtype());
  auto inputs =
      broadcast_arrays({astype(a, out_type, s), astype(b, out_type, s)}, s);
  return array(
      inputs[0].shape(),
      out_type,
      std::make_unique<Multiply>(to_stream(s)),
      inputs);
}

array operator*(const array& a, const array& b) {
  return multiply(a, b);
}

array divide(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(promote_types(a.dtype(), b.dtype()));
  auto inputs = broadcast_arrays(
      {astype(a, dtype, s), astype(b, dtype, to_stream(s))}, s);
  return array(
      inputs[0].shape(), dtype, std::make_unique<Divide>(to_stream(s)), inputs);
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
  if (is_floating_point(dtype)) {
    return floor(divide(a, b, s), s);
  }

  auto inputs = broadcast_arrays({astype(a, dtype, s), astype(b, dtype, s)}, s);
  return array(
      inputs[0].shape(), dtype, std::make_unique<Divide>(to_stream(s)), inputs);
}

array remainder(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  auto inputs = broadcast_arrays(
      {astype(a, dtype, s), astype(b, dtype, to_stream(s))}, s);
  return array(
      inputs[0].shape(),
      dtype,
      std::make_unique<Remainder>(to_stream(s)),
      inputs);
}
array operator%(const array& a, const array& b) {
  return remainder(a, b);
}

array maximum(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto out_type = promote_types(a.dtype(), b.dtype());
  auto inputs =
      broadcast_arrays({astype(a, out_type, s), astype(b, out_type, s)}, s);
  return array(
      inputs[0].shape(),
      out_type,
      std::make_unique<Maximum>(to_stream(s)),
      inputs);
}

array minimum(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto out_type = promote_types(a.dtype(), b.dtype());
  auto inputs =
      broadcast_arrays({astype(a, out_type, s), astype(b, out_type, s)}, s);
  return array(
      inputs[0].shape(),
      out_type,
      std::make_unique<Minimum>(to_stream(s)),
      inputs);
}

array floor(const array& a, StreamOrDevice s /* = {} */) {
  if (a.dtype() == complex64) {
    throw std::invalid_argument("[floor] Not supported for complex64.");
  }
  return array(
      a.shape(), a.dtype(), std::make_unique<Floor>(to_stream(s)), {a});
}

array ceil(const array& a, StreamOrDevice s /* = {} */) {
  if (a.dtype() == complex64) {
    throw std::invalid_argument("[floor] Not supported for complex64.");
  }
  return array(a.shape(), a.dtype(), std::make_unique<Ceil>(to_stream(s)), {a});
}

array square(const array& a, StreamOrDevice s /* = {} */) {
  return array(
      a.shape(), a.dtype(), std::make_unique<Square>(to_stream(s)), {a});
}

array exp(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(a.shape(), dtype, std::make_unique<Exp>(to_stream(s)), {input});
}

array sin(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(a.shape(), dtype, std::make_unique<Sin>(to_stream(s)), {input});
}

array cos(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(a.shape(), dtype, std::make_unique<Cos>(to_stream(s)), {input});
}

array tan(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(a.shape(), dtype, std::make_unique<Tan>(to_stream(s)), {input});
}

array arcsin(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_unique<ArcSin>(to_stream(s)), {input});
}

array arccos(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_unique<ArcCos>(to_stream(s)), {input});
}

array arctan(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_unique<ArcTan>(to_stream(s)), {input});
}

array sinh(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(a.shape(), dtype, std::make_unique<Sinh>(to_stream(s)), {input});
}

array cosh(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(a.shape(), dtype, std::make_unique<Cosh>(to_stream(s)), {input});
}

array tanh(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(a.shape(), dtype, std::make_unique<Tanh>(to_stream(s)), {input});
}

array arcsinh(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_unique<ArcSinh>(to_stream(s)), {input});
}

array arccosh(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_unique<ArcCosh>(to_stream(s)), {input});
}

array arctanh(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_unique<ArcTanh>(to_stream(s)), {input});
}

array log(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(),
      dtype,
      std::make_unique<Log>(to_stream(s), Log::Base::e),
      {input});
}

array log2(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(),
      dtype,
      std::make_unique<Log>(to_stream(s), Log::Base::two),
      {input});
}

array log10(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(),
      dtype,
      std::make_unique<Log>(to_stream(s), Log::Base::ten),
      {input});
}

array log1p(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_unique<Log1p>(to_stream(s)), {input});
}

array logaddexp(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  // Make sure out type is floating point
  auto out_type = at_least_float(promote_types(a.dtype(), b.dtype()));
  auto inputs =
      broadcast_arrays({astype(a, out_type, s), astype(b, out_type, s)}, s);
  return array(
      inputs[0].shape(),
      out_type,
      std::make_unique<LogAddExp>(to_stream(s)),
      inputs);
}

array sigmoid(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  auto input = astype(a, dtype, s);
  return array(
      a.shape(), dtype, std::make_unique<Sigmoid>(to_stream(s)), {input});
}

array erf(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  return array(
      a.shape(),
      dtype,
      std::make_unique<Erf>(to_stream(s)),
      {astype(a, dtype, s)});
}

array erfinv(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  return array(
      a.shape(),
      dtype,
      std::make_unique<ErfInv>(to_stream(s)),
      {astype(a, dtype, s)});
}

array stop_gradient(const array& a, StreamOrDevice s /* = {} */) {
  return array(
      a.shape(), a.dtype(), std::make_unique<StopGradient>(to_stream(s)), {a});
}

array round(const array& a, int decimals, StreamOrDevice s /* = {} */) {
  if (decimals == 0) {
    return array(
        a.shape(), a.dtype(), std::make_unique<Round>(to_stream(s)), {a});
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
    a = reshape(a, {1, -1}, s);
  }
  if (b.ndim() == 1) {
    // Insert a singleton dim at the end
    b = reshape(b, {-1, 1}, s);
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
  if (a.dtype() != out_type) {
    a = astype(a, out_type, s);
  }
  if (b.dtype() != out_type) {
    b = astype(b, out_type, s);
  }

  // We can batch the multiplication by reshaping a
  if (a.ndim() > 2 && b.ndim() == 2) {
    std::vector<int> out_shape = a.shape();
    a = reshape(a, {-1, out_shape.back()}, s);
    out_shape.back() = b.shape(-1);
    if (in_b.ndim() == 1) {
      out_shape.pop_back();
    }
    auto out = array(
        {a.shape(0), b.shape(1)},
        out_type,
        std::make_unique<Matmul>(to_stream(s)),
        {a, b});
    return reshape(out, out_shape, s);
  }

  if (a.ndim() > 2 || b.ndim() > 2) {
    std::vector<int> bsx_a(a.shape().begin(), a.shape().end() - 2);
    std::vector<int> bsx_b(b.shape().begin(), b.shape().end() - 2);
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

  auto out = array(
      out_shape, out_type, std::make_unique<Matmul>(to_stream(s)), {a, b});

  // Remove the possibly inserted singleton dimensions
  if (in_a.ndim() == 1 || in_b.ndim() == 1) {
    out_shape.erase(
        out_shape.end() - ((in_a.ndim() == 1) ? 2 : 1),
        out_shape.end() - ((in_b.ndim() == 1) ? 0 : 1));
    out = reshape(out, out_shape, s);
  }
  return out;
}

array gather(
    const array& a,
    const std::vector<array>& indices,
    const std::vector<int>& axes,
    const std::vector<int>& slice_sizes,
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
      throw("[Gather] Boolean indices not supported.");
    }
  }

  if (slice_sizes.size() != a.ndim()) {
    std::ostringstream msg;
    msg << "[gather] Got slice_sizes with size " << slice_sizes.size()
        << " for array with " << a.ndim() << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  for (int i = 0; i < a.ndim(); ++i) {
    if (slice_sizes[i] < 0 || slice_sizes[i] > a.shape(i)) {
      std::ostringstream msg;
      msg << "[gather] Slice sizes must be in [0, a.shape(i)]. Got "
          << slice_sizes << " for array with shape " << a.shape() << ".";
      throw std::invalid_argument(msg.str());
    }
  }

  // Promote indices to the same type
  auto dtype = result_type(indices);
  if (!is_integral(dtype)) {
    throw std::invalid_argument(
        "[gather] Got indices with invalid dtype. Indices must be integral.");
  }

  // Broadcast and cast indices if necessary
  auto inputs = broadcast_arrays(indices);
  for (auto& idx : inputs) {
    idx = astype(idx, dtype, s);
  }

  std::vector<int> out_shape;
  if (!inputs.empty()) {
    out_shape = inputs[0].shape();
  }
  out_shape.insert(out_shape.end(), slice_sizes.begin(), slice_sizes.end());

  inputs.insert(inputs.begin(), a);
  return array(
      out_shape,
      a.dtype(),
      std::make_unique<Gather>(to_stream(s), axes, slice_sizes),
      inputs);
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
  if (a.size() == 0 && indices.size() != 0) {
    throw std::invalid_argument(
        "[take] Cannot do a non-empty take from an array with zero elements.");
  }

  // Handle negative axis
  axis = axis < 0 ? a.ndim() + axis : axis;

  // Make slice sizes to pass to gather
  std::vector<int> slice_sizes = a.shape();
  slice_sizes[axis] = indices.size() > 0 ? 1 : 0;

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
  std::vector<int> out_shape = out.shape();
  out_shape.erase(out_shape.begin() + indices.ndim() + axis);
  return reshape(out, out_shape, s);
}

array take(const array& a, const array& indices, StreamOrDevice s /* = {} */) {
  return take(reshape(a, {-1}, s), indices, 0, s);
}

array take_along_axis(
    const array& a,
    const array& indices,
    int axis,
    StreamOrDevice s /* = {} */) {
  if (axis + a.ndim() < 0 || axis >= static_cast<int>(a.ndim())) {
    std::ostringstream msg;
    msg << "[take_along_axis] Received invalid axis "
        << " for array with " << a.ndim() << " dimensions.";
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

  std::vector<array> nd_indices;
  std::vector<int> index_shape(a.ndim(), 1);
  for (int i = 0; i < a.ndim(); ++i) {
    if (i == axis) {
      nd_indices.push_back(indices);
    } else {
      // Reshape so they can be broadcast
      index_shape[i] = a.shape(i);
      nd_indices.push_back(reshape(arange(a.shape(i), s), index_shape, s));
      index_shape[i] = 1;
    }
  }
  std::vector<int> dims(a.ndim());
  std::iota(dims.begin(), dims.end(), 0);
  std::vector<int> slice_sizes(a.ndim(), a.size() > 0);
  auto out = gather(a, nd_indices, dims, slice_sizes, s);

  // Squeeze out the slice shape
  std::vector<int> out_shape(
      out.shape().begin(), out.shape().begin() + a.ndim());
  return reshape(out, out_shape, s);
}

/** Scatter updates to given indices */
array scatter(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    Scatter::ReduceType mode /*= Scatter::ReduceType::None*/,
    StreamOrDevice s /*= {}*/) {
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

  std::vector<int> idx_shape;
  if (!inputs.empty()) {
    idx_shape = inputs[0].shape();
  }

  if (updates.ndim() != (a.ndim() + idx_shape.size())) {
    std::ostringstream msg;
    msg << "[scatter] Updates with " << updates.ndim()
        << " dimensions does not match the sum of the array and indices "
           "dimensions "
        << a.ndim() + idx_shape.size() << ".";
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
  if (!is_integral(dtype)) {
    throw std::invalid_argument(
        "[scatter] Got indices with invalid dtype. Indices must be integral.");
  }
  for (auto& idx : inputs) {
    idx = astype(idx, dtype, s);
  }

  inputs.insert(inputs.begin(), a);
  // TODO promote or cast?
  inputs.push_back(astype(updates, a.dtype(), s));
  return array(
      a.shape(),
      a.dtype(),
      std::make_unique<Scatter>(to_stream(s), mode, axes),
      inputs);
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

array sqrt(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  return array(
      a.shape(),
      dtype,
      std::make_unique<Sqrt>(to_stream(s)),
      {astype(a, dtype, s)});
}

array rsqrt(const array& a, StreamOrDevice s /* = {} */) {
  auto dtype = at_least_float(a.dtype());
  return array(
      a.shape(),
      dtype,
      std::make_unique<Sqrt>(to_stream(s), true),
      {astype(a, dtype, s)});
}

array softmax(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {}*/) {
  if (axes.size() == 1 && (a.ndim() == axes[0] + 1 || axes[0] == -1)) {
    auto dtype = at_least_float(a.dtype());
    return array(
        a.shape(),
        dtype,
        std::make_unique<Softmax>(to_stream(s)),
        {astype(a, dtype, s)});
  } else {
    auto a_max = stop_gradient(max(a, axes, /*keepdims = */ true, s), s);
    auto ex = exp(subtract(a, a_max, s), s);
    return divide(ex, sum(ex, axes, /*keepdims = */ true, s), s);
  }
}

array softmax(const array& a, StreamOrDevice s /* = {}*/) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return softmax(a, axes, s);
}

array power(const array& a, const array& b, StreamOrDevice s /* = {} */) {
  auto dtype = promote_types(a.dtype(), b.dtype());
  std::vector<array> inputs = {astype(a, dtype, s), astype(b, dtype, s)};
  if (a.shape() != b.shape()) {
    inputs = broadcast_arrays(inputs, s);
  }
  return array(
      inputs[0].shape(), dtype, std::make_unique<Power>(to_stream(s)), inputs);
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
      std::make_unique<Scan>(
          to_stream(s), Scan::ReduceType::Sum, axis, reverse, inclusive),
      {a});
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
      std::make_unique<Scan>(
          to_stream(s), Scan::ReduceType::Prod, axis, reverse, inclusive),
      {a});
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
      std::make_unique<Scan>(
          to_stream(s), Scan::ReduceType::Max, axis, reverse, inclusive),
      {a});
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
      std::make_unique<Scan>(
          to_stream(s), Scan::ReduceType::Min, axis, reverse, inclusive),
      {a});
}

/** Convolution operations */

namespace {

// Conv helpers
inline int conv_out_axis_size(
    int in_dim,
    int wt_dim,
    int stride,
    int padding,
    int dilation) {
  int ker = dilation * (wt_dim - 1);
  return ((in_dim + 2 * padding - ker - 1) / stride) + 1;
}

inline std::vector<int> conv_out_shape(
    const std::vector<int>& in_shape,
    const std::vector<int>& wt_shape,
    const std::vector<int>& strides,
    const std::vector<int>& pads,
    const std::vector<int>& dilation) {
  int N = in_shape[0];
  int O = wt_shape[0];
  std::vector<int> out_shape(in_shape.size());
  int i = 0;
  out_shape[i++] = N;
  for (; i < in_shape.size() - 1; i++) {
    out_shape[i] = conv_out_axis_size(
        in_shape[i], wt_shape[i], strides[i - 1], pads[i - 1], dilation[i - 1]);
  }
  out_shape[i] = O;

  return out_shape;
}

inline void run_conv_checks(const array& in, const array& wt, int n_dim) {
  if (!is_floating_point(in.dtype()) && kindof(in.dtype()) != Dtype::Kind::c) {
    std::ostringstream msg;
    msg << "[conv] Invalid input array with type " << in.dtype() << "."
        << " Convolution currently only supports floating point types";
    throw std::invalid_argument(msg.str());
  }

  if (in.ndim() != n_dim + 2) {
    std::ostringstream msg;
    msg << "[conv] Invalid input array with " << in.ndim() << " dimensions for "
        << n_dim << "D convolution."
        << " Expected an array with " << n_dim + 2
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

  if (in.shape(n_dim + 1) != wt.shape(n_dim + 1)) {
    std::ostringstream msg;
    msg << "[conv] Expect the input channels in the input"
        << " and weight array to match but got shapes -"
        << " input: " << in.shape() << " and weight: " << wt.shape();
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
  // Run checks
  if (groups != 1) {
    throw std::invalid_argument("[conv1d] Cannot handle groups != 1 yet");
  }
  if (dilation != 1) {
    throw std::invalid_argument("[conv1d] Cannot handle dilation != 1 yet");
  }

  // Run checks
  run_conv_checks(in_, wt_, 1);

  auto in = in_;
  auto wt = wt_;

  // Type promotion
  auto out_type = promote_types(in.dtype(), wt.dtype());
  in = astype(in, out_type, s);
  wt = astype(wt, out_type, s);

  std::vector<int> strides_vec = {stride};
  std::vector<int> padding_vec = {padding};
  std::vector<int> dilation_vec = {dilation};

  // Get output shapes
  std::vector<int> out_shape = conv_out_shape(
      in.shape(), wt.shape(), strides_vec, padding_vec, dilation_vec);

  return array(
      out_shape,
      in.dtype(),
      std::make_unique<Convolution>(
          to_stream(s),
          padding_vec,
          strides_vec,
          dilation_vec,
          std::vector<int>(1, 1)),
      {in, wt});
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
  // Run checks
  if (groups != 1) {
    throw std::invalid_argument("[conv2d] Cannot handle groups != 1 yet");
  }
  if (dilation.first != 1 || dilation.second != 1) {
    throw std::invalid_argument("[conv1d] Cannot handle dilation != 1 yet");
  }

  // Run checks
  run_conv_checks(in_, wt_, 2);

  auto in = in_;
  auto wt = wt_;

  // Type promotion
  auto out_type = promote_types(in.dtype(), wt.dtype());
  in = astype(in, out_type, s);
  wt = astype(wt, out_type, s);

  std::vector<int> strides_vec = {stride.first, stride.second};
  std::vector<int> padding_vec = {padding.first, padding.second};
  std::vector<int> dilation_vec = {dilation.first, dilation.second};

  // Get output shapes
  std::vector<int> out_shape = conv_out_shape(
      in.shape(), wt.shape(), strides_vec, padding_vec, dilation_vec);

  return array(
      out_shape,
      in.dtype(),
      std::make_unique<Convolution>(
          to_stream(s),
          padding_vec,
          strides_vec,
          dilation_vec,
          std::vector<int>(2, 1)),
      {in, wt});
}

array quantized_matmul(
    const array& in_x,
    const array& w,
    const array& scales,
    const array& biases,
    int group_size /* = 64 */,
    int bits /* = 4 */,
    StreamOrDevice s /* = {} */) {
  auto x = in_x;

  if (w.dtype() != uint32) {
    std::ostringstream msg;
    msg << "[quantized_matmul] The weight matrix should be uint32 "
        << "but received" << w.dtype();
    throw std::invalid_argument(msg.str());
  }
  if (w.ndim() != 2) {
    std::ostringstream msg;
    msg << "[quantized_matmul] Batched quantized matmul is not supported for now "
        << "received w with shape " << w.shape();
    throw std::invalid_argument(msg.str());
  }

  // Keep x's batch dimensions to reshape it back after the matmul
  auto original_shape = x.shape();
  int x_inner_dims = original_shape.back();
  original_shape.pop_back();

  // Reshape x into a matrix if it isn't already one
  if (x.ndim() != 2) {
    x = reshape(x, {-1, x_inner_dims}, s);
  }

  int w_inner_dims = w.shape(0) * (32 / bits);
  if (w_inner_dims != x_inner_dims) {
    std::ostringstream msg;
    msg << "[quantized_matmul] Last dimension of first input with "
        << "shape (..., " << x_inner_dims
        << ") does not match the expanded first "
        << "dimension of the quantized matrix " << w_inner_dims
        << ", computed from shape " << w.shape()
        << " with group_size=" << group_size << " and bits=" << bits;
    throw std::invalid_argument(msg.str());
  }

  int n_groups = x_inner_dims / group_size;
  if (scales.shape(-1) != n_groups || biases.shape(-1) != n_groups) {
    std::ostringstream msg;
    msg << "[quantized_matmul] Scales and biases provided do not match the "
        << "quantization arguments (group_size=" << group_size
        << ", bits=" << bits << "). Expected shapes (" << w.shape(1) << ", "
        << x_inner_dims / group_size
        << "), but got scales.shape=" << scales.shape()
        << " and biases.shape=" << biases.shape();
    throw std::invalid_argument(msg.str());
  }

  auto out = array(
      {x.shape(0), w.shape(1)},
      x.dtype(),
      std::make_unique<QuantizedMatmul>(to_stream(s), group_size, bits),
      {x, w, scales, biases});

  // If needed reshape x to the original batch shape
  if (original_shape.size() != 1) {
    original_shape.push_back(w.shape(1));
    out = reshape(out, original_shape, s);
  }

  return out;
}

std::tuple<array, array, array> quantize(
    const array& w,
    int group_size /* = 64 */,
    int bits /* = 4 */,
    StreamOrDevice s /* = {} */) {
  if (w.ndim() != 2) {
    throw std::invalid_argument("[quantize] Only matrices supported for now");
  }

  if ((w.shape(0) % 32) != 0) {
    throw std::invalid_argument(
        "[quantize] All dimensions should be divisible by 32 for now");
  }

  if ((w.shape(-1) % group_size) != 0) {
    std::ostringstream msg;
    msg << "[quantize] The last dimension of the matrix needs to be divisible by "
        << "the quantization group size " << group_size
        << ". However the provided "
        << " matrix has shape " << w.shape();
    throw std::invalid_argument(msg.str());
  }

  // Compute some constants used for the quantization
  int n_bins = (1 << bits) - 1; // 2**bits - 1
  int el_per_int = 32 / bits;
  array shifts = power(array(2, uint32), arange(0, 32, bits, uint32, s), s);
  shifts = reshape(shifts, {1, 1, -1}, s);

  // Compute scales and biases
  array packed_w =
      reshape(w, {w.shape(0), w.shape(1) / group_size, group_size}, s);
  array w_max = max(packed_w, /* axis= */ -1, /* keepdims= */ true, s);
  array w_min = min(packed_w, /* axis= */ -1, /* keepdims= */ true, s);
  array delta = divide(subtract(w_max, w_min, s), array(n_bins, w.dtype()), s);
  array scales = squeeze(delta, -1, s);
  array biases = squeeze(w_min, -1, s);

  // Quantize and pack w
  packed_w =
      astype(round(divide(subtract(packed_w, w_min, s), delta, s), s), uint32);
  packed_w = reshape(packed_w, {w.shape(0), -1, el_per_int}, s);
  packed_w = sum(
      multiply(packed_w, shifts, s), /* axis= */ 2, /* keepdims= */ false, s);

  return std::make_tuple(packed_w, scales, biases);
}

array dequantize(
    const array& w,
    const array& scales,
    const array& biases,
    int group_size /* = 64 */,
    int bits /* = 4 */,
    StreamOrDevice s /* = {} */) {
  if (w.ndim() != 2 || scales.ndim() != 2 || biases.ndim() != 2) {
    throw std::invalid_argument("[dequantize] Only matrices supported for now");
  }

  if ((w.shape(0) % 32) != 0) {
    throw std::invalid_argument(
        "[dequantize] All dimensions should be divisible by 32 for now");
  }

  if (w.shape(0) != scales.shape(0) || w.shape(0) != biases.shape(0)) {
    throw std::invalid_argument(
        "[dequantize] Shape of scales and biases does not match the matrix");
  }

  if (w.dtype() != uint32) {
    throw std::invalid_argument(
        "[dequantize] The matrix should be given as a uint32");
  }

  // Compute some constants for the dequantization
  int el_per_int = 32 / bits;

  if (w.shape(1) * el_per_int != scales.shape(1) * group_size) {
    std::ostringstream msg;
    msg << "[dequantize] Shape of scales and biases does not match the matrix "
        << "given the quantization parameters. Provided matrix of shape "
        << w.shape() << " and scales/biases of shape " << scales.shape()
        << " with group_size=" << group_size << " and bits=" << bits << ".";
    throw std::invalid_argument(msg.str());
  }

  // Extract the pieces from the passed quantized matrix
  std::vector<array> parts;
  for (int start = 0; start < 32; start += bits) {
    // TODO: Implement bitwise operators for integral types
    int shift_left = 32 - (start + bits);
    int shift_right = shift_left + start;
    array p = multiply(w, array(1 << shift_left, uint32), s);
    p = floor_divide(p, array(1 << shift_right, uint32), s);
    p = expand_dims(p, -1, s);
    parts.push_back(p);
  }
  array w_full = concatenate(parts, -1, s);

  // Dequantize
  w_full = reshape(w_full, {w.shape(0), -1, group_size}, s);
  w_full = multiply(w_full, expand_dims(scales, -1, s), s);
  w_full = add(w_full, expand_dims(biases, -1, s), s);
  w_full = reshape(w_full, {w.shape(0), -1}, s);

  return w_full;
}

} // namespace mlx::core
