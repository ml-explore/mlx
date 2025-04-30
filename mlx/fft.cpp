// Copyright © 2023 Apple Inc.
#include <numeric>
#include <set>

#include "mlx/fft.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fft {

array fft_impl(
    const array& a,
    Shape n,
    const std::vector<int>& axes,
    bool real,
    bool inverse,
    StreamOrDevice s) {
  if (a.ndim() < 1) {
    throw std::invalid_argument(
        "[fftn] Requires array with at least one dimension.");
  }
  if (n.size() != axes.size()) {
    throw std::invalid_argument("[fftn] Shape and axes have different sizes.");
  }
  if (axes.empty()) {
    return a;
  }

  std::vector<size_t> valid_axes;
  for (int ax : axes) {
    valid_axes.push_back(ax < 0 ? ax + a.ndim() : ax);
  }
  std::set<int> unique_axes(valid_axes.begin(), valid_axes.end());
  if (unique_axes.size() != axes.size()) {
    std::ostringstream msg;
    msg << "[fftn] Duplicated axis received " << axes;
    throw std::invalid_argument(msg.str());
  }
  if (*unique_axes.begin() < 0 || *unique_axes.rbegin() >= a.ndim()) {
    std::ostringstream msg;
    msg << "[fftn] Invalid axis received for array with " << a.ndim()
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }

  // In the following shape manipulations there are three cases to consider:
  // 1. In a complex to complex transform (fftn / ifftn) the output
  //    and input shapes are the same.
  // 2. In a real to complex transform (rfftn) n specifies the input dims
  //    and the output dims are n[i] / 2 + 1
  // 3  In a complex to real transform (irfftn) n specifies the output dims
  //    and the input dims are n[i] / 2 + 1

  if (std::any_of(n.begin(), n.end(), [](auto i) { return i <= 0; })) {
    std::ostringstream msg;
    msg << "[fftn] Invalid FFT output size requested " << n;
    throw std::invalid_argument(msg.str());
  }

  auto in_shape = a.shape();
  for (int i = 0; i < valid_axes.size(); ++i) {
    in_shape[valid_axes[i]] = n[i];
  }
  if (real && inverse) {
    in_shape[valid_axes.back()] = n.back() / 2 + 1;
  }

  bool any_greater = false;
  bool any_less = false;
  for (int i = 0; i < in_shape.size(); ++i) {
    any_greater |= in_shape[i] > a.shape()[i];
    any_less |= in_shape[i] < a.shape()[i];
  }

  auto in = a;
  if (any_less) {
    in = slice(in, Shape(in.ndim(), 0), in_shape, s);
  }
  if (any_greater) {
    // Pad with zeros
    auto tmp = zeros(in_shape, a.dtype(), s);
    in = slice_update(tmp, in, Shape(in.ndim(), 0), in.shape());
  }

  auto out_shape = in_shape;
  if (real) {
    auto ax = valid_axes.back();
    out_shape[ax] = inverse ? n.back() : out_shape[ax] / 2 + 1;
  }

  auto in_type = real && !inverse ? float32 : complex64;
  auto out_type = real && inverse ? float32 : complex64;
  return array(
      out_shape,
      out_type,
      std::make_shared<FFT>(to_stream(s), valid_axes, inverse, real),
      {astype(in, in_type, s)});
}

array fft_impl(
    const array& a,
    const std::vector<int>& axes,
    bool real,
    bool inverse,
    StreamOrDevice s) {
  Shape n;
  for (auto ax : axes) {
    n.push_back(a.shape(ax));
  }
  if (real && inverse && a.ndim() > 0) {
    n.back() = (n.back() - 1) * 2;
  }
  return fft_impl(a, n, axes, real, inverse, s);
}

array fft_impl(const array& a, bool real, bool inverse, StreamOrDevice s) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return fft_impl(a, axes, real, inverse, s);
}

array fftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, n, axes, false, false, s);
}
array fftn(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, axes, false, false, s);
}
array fftn(const array& a, StreamOrDevice s /* = {} */) {
  return fft_impl(a, false, false, s);
}

array ifftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, n, axes, false, true, s);
}
array ifftn(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, axes, false, true, s);
}
array ifftn(const array& a, StreamOrDevice s /* = {} */) {
  return fft_impl(a, false, true, s);
}

array rfftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, n, axes, true, false, s);
}
array rfftn(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, axes, true, false, s);
}
array rfftn(const array& a, StreamOrDevice s /* = {} */) {
  return fft_impl(a, true, false, s);
}

array irfftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, n, axes, true, true, s);
}
array irfftn(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, axes, true, true, s);
}

array irfftn(const array& a, StreamOrDevice s /* = {} */) {
  return fft_impl(a, true, true, s);
}

array fftshift(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  if (axes.empty()) {
    return a;
  }

  Shape shifts;
  for (int ax : axes) {
    // Convert negative axes to positive
    int axis = ax < 0 ? ax + a.ndim() : ax;
    if (axis < 0 || axis >= a.ndim()) {
      std::ostringstream msg;
      msg << "[fftshift] Invalid axis " << ax << " for array with " << a.ndim()
          << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    // Match NumPy's implementation
    shifts.push_back(a.shape(axis) / 2);
  }

  return roll(a, shifts, axes, s);
}

array ifftshift(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s /* = {} */) {
  if (axes.empty()) {
    return a;
  }

  Shape shifts;
  for (int ax : axes) {
    // Convert negative axes to positive
    int axis = ax < 0 ? ax + a.ndim() : ax;
    if (axis < 0 || axis >= a.ndim()) {
      std::ostringstream msg;
      msg << "[ifftshift] Invalid axis " << ax << " for array with " << a.ndim()
          << " dimensions.";
      throw std::invalid_argument(msg.str());
    }
    // Match NumPy's implementation
    int size = a.shape(axis);
    shifts.push_back(-(size / 2));
  }

  return roll(a, shifts, axes, s);
}

// Default versions that operate on all axes
array fftshift(const array& a, StreamOrDevice s /* = {} */) {
  if (a.ndim() < 1) {
    return a;
  }
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return fftshift(a, axes, s);
}

array ifftshift(const array& a, StreamOrDevice s /* = {} */) {
  if (a.ndim() < 1) {
    return a;
  }
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return ifftshift(a, axes, s);
}

} // namespace mlx::core::fft
