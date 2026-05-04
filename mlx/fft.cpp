// Copyright © 2023 Apple Inc.
#include <cmath>
#include <functional>
#include <numeric>
#include <set>

#include "mlx/fft.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mlx::core::fft {

double fft_scale_factor(const Shape& n, FFTNorm norm, bool inverse) {
  if (n.empty()) {
    return 1.0;
  }
  double n_elements =
      std::accumulate(n.begin(), n.end(), 1.0, std::multiplies<double>());
  switch (norm) {
    case FFTNorm::Backward:
      return 1.0;
    case FFTNorm::Ortho:
      return inverse ? std::sqrt(n_elements) : 1.0 / std::sqrt(n_elements);
    case FFTNorm::Forward:
      return inverse ? n_elements : 1.0 / n_elements;
  }
  throw std::invalid_argument("[fftn] Invalid FFT normalization mode.");
}

array fft_impl(
    const array& a,
    Shape n,
    const std::vector<int>& axes,
    bool real,
    bool inverse,
    FFTNorm norm,
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
  auto out = array(
      out_shape,
      out_type,
      std::make_shared<FFT>(to_stream(s), valid_axes, inverse, real),
      {astype(in, in_type, s)});
  auto scale = fft_scale_factor(n, norm, inverse);
  if (scale != 1.0) {
    return multiply(out, array(scale, out.dtype()), s);
  }
  return out;
}

array fft_impl(
    const array& a,
    const std::vector<int>& axes,
    bool real,
    bool inverse,
    FFTNorm norm,
    StreamOrDevice s) {
  Shape n;
  for (auto ax : axes) {
    n.push_back(a.shape(ax));
  }
  if (real && inverse && a.ndim() > 0) {
    n.back() = (n.back() - 1) * 2;
  }
  return fft_impl(a, n, axes, real, inverse, norm, s);
}

array fft_impl(
    const array& a,
    bool real,
    bool inverse,
    FFTNorm norm,
    StreamOrDevice s) {
  std::vector<int> axes(a.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  return fft_impl(a, axes, real, inverse, norm, s);
}

array fftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    FFTNorm norm /* = FFTNorm::Backward */,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, n, axes, false, false, norm, s);
}
array fftn(
    const array& a,
    const std::vector<int>& axes,
    FFTNorm norm /* = FFTNorm::Backward */,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, axes, false, false, norm, s);
}
array fftn(
    const array& a,
    FFTNorm norm /* = FFTNorm::Backward */,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, false, false, norm, s);
}

array ifftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    FFTNorm norm /* = FFTNorm::Backward */,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, n, axes, false, true, norm, s);
}
array ifftn(
    const array& a,
    const std::vector<int>& axes,
    FFTNorm norm /* = FFTNorm::Backward */,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, axes, false, true, norm, s);
}
array ifftn(
    const array& a,
    FFTNorm norm /* = FFTNorm::Backward */,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, false, true, norm, s);
}

array rfftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    FFTNorm norm /* = FFTNorm::Backward */,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, n, axes, true, false, norm, s);
}
array rfftn(
    const array& a,
    const std::vector<int>& axes,
    FFTNorm norm /* = FFTNorm::Backward */,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, axes, true, false, norm, s);
}
array rfftn(
    const array& a,
    FFTNorm norm /* = FFTNorm::Backward */,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, true, false, norm, s);
}

array irfftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    FFTNorm norm /* = FFTNorm::Backward */,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, n, axes, true, true, norm, s);
}
array irfftn(
    const array& a,
    const std::vector<int>& axes,
    FFTNorm norm /* = FFTNorm::Backward */,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, axes, true, true, norm, s);
}

array irfftn(
    const array& a,
    FFTNorm norm /* = FFTNorm::Backward */,
    StreamOrDevice s /* = {} */) {
  return fft_impl(a, true, true, norm, s);
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

array fftfreq(int n, double d /* = 1.0 */, StreamOrDevice s /* = {} */) {
  if (n <= 0) {
    throw std::invalid_argument("[fftfreq] `n` must be greater than 0.");
  }
  if (d == 0.0) {
    throw std::invalid_argument("[fftfreq] `d` must be non-zero.");
  }
  auto pos = arange(0, (n + 1) / 2, float32, s);
  auto neg = arange(-(n / 2), 0, float32, s);
  auto freqs = concatenate({pos, neg}, s);
  auto scale =
      array(static_cast<float>(1.0 / (static_cast<double>(n) * d)), float32);
  return multiply(freqs, scale, s);
}

array rfftfreq(int n, double d /* = 1.0 */, StreamOrDevice s /* = {} */) {
  if (n <= 0) {
    throw std::invalid_argument("[rfftfreq] `n` must be greater than 0.");
  }
  if (d == 0.0) {
    throw std::invalid_argument("[rfftfreq] `d` must be non-zero.");
  }
  auto freqs = arange(0, n / 2 + 1, float32, s);
  auto scale =
      array(static_cast<float>(1.0 / (static_cast<double>(n) * d)), float32);
  return multiply(freqs, scale, s);
}

} // namespace mlx::core::fft
