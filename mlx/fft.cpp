// Copyright © 2023 Apple Inc.

#include <cmath>
#include <numeric>
#include <set>
#include <sstream>

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
  if (real && inverse) {
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

array stft(
    const array& x,
    int n_fft = 2048,
    int hop_length = -1,
    int win_length = -1,
    const array& window,
    bool center = true,
    const std::string& pad_mode = "reflect",
    bool normalized = false,
    bool onesided = true,
    StreamOrDevice s /* = {} */) {
  if (hop_length == -1)
    hop_length = n_fft / 4;
  if (win_length == -1)
    win_length = n_fft;

  array win = (window.size() == 0) ? ones({win_length}, float32, s) : window;

  if (win_length < n_fft) {
    int pad_left = (n_fft - win_length) / 2;
    int pad_right = n_fft - win_length - pad_left;
    win = mlx::core::pad(
        win, {{pad_left, pad_right}}, array(0, float32), "constant", s);
  }

  array padded_x = x;
  if (center) {
    int pad_width = n_fft / 2;
    padded_x = mlx::core::pad(
        padded_x, {{pad_width, pad_width}}, array(0, x.dtype()), pad_mode, s);
  }

  int n_frames = 1 + (padded_x.shape(0) - n_fft) / hop_length;

  std::vector<array> frames;
  for (int i = 0; i < n_frames; ++i) {
    array frame =
        slice(padded_x, {i * hop_length}, {i * hop_length + n_fft}, s);
    frames.push_back(multiply(frame, win, s));
  }

  array stacked_frames = stack(frames, 0, s);

  array stft_result = mlx::core::fft::rfftn(stacked_frames, {n_fft}, {-1}, s);

  if (normalized) {
    array n_fft_array = full({1}, static_cast<float>(n_fft), float32, s);
    stft_result = divide(stft_result, sqrt(n_fft_array, s), s);
  }

  if (onesided) {
    stft_result = slice(stft_result, {}, {n_fft / 2 + 1}, s);
  }

  return stft_result;
}

array istft(
    const array& stft_matrix,
    int hop_length = -1,
    int win_length = -1,
    const array& window,
    bool center = true,
    int length = -1,
    bool normalized = false,
    StreamOrDevice s /* = {} */) {
  int n_fft = (stft_matrix.shape(-1) - 1) * 2;
  if (hop_length == -1)
    hop_length = n_fft / 4;
  if (win_length == -1)
    win_length = n_fft;

  array win = (window.size() == 0) ? ones({win_length}, float32, s) : window;

  if (win_length < n_fft) {
    int pad_left = (n_fft - win_length) / 2;
    int pad_right = n_fft - win_length - pad_left;
    win = mlx::core::pad(
        win, {{pad_left, pad_right}}, array(0, float32), "constant", s);
  }

  array frames = mlx::core::fft::irfftn(stft_matrix, {n_fft}, {-1}, s);

  frames = multiply(frames, win, s);

  int signal_length = (frames.shape(0) - 1) * hop_length + n_fft;
  array signal = zeros({signal_length}, float32, s);
  array window_sum = zeros({signal_length}, float32, s);

  for (int i = 0; i < frames.shape(0); ++i) {
    array frame = reshape(slice(frames, {i}, {i + 1}, s), {n_fft}, s);
    array signal_slice =
        slice(signal, {i * hop_length}, {i * hop_length + n_fft}, s);
    array window_slice =
        slice(window_sum, {i * hop_length}, {i * hop_length + n_fft}, s);

    signal_slice = add(signal_slice, frame, s);
    window_slice = add(window_slice, win, s);
  }

  signal = divide(signal, window_sum, s);

  if (center) {
    int pad_width = n_fft / 2;
    signal = slice(signal, {pad_width}, {signal.shape(0) - pad_width}, s);
  }

  if (length > 0) {
    if (signal.shape(0) > length) {
      signal = slice(signal, {0}, {length}, s);
    } else if (signal.shape(0) < length) {
      int pad_length = length - signal.shape(0);
      signal = mlx::core::pad(
          signal, {{0, pad_length}}, array(0, signal.dtype()), "constant", s);
    }
  }

  return signal;
}

} // namespace mlx::core::fft