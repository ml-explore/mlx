// Copyright © 2023 Apple Inc.
#include <algorithm>
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

namespace {

// Pad the last axis; mx::pad has no reflect mode, so build it from take.
array pad_last_axis(
    const array& a,
    int pad_width,
    const std::string& mode,
    StreamOrDevice s) {
  if (pad_width <= 0) {
    return a;
  }
  int ax = a.ndim() - 1;
  if (mode == "constant") {
    return pad(
        a, {ax}, {pad_width}, {pad_width}, array(0, a.dtype()), "constant", s);
  }
  if (mode == "edge") {
    int n = a.shape(ax);
    auto left = take(a, full(Shape{pad_width}, array(0), int32, s), ax, s);
    auto right = take(a, full(Shape{pad_width}, array(n - 1), int32, s), ax, s);
    return concatenate({left, a, right}, ax, s);
  }
  if (mode == "reflect") {
    int n = a.shape(ax);
    if (pad_width >= n) {
      std::ostringstream msg;
      msg << "[stft] Reflect padding (" << pad_width
          << ") requires an input longer than the padding along the last axis ("
          << n << ").";
      throw std::invalid_argument(msg.str());
    }
    auto left = take(a, arange(pad_width, 0, -1, s), ax, s);
    auto right = take(a, arange(n - 2, n - 2 - pad_width, -1, s), ax, s);
    return concatenate({left, a, right}, ax, s);
  }
  std::ostringstream msg;
  msg << "[stft] Invalid pad_mode '" << mode
      << "'. Expected one of {'constant', 'reflect', 'edge'}.";
  throw std::invalid_argument(msg.str());
}

// Center a window of length <= n_fft inside an n_fft-length frame.
array prepare_window(
    const std::optional<array>& window,
    int win_length,
    int n_fft,
    const std::string& op,
    StreamOrDevice s) {
  array win =
      window.has_value() ? window.value() : ones({win_length}, float32, s);
  if (win.ndim() != 1) {
    throw std::invalid_argument(
        "[" + op + "] window must be a one-dimensional array.");
  }
  int wl = win.shape(0);
  if (wl > n_fft) {
    throw std::invalid_argument(
        "[" + op + "] window length must not exceed n_fft.");
  }
  if (wl < n_fft) {
    int pad_left = (n_fft - wl) / 2;
    int pad_right = n_fft - wl - pad_left;
    win =
        pad(win,
            {0},
            {pad_left},
            {pad_right},
            array(0, win.dtype()),
            "constant",
            s);
  }
  return win;
}

} // namespace

array stft(
    const array& x_in,
    int n_fft /* = 2048 */,
    const std::optional<int>& hop_length_ /* = std::nullopt */,
    const std::optional<int>& win_length_ /* = std::nullopt */,
    const std::optional<array>& window /* = std::nullopt */,
    bool center /* = true */,
    const std::string& pad_mode /* = "reflect" */,
    FFTNorm norm /* = FFTNorm::Backward */,
    const std::optional<bool>& onesided_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  if (n_fft <= 0) {
    throw std::invalid_argument("[stft] n_fft must be positive.");
  }
  if (x_in.ndim() < 1) {
    throw std::invalid_argument(
        "[stft] Input must have at least one dimension.");
  }
  int hop_length = hop_length_.value_or(n_fft / 4);
  int win_length = win_length_.value_or(n_fft);
  if (hop_length <= 0) {
    throw std::invalid_argument("[stft] hop_length must be positive.");
  }
  bool onesided = onesided_.value_or(x_in.dtype() != complex64);

  array win = prepare_window(window, win_length, n_fft, "stft", s);

  // Collapse leading dims into a single batch axis.
  Shape lead = x_in.shape();
  lead.pop_back();
  int t = x_in.shape(-1);
  array x = reshape(x_in, {-1, t}, s);

  if (center) {
    x = pad_last_axis(x, n_fft / 2, pad_mode, s);
  }
  int tp = x.shape(-1);
  if (tp < n_fft) {
    throw std::invalid_argument(
        "[stft] Input is too short for the given n_fft.");
  }
  int n_frames = 1 + (tp - n_fft) / hop_length;
  int b = x.shape(0);

  // Overlapping frames via as_strided (element strides: tp, hop, 1).
  array frames = as_strided(
      x,
      {b, n_frames, n_fft},
      Strides{static_cast<int64_t>(tp), static_cast<int64_t>(hop_length), 1},
      0,
      s);
  frames = multiply(frames, win, s);

  array spec = onesided ? rfftn(frames, Shape{n_fft}, {-1}, norm, s)
                        : fftn(frames, Shape{n_fft}, {-1}, norm, s);
  spec = swapaxes(spec, -1, -2, s);

  Shape out_shape = lead;
  out_shape.push_back(spec.shape(-2));
  out_shape.push_back(n_frames);
  return reshape(spec, out_shape, s);
}

array istft(
    const array& stft_matrix,
    const std::optional<int>& n_fft_ /* = std::nullopt */,
    const std::optional<int>& hop_length_ /* = std::nullopt */,
    const std::optional<int>& win_length_ /* = std::nullopt */,
    const std::optional<array>& window /* = std::nullopt */,
    bool center /* = true */,
    FFTNorm norm /* = FFTNorm::Backward */,
    const std::optional<bool>& onesided_ /* = std::nullopt */,
    const std::optional<int>& length_ /* = std::nullopt */,
    StreamOrDevice s /* = {} */) {
  if (stft_matrix.ndim() < 2) {
    throw std::invalid_argument(
        "[istft] Input must have at least two dimensions (freq, frames).");
  }
  bool onesided = onesided_.value_or(true);
  int n_freq = stft_matrix.shape(-2);
  int n_fft = n_fft_.value_or(onesided ? (n_freq - 1) * 2 : n_freq);
  if (n_fft <= 0) {
    throw std::invalid_argument("[istft] n_fft must be positive.");
  }
  int hop_length = hop_length_.value_or(n_fft / 4);
  int win_length = win_length_.value_or(n_fft);
  if (hop_length <= 0) {
    throw std::invalid_argument("[istft] hop_length must be positive.");
  }

  array win = prepare_window(window, win_length, n_fft, "istft", s);

  Shape lead = stft_matrix.shape();
  int n_frames = lead.back();
  lead.pop_back();
  lead.pop_back();
  array z = reshape(stft_matrix, {-1, n_freq, n_frames}, s);
  int b = z.shape(0);
  z = swapaxes(z, -1, -2, s);

  array frames = onesided ? irfftn(z, Shape{n_fft}, {-1}, norm, s)
                          : real(ifftn(z, Shape{n_fft}, {-1}, norm, s), s);
  frames = multiply(frames, win, s);

  // Overlap-add: sub-block k of frame i lands on output block i + k, so
  // summing seg = ceil(n_fft / hop) shifted copies is independent of n_frames.
  int seg = (n_fft + hop_length - 1) / hop_length;
  int wp = seg * hop_length;
  if (wp > n_fft) {
    frames =
        pad(frames,
            {2},
            {0},
            {wp - n_fft},
            array(0, frames.dtype()),
            "constant",
            s);
  }
  array c = reshape(frames, {b, n_frames, seg, hop_length}, s);

  array win_sq = square(win, s);
  if (wp > n_fft) {
    win_sq =
        pad(win_sq,
            {0},
            {0},
            {wp - n_fft},
            array(0, win_sq.dtype()),
            "constant",
            s);
  }
  array cw = broadcast_to(
      reshape(win_sq, {1, 1, seg, hop_length}, s),
      {1, n_frames, seg, hop_length},
      s);

  int out_blocks = n_frames + seg - 1;
  int out_len = out_blocks * hop_length;
  array signal = zeros({b, out_len}, frames.dtype(), s);
  array envelope = zeros({1, out_len}, frames.dtype(), s);
  for (int k = 0; k < seg; ++k) {
    array ck = reshape(
        slice(c, {0, 0, k, 0}, {b, n_frames, k + 1, hop_length}, s),
        {b, n_frames, hop_length},
        s);
    ck = pad(ck, {1}, {k}, {seg - 1 - k}, array(0, ck.dtype()), "constant", s);
    signal = add(signal, reshape(ck, {b, out_len}, s), s);

    array wk = reshape(
        slice(cw, {0, 0, k, 0}, {1, n_frames, k + 1, hop_length}, s),
        {1, n_frames, hop_length},
        s);
    wk = pad(wk, {1}, {k}, {seg - 1 - k}, array(0, wk.dtype()), "constant", s);
    envelope = add(envelope, reshape(wk, {1, out_len}, s), s);
  }
  signal =
      divide(signal, maximum(envelope, array(1e-8f, envelope.dtype()), s), s);

  // Drop the centering pad from the start. The end is bounded by `length` when
  // given, otherwise by removing the centering pad from the end as well. The
  // full reconstruction spans n_fft + (n_frames - 1) * hop before centering.
  int sig_len = n_fft + (n_frames - 1) * hop_length;
  int start = center ? n_fft / 2 : 0;
  if (length_.has_value()) {
    int length = length_.value();
    int stop = std::min(sig_len, start + length);
    signal = slice(signal, {0, start}, {b, stop}, s);
    int cur = signal.shape(-1);
    if (cur < length) {
      signal =
          pad(signal,
              {1},
              {0},
              {length - cur},
              array(0, signal.dtype()),
              "constant",
              s);
    }
  } else {
    int stop = center ? sig_len - n_fft / 2 : sig_len;
    signal = slice(signal, {0, start}, {b, stop}, s);
  }

  Shape out_shape = lead;
  out_shape.push_back(signal.shape(-1));
  return reshape(signal, out_shape, s);
}

} // namespace mlx::core::fft
