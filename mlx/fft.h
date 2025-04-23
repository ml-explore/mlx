// Copyright © 2023 Apple Inc.

#pragma once

#include <variant>

#include "array.h"
#include "device.h"
#include "mlx/mlx.h"
#include "utils.h"

namespace mx = mlx::core;

namespace mlx::core::fft {

/** Compute the n-dimensional Fourier Transform. */
array fftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
array fftn(const array& a, const std::vector<int>& axes, StreamOrDevice s = {});
array fftn(const array& a, StreamOrDevice s = {});

/** Compute the n-dimensional inverse Fourier Transform. */
array ifftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
array ifftn(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
array ifftn(const array& a, StreamOrDevice s = {});

/** Compute the one-dimensional Fourier Transform. */
inline array fft(const array& a, int n, int axis, StreamOrDevice s = {}) {
  return fftn(a, {n}, {axis}, s);
}
inline array fft(const array& a, int axis = -1, StreamOrDevice s = {}) {
  return fftn(a, {axis}, s);
}

/** Compute the one-dimensional inverse Fourier Transform. */
inline array ifft(const array& a, int n, int axis, StreamOrDevice s = {}) {
  return ifftn(a, {n}, {axis}, s);
}
inline array ifft(const array& a, int axis = -1, StreamOrDevice s = {}) {
  return ifftn(a, {axis}, s);
}

/** Compute the two-dimensional Fourier Transform. */
inline array fft2(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    StreamOrDevice s = {}) {
  return fftn(a, n, axes, s);
}
inline array fft2(
    const array& a,
    const std::vector<int>& axes = {-2, -1},
    StreamOrDevice s = {}) {
  return fftn(a, axes, s);
}

/** Compute the two-dimensional inverse Fourier Transform. */
inline array ifft2(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    StreamOrDevice s = {}) {
  return ifftn(a, n, axes, s);
}
inline array ifft2(
    const array& a,
    const std::vector<int>& axes = {-2, -1},
    StreamOrDevice s = {}) {
  return ifftn(a, axes, s);
}

/** Compute the n-dimensional Fourier Transform on a real input. */
array rfftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
array rfftn(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
array rfftn(const array& a, StreamOrDevice s = {});

/** Compute the n-dimensional inverse of `rfftn`. */
array irfftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
array irfftn(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
array irfftn(const array& a, StreamOrDevice s = {});

/** Compute the one-dimensional Fourier Transform on a real input. */
inline array rfft(const array& a, int n, int axis, StreamOrDevice s = {}) {
  return rfftn(a, {n}, {axis}, s);
}
inline array rfft(const array& a, int axis = -1, StreamOrDevice s = {}) {
  return rfftn(a, {axis}, s);
}
/** Compute the one-dimensional inverse of `rfft`. */
inline array irfft(const array& a, int n, int axis, StreamOrDevice s = {}) {
  return irfftn(a, {n}, {axis}, s);
}
inline array irfft(const array& a, int axis = -1, StreamOrDevice s = {}) {
  return irfftn(a, {axis}, s);
}

/** Compute the two-dimensional Fourier Transform on a real input. */
inline array rfft2(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    StreamOrDevice s = {}) {
  return rfftn(a, n, axes, s);
}
inline array rfft2(
    const array& a,
    const std::vector<int>& axes = {-2, -1},
    StreamOrDevice s = {}) {
  return rfftn(a, axes, s);
}

/** Compute the two-dimensional inverse of `rfft2`. */
inline array irfft2(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    StreamOrDevice s = {}) {
  return irfftn(a, n, axes, s);
}
inline array irfft2(
    const array& a,
    const std::vector<int>& axes = {-2, -1},
    StreamOrDevice s = {}) {
  return irfftn(a, axes, s);
}

inline array stft(
    const array& x,
    int n_fft = 2048,
    int hop_length = -1,
    int win_length = -1,
    const array& window = mx::array({}),
    bool center = true,
    const std::string& pad_mode = "reflect",
    bool normalized = false,
    bool onesided = true,
    StreamOrDevice s = {}) {
  return mlx::core::fft::stft(
      x,
      n_fft,
      hop_length,
      win_length,
      window,
      center,
      pad_mode,
      normalized,
      onesided,
      s);
}

inline array istft(
    const array& stft_matrix,
    int hop_length = -1,
    int win_length = -1,
    const array& window = mx::array({}),
    bool center = true,
    int length = -1,
    bool normalized = false,
    StreamOrDevice s = {}) {
  return mlx::core::fft::istft(
      stft_matrix,
      hop_length,
      win_length,
      window,
      center,
      length,
      normalized,
      s);
}

} // namespace mlx::core::fft
