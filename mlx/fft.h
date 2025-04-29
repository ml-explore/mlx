// Copyright © 2023 Apple Inc.

#pragma once

#include <variant>

#include "array.h"
#include "device.h"
#include "utils.h"

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
/** Shift the zero-frequency component to the center of the spectrum. */
array fftshift(const array& a, StreamOrDevice s = {});

/** Shift the zero-frequency component to the center of the spectrum along
 * specified axes. */
array fftshift(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s = {});

/** The inverse of fftshift. */
array ifftshift(const array& a, StreamOrDevice s = {});

/** The inverse of fftshift along specified axes. */
array ifftshift(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s = {});

} // namespace mlx::core::fft
