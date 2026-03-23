// Copyright © 2023 Apple Inc.

#pragma once

#include <cstdint>
#include <variant>

#include "array.h"
#include "device.h"
#include "mlx/api.h"
#include "utils.h"

namespace mlx::core::fft {

enum class FFTNorm {
  Backward,
  Ortho,
  Forward,
};

/** Compute the n-dimensional Fourier Transform. */
MLX_API array fftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {});
MLX_API array fftn(
    const array& a,
    const std::vector<int>& axes,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {});
MLX_API array
fftn(const array& a, FFTNorm norm = FFTNorm::Backward, StreamOrDevice s = {});

/** Compute the n-dimensional inverse Fourier Transform. */
MLX_API array ifftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {});
MLX_API array ifftn(
    const array& a,
    const std::vector<int>& axes,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {});
MLX_API array
ifftn(const array& a, FFTNorm norm = FFTNorm::Backward, StreamOrDevice s = {});

/** Compute the one-dimensional Fourier Transform. */
inline array fft(
    const array& a,
    int n,
    int axis,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return fftn(a, {n}, {axis}, norm, s);
}
inline array fft(
    const array& a,
    int axis = -1,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return fftn(a, {axis}, norm, s);
}

/** Compute the one-dimensional inverse Fourier Transform. */
inline array ifft(
    const array& a,
    int n,
    int axis,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return ifftn(a, {n}, {axis}, norm, s);
}
inline array ifft(
    const array& a,
    int axis = -1,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return ifftn(a, {axis}, norm, s);
}

/** Compute the two-dimensional Fourier Transform. */
inline array fft2(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return fftn(a, n, axes, norm, s);
}
inline array fft2(
    const array& a,
    const std::vector<int>& axes = {-2, -1},
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return fftn(a, axes, norm, s);
}

/** Compute the two-dimensional inverse Fourier Transform. */
inline array ifft2(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return ifftn(a, n, axes, norm, s);
}
inline array ifft2(
    const array& a,
    const std::vector<int>& axes = {-2, -1},
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return ifftn(a, axes, norm, s);
}

/** Compute the n-dimensional Fourier Transform on a real input. */
MLX_API array rfftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {});
MLX_API array rfftn(
    const array& a,
    const std::vector<int>& axes,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {});
MLX_API array
rfftn(const array& a, FFTNorm norm = FFTNorm::Backward, StreamOrDevice s = {});

/** Compute the n-dimensional inverse of `rfftn`. */
MLX_API array irfftn(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {});
MLX_API array irfftn(
    const array& a,
    const std::vector<int>& axes,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {});
MLX_API array
irfftn(const array& a, FFTNorm norm = FFTNorm::Backward, StreamOrDevice s = {});

/** Compute the one-dimensional Fourier Transform on a real input. */
inline array rfft(
    const array& a,
    int n,
    int axis,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return rfftn(a, {n}, {axis}, norm, s);
}
inline array rfft(
    const array& a,
    int axis = -1,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return rfftn(a, {axis}, norm, s);
}
/** Compute the one-dimensional inverse of `rfft`. */
inline array irfft(
    const array& a,
    int n,
    int axis,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return irfftn(a, {n}, {axis}, norm, s);
}
inline array irfft(
    const array& a,
    int axis = -1,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return irfftn(a, {axis}, norm, s);
}

/** Compute the two-dimensional Fourier Transform on a real input. */
inline array rfft2(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return rfftn(a, n, axes, norm, s);
}
inline array rfft2(
    const array& a,
    const std::vector<int>& axes = {-2, -1},
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return rfftn(a, axes, norm, s);
}

/** Compute the two-dimensional inverse of `rfft2`. */
inline array irfft2(
    const array& a,
    const Shape& n,
    const std::vector<int>& axes,
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return irfftn(a, n, axes, norm, s);
}
inline array irfft2(
    const array& a,
    const std::vector<int>& axes = {-2, -1},
    FFTNorm norm = FFTNorm::Backward,
    StreamOrDevice s = {}) {
  return irfftn(a, axes, norm, s);
}
/** Shift the zero-frequency component to the center of the spectrum. */
MLX_API array fftshift(const array& a, StreamOrDevice s = {});

/** Shift the zero-frequency component to the center of the spectrum along
 * specified axes. */
MLX_API array
fftshift(const array& a, const std::vector<int>& axes, StreamOrDevice s = {});

/** The inverse of fftshift. */
MLX_API array ifftshift(const array& a, StreamOrDevice s = {});

/** The inverse of fftshift along specified axes. */
MLX_API array
ifftshift(const array& a, const std::vector<int>& axes, StreamOrDevice s = {});

} // namespace mlx::core::fft
