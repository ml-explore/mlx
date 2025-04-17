// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <numeric>

#include "mlx/fft.h"
#include "mlx/ops.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

void init_fft(nb::module_& parent_module) {
  auto m = parent_module.def_submodule(
      "fft", "mlx.core.fft: Fast Fourier Transforms.");
  m.def(
      "fft",
      [](const mx::array& a,
         const std::optional<int>& n,
         int axis,
         mx::StreamOrDevice s) {
        if (n.has_value()) {
          return mx::fft::fft(a, n.value(), axis, s);
        } else {
          return mx::fft::fft(a, axis, s);
        }
      },
      "a"_a,
      "n"_a = nb::none(),
      "axis"_a = -1,
      "stream"_a = nb::none(),
      R"pbdoc(
        One dimensional discrete Fourier Transform.

        Args:
            a (array): The input array.
            n (int, optional): Size of the transformed axis. The
               corresponding axis in the input is truncated or padded with
               zeros to match ``n``. The default value is ``a.shape[axis]``.
            axis (int, optional): Axis along which to perform the FFT. The
               default is ``-1``.

        Returns:
            array: The DFT of the input along the given axis.
      )pbdoc");
  m.def(
      "ifft",
      [](const mx::array& a,
         const std::optional<int>& n,
         int axis,
         mx::StreamOrDevice s) {
        if (n.has_value()) {
          return mx::fft::ifft(a, n.value(), axis, s);
        } else {
          return mx::fft::ifft(a, axis, s);
        }
      },
      "a"_a,
      "n"_a = nb::none(),
      "axis"_a = -1,
      "stream"_a = nb::none(),
      R"pbdoc(
        One dimensional inverse discrete Fourier Transform.

        Args:
            a (array): The input array.
            n (int, optional): Size of the transformed axis. The
               corresponding axis in the input is truncated or padded with
               zeros to match ``n``. The default value is ``a.shape[axis]``.
            axis (int, optional): Axis along which to perform the FFT. The
               default is ``-1``.

        Returns:
            array: The inverse DFT of the input along the given axis.
      )pbdoc");
  m.def(
      "fft2",
      [](const mx::array& a,
         const std::optional<mx::Shape>& n,
         const std::optional<std::vector<int>>& axes,
         mx::StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return mx::fft::fftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return mx::fft::fftn(a, axes.value(), s);
        } else if (n.has_value()) {
          throw std::invalid_argument(
              "[fft2] `axes` should not be `None` if `s` is not `None`.");
        } else {
          return mx::fft::fftn(a, s);
        }
      },
      "a"_a,
      "s"_a = nb::none(),
      "axes"_a.none() = std::vector<int>{-2, -1},
      "stream"_a = nb::none(),
      R"pbdoc(
        Two dimensional discrete Fourier Transform.

        Args:
            a (array): The input array.
            s (list(int), optional): Sizes of the transformed axes. The
               corresponding axes in the input are truncated or padded with
               zeros to match the sizes in ``s``. The default value is the
               sizes of ``a`` along ``axes``.
            axes (list(int), optional): Axes along which to perform the FFT.
               The default is ``[-2, -1]``.

        Returns:
            array: The DFT of the input along the given axes.
      )pbdoc");
  m.def(
      "ifft2",
      [](const mx::array& a,
         const std::optional<mx::Shape>& n,
         const std::optional<std::vector<int>>& axes,
         mx::StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return mx::fft::ifftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return mx::fft::ifftn(a, axes.value(), s);
        } else if (n.has_value()) {
          throw std::invalid_argument(
              "[ifft2] `axes` should not be `None` if `s` is not `None`.");
        } else {
          return mx::fft::ifftn(a, s);
        }
      },
      "a"_a,
      "s"_a = nb::none(),
      "axes"_a.none() = std::vector<int>{-2, -1},
      "stream"_a = nb::none(),
      R"pbdoc(
        Two dimensional inverse discrete Fourier Transform.

        Args:
            a (array): The input array.
            s (list(int), optional): Sizes of the transformed axes. The
               corresponding axes in the input are truncated or padded with
               zeros to match the sizes in ``s``. The default value is the
               sizes of ``a`` along ``axes``.
            axes (list(int), optional): Axes along which to perform the FFT.
               The default is ``[-2, -1]``.

        Returns:
            array: The inverse DFT of the input along the given axes.
      )pbdoc");
  m.def(
      "fftn",
      [](const mx::array& a,
         const std::optional<mx::Shape>& n,
         const std::optional<std::vector<int>>& axes,
         mx::StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return mx::fft::fftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return mx::fft::fftn(a, axes.value(), s);
        } else if (n.has_value()) {
          throw std::invalid_argument(
              "[fftn] `axes` should not be `None` if `s` is not `None`.");
        } else {
          return mx::fft::fftn(a, s);
        }
      },
      "a"_a,
      "s"_a = nb::none(),
      "axes"_a = nb::none(),
      "stream"_a = nb::none(),
      R"pbdoc(
        n-dimensional discrete Fourier Transform.

        Args:
            a (array): The input array.
            s (list(int), optional): Sizes of the transformed axes. The
               corresponding axes in the input are truncated or padded with
               zeros to match the sizes in ``s``. The default value is the
               sizes of ``a`` along ``axes``.
            axes (list(int), optional): Axes along which to perform the FFT.
               The default is ``None`` in which case the FFT is over the last
               ``len(s)`` axes are or all axes if ``s`` is also ``None``.

        Returns:
            array: The DFT of the input along the given axes.
      )pbdoc");
  m.def(
      "ifftn",
      [](const mx::array& a,
         const std::optional<mx::Shape>& n,
         const std::optional<std::vector<int>>& axes,
         mx::StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return mx::fft::ifftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return mx::fft::ifftn(a, axes.value(), s);
        } else if (n.has_value()) {
          throw std::invalid_argument(
              "[ifftn] `axes` should not be `None` if `s` is not `None`.");
        } else {
          return mx::fft::ifftn(a, s);
        }
      },
      "a"_a,
      "s"_a = nb::none(),
      "axes"_a = nb::none(),
      "stream"_a = nb::none(),
      R"pbdoc(
        n-dimensional inverse discrete Fourier Transform.

        Args:
            a (array): The input array.
            s (list(int), optional): Sizes of the transformed axes. The
               corresponding axes in the input are truncated or padded with
               zeros to match the sizes in ``s``. The default value is the
               sizes of ``a`` along ``axes``.
            axes (list(int), optional): Axes along which to perform the FFT.
               The default is ``None`` in which case the FFT is over the last
               ``len(s)`` axes or all axes if ``s`` is also ``None``.

        Returns:
            array: The inverse DFT of the input along the given axes.
      )pbdoc");
  m.def(
      "rfft",
      [](const mx::array& a,
         const std::optional<int>& n,
         int axis,
         mx::StreamOrDevice s) {
        if (n.has_value()) {
          return mx::fft::rfft(a, n.value(), axis, s);
        } else {
          return mx::fft::rfft(a, axis, s);
        }
      },
      "a"_a,
      "n"_a = nb::none(),
      "axis"_a = -1,
      "stream"_a = nb::none(),
      R"pbdoc(
        One dimensional discrete Fourier Transform on a real input.

        The output has the same shape as the input except along ``axis`` in
        which case it has size ``n // 2 + 1``.

        Args:
            a (array): The input array. If the array is complex it will be silently
               cast to a real type.
            n (int, optional): Size of the transformed axis. The
               corresponding axis in the input is truncated or padded with
               zeros to match ``n``. The default value is ``a.shape[axis]``.
            axis (int, optional): Axis along which to perform the FFT. The
               default is ``-1``.

        Returns:
            array: The DFT of the input along the given axis. The output
            data type will be complex.
      )pbdoc");
  m.def(
      "irfft",
      [](const mx::array& a,
         const std::optional<int>& n,
         int axis,
         mx::StreamOrDevice s) {
        if (n.has_value()) {
          return mx::fft::irfft(a, n.value(), axis, s);
        } else {
          return mx::fft::irfft(a, axis, s);
        }
      },
      "a"_a,
      "n"_a = nb::none(),
      "axis"_a = -1,
      "stream"_a = nb::none(),
      R"pbdoc(
        The inverse of :func:`rfft`.

        The output has the same shape as the input except along ``axis`` in
        which case it has size ``n``.

        Args:
            a (array): The input array.
            n (int, optional): Size of the transformed axis. The
               corresponding axis in the input is truncated or padded with
               zeros to match ``n // 2 + 1``. The default value is
               ``a.shape[axis] // 2 + 1``.
            axis (int, optional): Axis along which to perform the FFT. The
               default is ``-1``.

        Returns:
            array: The real array containing the inverse of :func:`rfft`.
      )pbdoc");
  m.def(
      "rfft2",
      [](const mx::array& a,
         const std::optional<mx::Shape>& n,
         const std::optional<std::vector<int>>& axes,
         mx::StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return mx::fft::rfftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return mx::fft::rfftn(a, axes.value(), s);
        } else if (n.has_value()) {
          throw std::invalid_argument(
              "[rfft2] `axes` should not be `None` if `s` is not `None`.");
        } else {
          return mx::fft::rfftn(a, s);
        }
      },
      "a"_a,
      "s"_a = nb::none(),
      "axes"_a.none() = std::vector<int>{-2, -1},
      "stream"_a = nb::none(),
      R"pbdoc(
        Two dimensional real discrete Fourier Transform.

        The output has the same shape as the input except along the dimensions in
        ``axes`` in which case it has sizes from ``s``. The last axis in ``axes`` is
        treated as the real axis and will have size ``s[-1] // 2 + 1``.

        Args:
            a (array): The input array. If the array is complex it will be silently
               cast to a real type.
            s (list(int), optional): Sizes of the transformed axes. The
               corresponding axes in the input are truncated or padded with
               zeros to match the sizes in ``s``. The default value is the
               sizes of ``a`` along ``axes``.
            axes (list(int), optional): Axes along which to perform the FFT.
               The default is ``[-2, -1]``.

        Returns:
            array: The real DFT of the input along the given axes. The output
            data type will be complex.
      )pbdoc");
  m.def(
      "irfft2",
      [](const mx::array& a,
         const std::optional<mx::Shape>& n,
         const std::optional<std::vector<int>>& axes,
         mx::StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return mx::fft::irfftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return mx::fft::irfftn(a, axes.value(), s);
        } else if (n.has_value()) {
          throw std::invalid_argument(
              "[irfft2] `axes` should not be `None` if `s` is not `None`.");
        } else {
          return mx::fft::irfftn(a, s);
        }
      },
      "a"_a,
      "s"_a = nb::none(),
      "axes"_a.none() = std::vector<int>{-2, -1},
      "stream"_a = nb::none(),
      R"pbdoc(
        The inverse of :func:`rfft2`.

        Note the input is generally complex. The dimensions of the input
        specified in ``axes`` are padded or truncated to match the sizes
        from ``s``. The last axis in ``axes`` is treated as the real axis
        and will have size ``s[-1] // 2 + 1``.

        Args:
            a (array): The input array.
            s (list(int), optional): Sizes of the transformed axes. The
               corresponding axes in the input are truncated or padded with
               zeros to match the sizes in ``s`` except for the last axis
               which has size ``s[-1] // 2 + 1``. The default value is the
               sizes of ``a`` along ``axes``.
            axes (list(int), optional): Axes along which to perform the FFT.
               The default is ``[-2, -1]``.

        Returns:
            array: The real array containing the inverse of :func:`rfft2`.
      )pbdoc");
  m.def(
      "rfftn",
      [](const mx::array& a,
         const std::optional<mx::Shape>& n,
         const std::optional<std::vector<int>>& axes,
         mx::StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return mx::fft::rfftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return mx::fft::rfftn(a, axes.value(), s);
        } else if (n.has_value()) {
          throw std::invalid_argument(
              "[rfftn] `axes` should not be `None` if `s` is not `None`.");
        } else {
          return mx::fft::rfftn(a, s);
        }
      },
      "a"_a,
      "s"_a = nb::none(),
      "axes"_a = nb::none(),
      "stream"_a = nb::none(),
      R"pbdoc(
        n-dimensional real discrete Fourier Transform.

        The output has the same shape as the input except along the dimensions in
        ``axes`` in which case it has sizes from ``s``. The last axis in ``axes`` is
        treated as the real axis and will have size ``s[-1] // 2 + 1``.

        Args:
            a (array): The input array. If the array is complex it will be silently
               cast to a real type.
            s (list(int), optional): Sizes of the transformed axes. The
               corresponding axes in the input are truncated or padded with
               zeros to match the sizes in ``s``. The default value is the
               sizes of ``a`` along ``axes``.
            axes (list(int), optional): Axes along which to perform the FFT.
               The default is ``None`` in which case the FFT is over the last
               ``len(s)`` axes or all axes if ``s`` is also ``None``.

        Returns:
            array: The real DFT of the input along the given axes. The output
      )pbdoc");
  m.def(
      "irfftn",
      [](const mx::array& a,
         const std::optional<mx::Shape>& n,
         const std::optional<std::vector<int>>& axes,
         mx::StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return mx::fft::irfftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return mx::fft::irfftn(a, axes.value(), s);
        } else if (n.has_value()) {
          throw std::invalid_argument(
              "[irfftn] `axes` should not be `None` if `s` is not `None`.");
        } else {
          return mx::fft::irfftn(a, s);
        }
      },
      "a"_a,
      "s"_a = nb::none(),
      "axes"_a = nb::none(),
      "stream"_a = nb::none(),
      R"pbdoc(
        The inverse of :func:`rfftn`.

        Note the input is generally complex. The dimensions of the input
        specified in ``axes`` are padded or truncated to match the sizes
        from ``s``. The last axis in ``axes`` is treated as the real axis
        and will have size ``s[-1] // 2 + 1``.

        Args:
            a (array): The input array.
            s (list(int), optional): Sizes of the transformed axes. The
               corresponding axes in the input are truncated or padded with
               zeros to match the sizes in ``s``. The default value is the
               sizes of ``a`` along ``axes``.
            axes (list(int), optional): Axes along which to perform the FFT.
               The default is ``None`` in which case the FFT is over the last
               ``len(s)`` axes or all axes if ``s`` is also ``None``.

        Returns:
            array: The real array containing the inverse of :func:`rfftn`.
      )pbdoc");

  m.def(
      "stft",
      [](const mx::array& x,
         int n_fft = 2048,
         int hop_length = -1,
         int win_length = -1,
         const mx::array& window = mx::array(),
         bool center = true,
         const std::string& pad_mode = "reflect",
         bool normalized = false,
         bool onesided = true,
         mx::StreamOrDevice s = {}) {
        return mx::stft::stft(
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
      },
      "x"_a,
      "n_fft"_a = 2048,
      "hop_length"_a = -1,
      "win_length"_a = -1,
      "window"_a = mx::array(),
      "center"_a = true,
      "pad_mode"_a = "reflect",
      "normalized"_a = false,
      "onesided"_a = true,
      "stream"_a = nb::none(),
      R"pbdoc(
        Short-time Fourier Transform (STFT).

        Args:
            x (array): Input signal.
            n_fft (int, optional): Number of FFT points. Default is 2048.
            hop_length (int, optional): Number of samples between successive frames. Default is `n_fft // 4`.
            win_length (int, optional): Window size. Default is `n_fft`.
            window (array, optional): Window function. Default is a rectangular window.
            center (bool, optional): Whether to pad the signal to center the frames. Default is True.
            pad_mode (str, optional): Padding mode. Default is "reflect".
            normalized (bool, optional): Whether to normalize the STFT. Default is False.
            onesided (bool, optional): Whether to return a one-sided STFT. Default is True.

        Returns:
            array: The STFT of the input signal.
      )pbdoc");

  m.def(
      "istft",
      [](const mx::array& stft_matrix,
         int hop_length = -1,
         int win_length = -1,
         const mx::array& window = mx::array(),
         bool center = true,
         int length = -1,
         bool normalized = false,
         mx::StreamOrDevice s = {}) {
        return mx::stft::istft(
            stft_matrix,
            hop_length,
            win_length,
            window,
            center,
            length,
            normalized,
            s);
      },
      "stft_matrix"_a,
      "hop_length"_a = -1,
      "win_length"_a = -1,
      "window"_a = mx::array(),
      "center"_a = true,
      "length"_a = -1,
      "normalized"_a = false,
      "stream"_a = nb::none(),
      R"pbdoc(
        Inverse Short-time Fourier Transform (ISTFT).

        Args:
            stft_matrix (array): Input STFT matrix.
            hop_length (int, optional): Number of samples between successive frames. Default is `n_fft // 4`.
            win_length (int, optional): Window size. Default is `n_fft`.
            window (array, optional): Window function. Default is a rectangular window.
            center (bool, optional): Whether the signal was padded to center the frames. Default is True.
            length (int, optional): Length of the output signal. Default is inferred from the STFT matrix.
            normalized (bool, optional): Whether the STFT was normalized. Default is False.

        Returns:
            array: The reconstructed signal.
      )pbdoc");
}
