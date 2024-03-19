// Copyright © 2023-2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <numeric>

#include "mlx/fft.h"
#include "mlx/ops.h"

namespace nb = nanobind;
using namespace nb::literals;

using namespace mlx::core;

void init_fft(nb::module_& parent_module) {
  auto m = parent_module.def_submodule(
      "fft", "mlx.core.fft: Fast Fourier Transforms.");
  m.def(
      "fft",
      [](const array& a,
         const std::optional<int>& n,
         int axis,
         StreamOrDevice s) {
        if (n.has_value()) {
          return fft::fft(a, n.value(), axis, s);
        } else {
          return fft::fft(a, axis, s);
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
      [](const array& a,
         const std::optional<int>& n,
         int axis,
         StreamOrDevice s) {
        if (n.has_value()) {
          return fft::ifft(a, n.value(), axis, s);
        } else {
          return fft::ifft(a, axis, s);
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
      [](const array& a,
         const std::optional<std::vector<int>>& n,
         const std::optional<std::vector<int>>& axes,
         StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return fft::fftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return fft::fftn(a, axes.value(), s);
        } else if (n.has_value()) {
          std::vector<int> axes_(n.value().size());
          std::iota(axes_.begin(), axes_.end(), -n.value().size());
          return fft::fftn(a, n.value(), axes_, s);
        } else {
          return fft::fftn(a, s);
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
      [](const array& a,
         const std::optional<std::vector<int>>& n,
         const std::optional<std::vector<int>>& axes,
         StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return fft::ifftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return fft::ifftn(a, axes.value(), s);
        } else if (n.has_value()) {
          std::vector<int> axes_(n.value().size());
          std::iota(axes_.begin(), axes_.end(), -n.value().size());
          return fft::ifftn(a, n.value(), axes_, s);
        } else {
          return fft::ifftn(a, s);
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
      [](const array& a,
         const std::optional<std::vector<int>>& n,
         const std::optional<std::vector<int>>& axes,
         StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return fft::fftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return fft::fftn(a, axes.value(), s);
        } else if (n.has_value()) {
          std::vector<int> axes_(n.value().size());
          std::iota(axes_.begin(), axes_.end(), -n.value().size());
          return fft::fftn(a, n.value(), axes_, s);
        } else {
          return fft::fftn(a, s);
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
      [](const array& a,
         const std::optional<std::vector<int>>& n,
         const std::optional<std::vector<int>>& axes,
         StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return fft::ifftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return fft::ifftn(a, axes.value(), s);
        } else if (n.has_value()) {
          std::vector<int> axes_(n.value().size());
          std::iota(axes_.begin(), axes_.end(), -n.value().size());
          return fft::ifftn(a, n.value(), axes_, s);
        } else {
          return fft::ifftn(a, s);
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
      [](const array& a,
         const std::optional<int>& n,
         int axis,
         StreamOrDevice s) {
        if (n.has_value()) {
          return fft::rfft(a, n.value(), axis, s);
        } else {
          return fft::rfft(a, axis, s);
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
      [](const array& a,
         const std::optional<int>& n,
         int axis,
         StreamOrDevice s) {
        if (n.has_value()) {
          return fft::irfft(a, n.value(), axis, s);
        } else {
          return fft::irfft(a, axis, s);
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
      [](const array& a,
         const std::optional<std::vector<int>>& n,
         const std::optional<std::vector<int>>& axes,
         StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return fft::rfftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return fft::rfftn(a, axes.value(), s);
        } else if (n.has_value()) {
          std::vector<int> axes_(n.value().size());
          std::iota(axes_.begin(), axes_.end(), -n.value().size());
          return fft::rfftn(a, n.value(), axes_, s);
        } else {
          return fft::rfftn(a, s);
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
      [](const array& a,
         const std::optional<std::vector<int>>& n,
         const std::optional<std::vector<int>>& axes,
         StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return fft::irfftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return fft::irfftn(a, axes.value(), s);
        } else if (n.has_value()) {
          std::vector<int> axes_(n.value().size());
          std::iota(axes_.begin(), axes_.end(), -n.value().size());
          return fft::irfftn(a, n.value(), axes_, s);
        } else {
          return fft::irfftn(a, s);
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
      [](const array& a,
         const std::optional<std::vector<int>>& n,
         const std::optional<std::vector<int>>& axes,
         StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return fft::rfftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return fft::rfftn(a, axes.value(), s);
        } else if (n.has_value()) {
          std::vector<int> axes_(n.value().size());
          std::iota(axes_.begin(), axes_.end(), -n.value().size());
          return fft::rfftn(a, n.value(), axes_, s);
        } else {
          return fft::rfftn(a, s);
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
      [](const array& a,
         const std::optional<std::vector<int>>& n,
         const std::optional<std::vector<int>>& axes,
         StreamOrDevice s) {
        if (axes.has_value() && n.has_value()) {
          return fft::irfftn(a, n.value(), axes.value(), s);
        } else if (axes.has_value()) {
          return fft::irfftn(a, axes.value(), s);
        } else if (n.has_value()) {
          std::vector<int> axes_(n.value().size());
          std::iota(axes_.begin(), axes_.end(), -n.value().size());
          return fft::irfftn(a, n.value(), axes_, s);
        } else {
          return fft::irfftn(a, s);
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
}
