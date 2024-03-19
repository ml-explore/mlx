// Copyright © 2023-2024 Apple Inc.

#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/linalg.h"

namespace nb = nanobind;
using namespace nb::literals;

using namespace mlx::core;
using namespace mlx::core::linalg;

namespace {
nb::tuple svd_helper(const array& a, StreamOrDevice s /* = {} */) {
  const auto result = svd(a, s);
  return nb::make_tuple(result.at(0), result.at(1), result.at(2));
}
} // namespace

void init_linalg(nb::module_& parent_module) {
  auto m = parent_module.def_submodule(
      "linalg", "mlx.core.linalg: linear algebra routines.");

  m.def(
      "norm",
      [](const array& a,
         const std::variant<std::monostate, int, double, std::string>& ord_,
         const std::variant<std::monostate, int, std::vector<int>>& axis_,
         const bool keepdims,
         const StreamOrDevice stream) {
        std::optional<std::vector<int>> axis = std::nullopt;
        if (auto pv = std::get_if<int>(&axis_); pv) {
          axis = std::vector<int>{*pv};
        } else if (auto pv = std::get_if<std::vector<int>>(&axis_); pv) {
          axis = *pv;
        }

        if (std::holds_alternative<std::monostate>(ord_)) {
          return norm(a, axis, keepdims, stream);
        } else {
          if (auto pv = std::get_if<std::string>(&ord_); pv) {
            return norm(a, *pv, axis, keepdims, stream);
          }
          double ord;
          if (auto pv = std::get_if<int>(&ord_); pv) {
            ord = *pv;
          } else {
            ord = std::get<double>(ord_);
          }
          return norm(a, ord, axis, keepdims, stream);
        }
      },
      nb::arg(),
      "ord"_a = nb::none(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def norm(a: array, /, ord: Union[None, scalar, str] = None, axis: Union[None, int, List[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Matrix or vector norm.

        This function computes vector or  matrix norms depending on the value of
        the ``ord`` and ``axis`` parameters.

        Args:
          a (array): Input array.  If ``axis`` is ``None``, ``a`` must be 1-D or 2-D,
            unless ``ord`` is ``None``. If both ``axis`` and ``ord`` are ``None``, the
            2-norm of ``a.flatten`` will be returned.
          ord (scalar or str, optional): Order of the norm (see table under ``Notes``).
            If ``None``, the 2-norm (or Frobenius norm for matrices) will be computed
            along the given ``axis``.  Default: ``None``.
          axis (int or list(int), optional): If ``axis`` is an integer, it specifies the
            axis of ``a`` along which to compute the vector norms.  If ``axis`` is a
            2-tuple, it specifies the axes that hold 2-D matrices, and the matrix
            norms of these matrices are computed. If `axis` is ``None`` then
            either a vector norm (when ``a`` is 1-D) or a matrix norm (when ``a`` is
            2-D) is returned. Default: ``None``.
          keepdims (bool, optional): If ``True``, the axes which are normed over are
            left in the result as dimensions with size one. Default ``False``.

        Returns:
          array: The output containing the norm(s).

        Notes:
          For values of ``ord < 1``, the result is, strictly speaking, not a
          mathematical norm, but it may still be useful for various numerical
          purposes.

          The following norms can be calculated:

          =====  ============================  ==========================
          ord    norm for matrices             norm for vectors
          =====  ============================  ==========================
          None   Frobenius norm                2-norm
          'fro'  Frobenius norm                --
          inf    max(sum(abs(x), axis=1))      max(abs(x))
          -inf   min(sum(abs(x), axis=1))      min(abs(x))
          0      --                            sum(x != 0)
          1      max(sum(abs(x), axis=0))      as below
          -1     min(sum(abs(x), axis=0))      as below
          2      2-norm (largest sing. value)  as below
          -2     smallest singular value       as below
          other  --                            sum(abs(x)**ord)**(1./ord)
          =====  ============================  ==========================

          .. warning::
            Nuclear norm and norms based on singular values are not yet implemented.

          The Frobenius norm is given by [1]_:

              :math:`||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

          The nuclear norm is the sum of the singular values.

          Both the Frobenius and nuclear norm orders are only defined for
          matrices and raise a ``ValueError`` when ``a.ndim != 2``.

        References:
          .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
                 Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

        Examples:
          >>> import mlx.core as mx
          >>> from mlx.core import linalg as la
          >>> a = mx.arange(9) - 4
          >>> a
          array([-4, -3, -2, ..., 2, 3, 4], dtype=int32)
          >>> b = a.reshape((3,3))
          >>> b
          array([[-4, -3, -2],
                 [-1,  0,  1],
                 [ 2,  3,  4]], dtype=int32)
          >>> la.norm(a)
          array(7.74597, dtype=float32)
          >>> la.norm(b)
          array(7.74597, dtype=float32)
          >>> la.norm(b, 'fro')
          array(7.74597, dtype=float32)
          >>> la.norm(a, float("inf"))
          array(4, dtype=float32)
          >>> la.norm(b, float("inf"))
          array(9, dtype=float32)
          >>> la.norm(a, -float("inf"))
          array(0, dtype=float32)
          >>> la.norm(b, -float("inf"))
          array(2, dtype=float32)
          >>> la.norm(a, 1)
          array(20, dtype=float32)
          >>> la.norm(b, 1)
          array(7, dtype=float32)
          >>> la.norm(a, -1)
          array(0, dtype=float32)
          >>> la.norm(b, -1)
          array(6, dtype=float32)
          >>> la.norm(a, 2)
          array(7.74597, dtype=float32)
          >>> la.norm(a, 3)
          array(5.84804, dtype=float32)
          >>> la.norm(a, -3)
          array(0, dtype=float32)
          >>> c = mx.array([[ 1, 2, 3],
          ...               [-1, 1, 4]])
          >>> la.norm(c, axis=0)
          array([1.41421, 2.23607, 5], dtype=float32)
          >>> la.norm(c, axis=1)
          array([3.74166, 4.24264], dtype=float32)
          >>> la.norm(c, ord=1, axis=1)
          array([6, 6], dtype=float32)
          >>> m = mx.arange(8).reshape(2,2,2)
          >>> la.norm(m, axis=(1,2))
          array([3.74166, 11.225], dtype=float32)
          >>> la.norm(m[0, :, :]), LA.norm(m[1, :, :])
          (array(3.74166, dtype=float32), array(11.225, dtype=float32))
      )pbdoc");
  m.def(
      "qr",
      &qr,
      "a"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def qr(a: array, *, stream: Union[None, Stream, Device] = None) -> (array, array)"),
      R"pbdoc(
        The QR factorization of the input matrix.

        This function supports arrays with at least 2 dimensions. The matrices
        which are factorized are assumed to be in the last two dimensions of
        the input.

        Args:
            a (array): Input array.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            tuple(array, array): The ``Q`` and ``R`` matrices.

        Example:
            >>> A = mx.array([[2., 3.], [1., 2.]])
            >>> Q, R = mx.linalg.qr(A, stream=mx.cpu)
            >>> Q
            array([[-0.894427, -0.447214],
                   [-0.447214, 0.894427]], dtype=float32)
            >>> R
            array([[-2.23607, -3.57771],
                   [0, 0.447214]], dtype=float32)
      )pbdoc");
  m.def(
      "svd",
      &svd_helper,
      "a"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def svd(a: array, *, stream: Union[None, Stream, Device] = None) -> (array, array, array)"),
      R"pbdoc(
        The Singular Value Decomposition (SVD) of the input matrix.

        This function supports arrays with at least 2 dimensions. When the input
        has more than two dimensions, the function iterates over all indices of the first
        a.ndim - 2 dimensions and for each combination SVD is applied to the last two indices.

        Args:
            a (array): Input array.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            tuple(array, array, array): The ``U``, ``S``, and ``Vt`` matrices, such that
              ``A = U @ diag(S) @ Vt``
      )pbdoc");
  m.def(
      "inv",
      &inv,
      "a"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def inv(a: array, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Compute the inverse of a square matrix.

        This function supports arrays with at least 2 dimensions. When the input
        has more than two dimensions, the inverse is computed for each matrix
        in the last two dimensions of ``a``.

        Args:
            a (array): Input array.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            array: ``ainv`` such that ``dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])``
      )pbdoc");
}
