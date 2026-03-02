// Copyright © 2023-2024 Apple Inc.

#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/linalg.h"
#include "python/src/small_vector.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

void init_linalg(nb::module_& parent_module) {
  auto m = parent_module.def_submodule(
      "linalg", "mlx.core.linalg: linear algebra routines.");

  m.def(
      "norm",
      [](const mx::array& a,
         const std::variant<std::monostate, int, double, std::string>& ord_,
         const std::variant<std::monostate, int, std::vector<int>>& axis_,
         const bool keepdims,
         const mx::StreamOrDevice stream) {
        std::optional<std::vector<int>> axis = std::nullopt;
        if (auto pv = std::get_if<int>(&axis_); pv) {
          axis = std::vector<int>{*pv};
        } else if (auto pv = std::get_if<std::vector<int>>(&axis_); pv) {
          axis = *pv;
        }

        if (std::holds_alternative<std::monostate>(ord_)) {
          return mx::linalg::norm(a, axis, keepdims, stream);
        } else {
          if (auto pv = std::get_if<std::string>(&ord_); pv) {
            return mx::linalg::norm(a, *pv, axis, keepdims, stream);
          }
          double ord;
          if (auto pv = std::get_if<int>(&ord_); pv) {
            ord = *pv;
          } else {
            ord = std::get<double>(ord_);
          }
          return mx::linalg::norm(a, ord, axis, keepdims, stream);
        }
      },
      nb::arg(),
      "ord"_a = nb::none(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def norm(a: array, /, ord: Union[None, int, float, str] = None, axis: Union[None, int, list[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Matrix or vector norm.

        This function computes vector or  matrix norms depending on the value of
        the ``ord`` and ``axis`` parameters.

        Args:
          a (array): Input array.  If ``axis`` is ``None``, ``a`` must be 1-D or 2-D,
            unless ``ord`` is ``None``. If both ``axis`` and ``ord`` are ``None``, the
            2-norm of ``a.flatten`` will be returned.
          ord (int, float or str, optional): Order of the norm (see table under ``Notes``).
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
          'nuc'  nuclear norm                  --
          inf    max(sum(abs(x), axis=1))      max(abs(x))
          -inf   min(sum(abs(x), axis=1))      min(abs(x))
          0      --                            sum(x != 0)
          1      max(sum(abs(x), axis=0))      as below
          -1     min(sum(abs(x), axis=0))      as below
          2      2-norm (largest sing. value)  as below
          -2     smallest singular value       as below
          other  --                            sum(abs(x)**ord)**(1./ord)
          =====  ============================  ==========================

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
      &mx::linalg::qr,
      "a"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def qr(a: array, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array]"),
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
            tuple(array, array): ``Q`` and ``R`` matrices such that ``Q @ R = a``.

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
      [](const mx::array& a,
         bool compute_uv /* = true */,
         mx::StreamOrDevice s /* = {} */) -> nb::object {
        const auto result = mx::linalg::svd(a, compute_uv, s);
        if (result.size() == 1) {
          return nb::cast(result.at(0));
        } else {
          return nb::make_tuple(result.at(0), result.at(1), result.at(2));
        }
      },
      "a"_a,
      "compute_uv"_a = true,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def svd(a: array, compute_uv: bool = True, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array, array]"),
      R"pbdoc(
        The Singular Value Decomposition (SVD) of the input matrix.

        This function supports arrays with at least 2 dimensions. When the input
        has more than two dimensions, the function iterates over all indices of the first
        a.ndim - 2 dimensions and for each combination SVD is applied to the last two indices.

        Args:
            a (array): Input array.
            compute_uv (bool, optional): If ``True``, return the ``U``, ``S``, and ``Vt`` components.
              If ``False``, return only the ``S`` array. Default: ``True``.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            Union[tuple(array, ...), array]:
              If compute_uv is ``True`` returns the ``U``, ``S``, and ``Vt`` matrices, such that
              ``A = U @ diag(S) @ Vt``. If compute_uv is ``False`` returns singular values array ``S``.
      )pbdoc");
  m.def(
      "inv",
      &mx::linalg::inv,
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
  m.def(
      "tri_inv",
      &mx::linalg::tri_inv,
      "a"_a,
      "upper"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tri_inv(a: array, upper: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Compute the inverse of a triangular square matrix.

        This function supports arrays with at least 2 dimensions. When the input
        has more than two dimensions, the inverse is computed for each matrix
        in the last two dimensions of ``a``.

        Args:
            a (array): Input array.
            upper (bool, optional): Whether the array is upper or lower triangular. Defaults to ``False``.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            array: ``ainv`` such that ``dot(a, ainv) = dot(ainv, a) = eye(a.shape[0])``
      )pbdoc");
  m.def(
      "cholesky",
      &mx::linalg::cholesky,
      "a"_a,
      "upper"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def cholesky(a: array, upper: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Compute the Cholesky decomposition of a real symmetric positive semi-definite matrix.

        This function supports arrays with at least 2 dimensions. When the input
        has more than two dimensions, the Cholesky decomposition is computed for each matrix
        in the last two dimensions of ``a``.

        If the input matrix is not symmetric positive semi-definite, behaviour is undefined.

        Args:
            a (array): Input array.
            upper (bool, optional): If ``True``, return the upper triangular Cholesky factor.
              If ``False``, return the lower triangular Cholesky factor. Default: ``False``.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
          array: If ``upper = False``, it returns a lower triangular ``L`` matrix such
          that ``L @ L.T = a``.  If ``upper = True``, it returns an upper triangular
          ``U`` matrix such that ``U.T @ U = a``.
      )pbdoc");
  m.def(
      "cholesky_inv",
      &mx::linalg::cholesky_inv,
      "a"_a,
      "upper"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def cholesky_inv(L: array, upper: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Compute the inverse of a real symmetric positive semi-definite matrix using it's Cholesky decomposition.

        Let :math:`\mathbf{A}` be a real symmetric positive semi-definite matrix and :math:`\mathbf{L}` its Cholesky decomposition such that:

        .. math::

          \begin{aligned}
            \mathbf{A} = \mathbf{L}\mathbf{L}^T
          \end{aligned}

        This function computes :math:`\mathbf{A}^{-1}`.

        This function supports arrays with at least 2 dimensions. When the input
        has more than two dimensions, the Cholesky inverse is computed for each matrix
        in the last two dimensions of :math:`\mathbf{L}`.

        If the input matrix is not a triangular matrix behaviour is undefined.

        Args:
            L (array): Input array.
            upper (bool, optional): If ``True``, return the upper triangular Cholesky factor.
              If ``False``, return the lower triangular Cholesky factor. Default: ``False``.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
          array: :math:`\mathbf{A^{-1}}` where :math:`\mathbf{A} = \mathbf{L}\mathbf{L}^T`.
      )pbdoc");
  m.def(
      "pinv",
      &mx::linalg::pinv,
      "a"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def pinv(a: array, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Compute the (Moore-Penrose) pseudo-inverse of a matrix.

        This function calculates a generalized inverse of a matrix using its
        singular-value decomposition. This function supports arrays with at least 2 dimensions.
        When the input has more than two dimensions, the inverse is computed for each
        matrix in the last two dimensions of ``a``.

        Args:
            a (array): Input array.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            array: ``aplus`` such that ``a @ aplus @ a = a``
      )pbdoc");
  m.def(
      "cross",
      &mx::linalg::cross,
      "a"_a,
      "b"_a,
      "axis"_a = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def cross(a: array, b: array, axis: int = -1, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Compute the cross product of two arrays along a specified axis.

        The cross product is defined for arrays with size 2 or 3 in the
        specified axis. If the size is 2 then the third value is assumed
        to be zero.

        Args:
            a (array): Input array.
            b (array): Input array.
            axis (int, optional): Axis along which to compute the cross
              product. Default: ``-1``.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            array: The cross product of ``a`` and ``b`` along the specified axis.
      )pbdoc");
  m.def(
      "eigvals",
      &mx::linalg::eigvals,
      "a"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"pbdoc(
        Compute the eigenvalues of a square matrix.

        This function differs from :func:`numpy.linalg.eigvals` in that the
        return type is always complex even if the eigenvalues are all real.

        This function supports arrays with at least 2 dimensions. When the
        input has more than two dimensions, the eigenvalues are computed for
        each matrix in the last two dimensions.

        Args:
            a (array): The input array.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            array: The eigenvalues (not necessarily in order).

        Example:
            >>> A = mx.array([[1., -2.], [-2., 1.]])
            >>> eigenvalues = mx.linalg.eigvals(A, stream=mx.cpu)
            >>> eigenvalues
            array([3+0j, -1+0j], dtype=complex64)
      )pbdoc");
  m.def(
      "eig",
      [](const mx::array& a, mx::StreamOrDevice s) {
        auto result = mx::linalg::eig(a, s);
        return nb::make_tuple(result.first, result.second);
      },
      "a"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def eig(a: array, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array]"),
      R"pbdoc(
        Compute the eigenvalues and eigenvectors of a square matrix.

        This function differs from :func:`numpy.linalg.eig` in that the
        return type is always complex even if the eigenvalues are all real.

        This function supports arrays with at least 2 dimensions. When the input
        has more than two dimensions, the eigenvalues and eigenvectors are
        computed for each matrix in the last two dimensions.

        Args:
            a (array): The input array.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            Tuple[array, array]:
              A tuple containing the eigenvalues and the normalized right
              eigenvectors. The column ``v[:, i]`` is the eigenvector
              corresponding to the i-th eigenvalue.

        Example:
            >>> A = mx.array([[1., -2.], [-2., 1.]])
            >>> w, v = mx.linalg.eig(A, stream=mx.cpu)
            >>> w
            array([3+0j, -1+0j], dtype=complex64)
            >>> v
            array([[0.707107+0j, 0.707107+0j],
                   [-0.707107+0j, 0.707107+0j]], dtype=complex64)
      )pbdoc");

  m.def(
      "eigvalsh",
      &mx::linalg::eigvalsh,
      "a"_a,
      "UPLO"_a = "L",
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"pbdoc(
        Compute the eigenvalues of a complex Hermitian or real symmetric matrix.

        This function supports arrays with at least 2 dimensions. When the
        input has more than two dimensions, the eigenvalues are computed for
        each matrix in the last two dimensions.

        Args:
            a (array): Input array. Must be a real symmetric or complex
              Hermitian matrix.
            UPLO (str, optional): Whether to use the upper (``"U"``) or
              lower (``"L"``) triangle of the matrix.  Default: ``"L"``.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            array: The eigenvalues in ascending order.

        Note:
            The input matrix is assumed to be symmetric (or Hermitian). Only
            the selected triangle is used. No checks for symmetry are performed.

        Example:
            >>> A = mx.array([[1., -2.], [-2., 1.]])
            >>> eigenvalues = mx.linalg.eigvalsh(A, stream=mx.cpu)
            >>> eigenvalues
            array([-1., 3.], dtype=float32)
      )pbdoc");
  m.def(
      "eigh",
      [](const mx::array& a, const std::string& UPLO, mx::StreamOrDevice s) {
        auto result = mx::linalg::eigh(a, UPLO, s);
        return nb::make_tuple(result.first, result.second);
      },
      "a"_a,
      "UPLO"_a = "L",
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def eigh(a: array, UPLO: str = 'L', *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array]"),
      R"pbdoc(
        Compute the eigenvalues and eigenvectors of a complex Hermitian or
        real symmetric matrix.

        This function supports arrays with at least 2 dimensions. When the input
        has more than two dimensions, the eigenvalues and eigenvectors are
        computed for each matrix in the last two dimensions.

        Args:
            a (array): Input array. Must be a real symmetric or complex
              Hermitian matrix.
            UPLO (str, optional): Whether to use the upper (``"U"``) or
               lower (``"L"``) triangle of the matrix.  Default: ``"L"``.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            Tuple[array, array]:
              A tuple containing the eigenvalues in ascending order and
              the normalized eigenvectors. The column ``v[:, i]`` is the
              eigenvector corresponding to the i-th eigenvalue.

        Note:
            The input matrix is assumed to be symmetric (or Hermitian). Only
            the selected triangle is used. No checks for symmetry are performed.

        Example:
            >>> A = mx.array([[1., -2.], [-2., 1.]])
            >>> w, v = mx.linalg.eigh(A, stream=mx.cpu)
            >>> w
            array([-1., 3.], dtype=float32)
            >>> v
            array([[ 0.707107, -0.707107],
                  [ 0.707107,  0.707107]], dtype=float32)
      )pbdoc");
  m.def(
      "lu",
      [](const mx::array& a, mx::StreamOrDevice s /* = {} */) {
        auto result = mx::linalg::lu(a, s);
        return nb::make_tuple(result.at(0), result.at(1), result.at(2));
      },
      "a"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def lu(a: array, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array, array]"),
      R"pbdoc(
        Compute the LU factorization of the given matrix ``A``.

        Note, unlike the default behavior of ``scipy.linalg.lu``, the pivots
        are indices. To reconstruct the input use ``L[P, :] @ U`` for 2
        dimensions or ``mx.take_along_axis(L, P[..., None], axis=-2) @ U``
        for more than 2 dimensions.

        To construct the full permuation matrix do:

        .. code-block::

          P = mx.put_along_axis(mx.zeros_like(L), p[..., None], mx.array(1.0), axis=-1)

        Args:
            a (array): Input array.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            tuple(array, array, array):
              The ``p``, ``L``, and ``U`` arrays, such that ``A = L[P, :] @ U``
      )pbdoc");
  m.def(
      "lu_factor",
      &mx::linalg::lu_factor,
      "a"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def lu_factor(a: array, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array]"),
      R"pbdoc(
        Computes a compact representation of the LU factorization.

        Args:
            a (array): Input array.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            tuple(array, array): The ``LU`` matrix and ``pivots`` array.
      )pbdoc");
  m.def(
      "solve",
      &mx::linalg::solve,
      "a"_a,
      "b"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def solve(a: array, b: array, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Compute the solution to a system of linear equations ``AX = B``.

        Args:
            a (array): Input array.
            b (array): Input array.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            array: The unique solution to the system ``AX = B``.
      )pbdoc");
  m.def(
      "solve_triangular",
      &mx::linalg::solve_triangular,
      "a"_a,
      "b"_a,
      nb::kw_only(),
      "upper"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def solve_triangular(a: array, b: array, *, upper: bool = False, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Computes the solution of a triangular system of linear equations ``AX = B``.

        Args:
            a (array): Input array.
            b (array): Input array.
            upper (bool, optional): Whether the array is upper or lower
              triangular. Default: ``False``.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            array: The unique solution to the system ``AX = B``.
      )pbdoc");
}
