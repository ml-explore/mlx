
// Copyright © 2023 Apple Inc.

#include <limits>
#include <numeric>
#include <ostream>
#include <string>
#include <variant>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlx/linalg.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

#include "python/src/load.h"
#include "python/src/overloaded.h"
#include "python/src/utils.h"

namespace py = pybind11;
using namespace py::literals;

using namespace mlx::core;
using namespace mlx::core::linalg;

void init_linalg(py::module_& parent_module) {
  auto m =
      parent_module.def_submodule("linalg", "mlx.core.linalg: Linear Algebra.");

  m.def(
      "norm",
      [](const array& a,
         const std::variant<std::monostate, int, double, std::string>& ord,
         const std::variant<std::monostate, int, std::vector<int>>& axis,
         const bool keepdims,
         const StreamOrDevice stream) {
        return std::visit(
            overloaded{
                [&](const double p) {
                  if (std::isinf((float)p) || std::isinf(p)) {
                    if (p > 0) {
                      return norm(
                          a,
                          "inf",
                          get_reduce_axes(axis, a.ndim()),
                          keepdims,
                          stream);
                    }
                    return norm(
                        a,
                        "-inf",
                        get_reduce_axes(axis, a.ndim()),
                        keepdims,
                        stream);
                  }
                  return norm(
                      a, p, get_reduce_axes(axis, a.ndim()), keepdims, stream);
                },
                [&](const std::string& p) {
                  return norm(
                      a, p, get_reduce_axes(axis, a.ndim()), keepdims, stream);
                },
                [&](const std::monostate _) {
                  return norm(
                      a, get_reduce_axes(axis, a.ndim()), keepdims, stream);
                }},
            ord);
      },
      "a"_a,
      "ord"_a = none,
      "axis"_a = none,
      "keepdims"_a = false,
      "stream"_a = none,
      R"pbdoc(
    Matrix or vector norm.

    This function is able to return matrix or vector norms,
    depending on the value of the ``ord`` parameter.

    Parameters
    ----------
    a : array_like
        Input array.  If `axis` is None, `a` must be 1-D or 2-D, unless `ord`
        is None. If both `axis` and `ord` are None, the 2-norm of ``a.flatten`` will be returned.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes``). inf means float(`inf`) object. The default is None.
    axis : {None, int, 2-tuple of ints}, optional.
        If `axis` is an integer, it specifies the axis of `a` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `a`
        is 1-D) or a matrix norm (when `a` is 2-D) is returned. The default
        is None.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `a`.

    Returns
    -------
    n : array
        Norm of the matrix or vector(s).

    Notes
    -----
    For values of ``ord < 1``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
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

    Nuclear norm and norms based on singular values are not yet implemented.

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    Both the Frobenius and nuclear norm orders are only defined for
    matrices and raise a ValueError when ``a.ndim != 2``.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    Examples
    --------
    >>> import mlx.core as mx
    >>> from mlx.core import linalg as LA
    >>> a = mx.arange(9) - 4
    >>> a
    array([-4, -3, -2, ..., 2, 3, 4], dtype=int32)
    >>> b = a.reshape((3,3))
    >>> b
    array([[-4, -3, -2],
           [-1,  0,  1],
           [ 2,  3,  4]], dtype=int32)
    >>> LA.norm(a)
    array(7.74597, dtype=float32)
    >>> LA.norm(b)
    array(7.74597, dtype=float32)
    >>> LA.norm(b, 'fro')
    array(7.74597, dtype=float32)
    >>> LA.norm(a, float("inf"))
    array(4, dtype=int32)
    >>> LA.norm(b, float("inf"))
    array(9, dtype=int32)
    >>> LA.norm(a, -float("inf"))
    array(0, dtype=int32)
    >>> LA.norm(b, -float("inf"))
    array(2, dtype=int32)
    >>> LA.norm(a, 1)
    array(20, dtype=int32)
    >>> LA.norm(b, 1)
    array(7, dtype=int32)
    >>> LA.norm(a, -1)
    array(0, dtype=float32)
    >>> LA.norm(b, -1)
    array(6, dtype=int32)
    >>> LA.norm(a, 2)
    array(7.74597, dtype=float32)
    >>> LA.norm(a, 3)
    array(5.84804, dtype=float32)
    >>> LA.norm(a, -3)
    array(0, dtype=float32)
    >>> c = mx.array([[ 1, 2, 3],
    ...               [-1, 1, 4]])
    >>> LA.norm(c, axis=0)
    array([1.41421, 2.23607, 5], dtype=float32)
    >>> LA.norm(c, axis=1)
    array([3.74166, 4.24264], dtype=float32)
    >>> LA.norm(c, ord=1, axis=1)
    array([6, 6], dtype=int32)
    >>> m = mx.arange(8).reshape(2,2,2)
    >>> LA.norm(m, axis=(1,2))
    array([3.74166, 11.225], dtype=float32)
    >>> LA.norm(m[0, :, :]), LA.norm(m[1, :, :])
    (array(3.74166, dtype=float32), array(11.225, dtype=float32))
    )pbdoc");
}
