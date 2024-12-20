// Copyright © 2023-2024 Apple Inc.

#include <numeric>
#include <ostream>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/einsum.h"
#include "mlx/ops.h"
#include "mlx/utils.h"
#include "python/src/load.h"
#include "python/src/utils.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

using Scalar = std::variant<bool, int, double>;

mx::Dtype scalar_to_dtype(Scalar s) {
  if (std::holds_alternative<int>(s)) {
    return mx::int32;
  } else if (std::holds_alternative<double>(s)) {
    return mx::float32;
  } else {
    return mx::bool_;
  }
}

double scalar_to_double(Scalar s) {
  if (auto pv = std::get_if<int>(&s); pv) {
    return static_cast<double>(*pv);
  } else if (auto pv = std::get_if<double>(&s); pv) {
    return *pv;
  } else {
    return static_cast<double>(std::get<bool>(s));
  }
}

void init_ops(nb::module_& m) {
  m.def(
      "reshape",
      &mx::reshape,
      nb::arg(),
      "shape"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def reshape(a: array, /, shape: Sequence[int], *, stream: "
              "Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Reshape an array while preserving the size.

        Args:
            a (array): Input array.
            shape (tuple(int)): New shape.
            stream (Stream, optional): Stream or device. Defaults to ``None``
              in which case the default stream of the default device is used.

        Returns:
            array: The reshaped array.
      )pbdoc");
  m.def(
      "flatten",
      [](const mx::array& a,
         int start_axis,
         int end_axis,
         const mx::StreamOrDevice& s) {
        return mx::flatten(a, start_axis, end_axis);
      },
      nb::arg(),
      "start_axis"_a = 0,
      "end_axis"_a = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def flatten(a: array, /, start_axis: int = 0, end_axis: int = "
              "-1, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
      Flatten an array.

      The axes flattened will be between ``start_axis`` and ``end_axis``,
      inclusive. Negative axes are supported. After converting negative axis to
      positive, axes outside the valid range will be clamped to a valid value,
      ``start_axis`` to ``0`` and ``end_axis`` to ``ndim - 1``.

      Args:
          a (array): Input array.
          start_axis (int, optional): The first dimension to flatten. Defaults to ``0``.
          end_axis (int, optional): The last dimension to flatten. Defaults to ``-1``.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

      Returns:
          array: The flattened array.

      Example:
          >>> a = mx.array([[1, 2], [3, 4]])
          >>> mx.flatten(a)
          array([1, 2, 3, 4], dtype=int32)
          >>>
          >>> mx.flatten(a, start_axis=0, end_axis=-1)
          array([1, 2, 3, 4], dtype=int32)
  )pbdoc");
  m.def(
      "unflatten",
      &mx::unflatten,
      nb::arg(),
      "axis"_a,
      "shape"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def unflatten(a: array, /, axis: int, shape: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
      Unflatten an axis of an array to a shape.

      Args:
          a (array): Input array.
          axis (int): The axis to unflatten.
          shape (tuple(int)): The shape to unflatten to. At most one
            entry can be ``-1`` in which case the corresponding size will be
            inferred.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

      Returns:
          array: The unflattened array.

      Example:
          >>> a = mx.array([1, 2, 3, 4])
          >>> mx.unflatten(a, 0, (2, -1))
          array([[1, 2], [3, 4]], dtype=int32)
  )pbdoc");
  m.def(
      "squeeze",
      [](const mx::array& a, const IntOrVec& v, const mx::StreamOrDevice& s) {
        if (std::holds_alternative<std::monostate>(v)) {
          return mx::squeeze(a, s);
        } else if (auto pv = std::get_if<int>(&v); pv) {
          return mx::squeeze(a, *pv, s);
        } else {
          return mx::squeeze(a, std::get<std::vector<int>>(v), s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def squeeze(a: array, /, axis: Union[None, int, Sequence[int]] = "
          "None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Remove length one axes from an array.

        Args:
            a (array): Input array.
            axis (int or tuple(int), optional): Axes to remove. Defaults
              to ``None`` in which case all size one axes are removed.

        Returns:
            array: The output array with size one axes removed.
      )pbdoc");
  m.def(
      "expand_dims",
      [](const mx::array& a,
         const std::variant<int, std::vector<int>>& v,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&v); pv) {
          return mx::expand_dims(a, *pv, s);
        } else {
          return mx::expand_dims(a, std::get<std::vector<int>>(v), s);
        }
      },
      nb::arg(),
      "axis"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig("def expand_dims(a: array, /, axis: Union[int, Sequence[int]], "
              "*, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Add a size one dimension at the given axis.

        Args:
            a (array): Input array.
            axes (int or tuple(int)): The index of the inserted dimensions.

        Returns:
            array: The array with inserted dimensions.
      )pbdoc");
  m.def(
      "abs",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::abs(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def abs(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise absolute value.

        Args:
            a (array): Input array.

        Returns:
            array: The absolute value of ``a``.
      )pbdoc");
  m.def(
      "sign",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sign(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sign(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise sign.

        Args:
            a (array): Input array.

        Returns:
            array: The sign of ``a``.
      )pbdoc");
  m.def(
      "negative",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::negative(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def negative(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise negation.

        Args:
            a (array): Input array.

        Returns:
            array: The negative of ``a``.
      )pbdoc");
  m.def(
      "add",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::add(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def add(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise addition.

        Add two arrays with numpy-style broadcasting semantics. Either or both input arrays
        can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The sum of ``a`` and ``b``.
      )pbdoc");
  m.def(
      "subtract",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::subtract(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def subtract(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise subtraction.

        Subtract one array from another with numpy-style broadcasting semantics. Either or both
        input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The difference ``a - b``.
      )pbdoc");
  m.def(
      "multiply",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::multiply(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def multiply(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise multiplication.

        Multiply two arrays with numpy-style broadcasting semantics. Either or both
        input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The multiplication ``a * b``.
      )pbdoc");
  m.def(
      "divide",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::divide(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def divide(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise division.

        Divide two arrays with numpy-style broadcasting semantics. Either or both
        input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The quotient ``a / b``.
      )pbdoc");
  m.def(
      "divmod",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::divmod(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def divmod(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise quotient and remainder.

        The fuction ``divmod(a, b)`` is equivalent to but faster than
        ``(a // b, a % b)``. The function uses numpy-style broadcasting
        semantics. Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            tuple(array, array): The quotient ``a // b`` and remainder ``a % b``.
      )pbdoc");
  m.def(
      "floor_divide",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::floor_divide(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def floor_divide(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise integer division.

        If either array is a floating point type then it is equivalent to
        calling :func:`floor` after :func:`divide`.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The quotient ``a // b``.
      )pbdoc");
  m.def(
      "remainder",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::remainder(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def remainder(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise remainder of division.

        Computes the remainder of dividing a with b with numpy-style
        broadcasting semantics. Either or both input arrays can also be
        scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The remainder of ``a // b``.
      )pbdoc");
  m.def(
      "equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise equality.

        Equality comparison on two arrays with numpy-style broadcasting semantics.
        Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The element-wise comparison ``a == b``.
      )pbdoc");
  m.def(
      "not_equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::not_equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def not_equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise not equal.

        Not equal comparison on two arrays with numpy-style broadcasting semantics.
        Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The element-wise comparison ``a != b``.
      )pbdoc");
  m.def(
      "less",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::less(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def less(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise less than.

        Strict less than on two arrays with numpy-style broadcasting semantics.
        Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The element-wise comparison ``a < b``.
      )pbdoc");
  m.def(
      "less_equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::less_equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def less_equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise less than or equal.

        Less than or equal on two arrays with numpy-style broadcasting semantics.
        Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The element-wise comparison ``a <= b``.
      )pbdoc");
  m.def(
      "greater",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::greater(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def greater(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise greater than.

        Strict greater than on two arrays with numpy-style broadcasting semantics.
        Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The element-wise comparison ``a > b``.
      )pbdoc");
  m.def(
      "greater_equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::greater_equal(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def greater_equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise greater or equal.

        Greater than or equal on two arrays with numpy-style broadcasting semantics.
        Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The element-wise comparison ``a >= b``.
      )pbdoc");
  m.def(
      "array_equal",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         bool equal_nan,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::array_equal(a, b, equal_nan, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "equal_nan"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def array_equal(a: Union[scalar, array], b: Union[scalar, array], equal_nan: bool = False, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Array equality check.

        Compare two arrays for equality. Returns ``True`` if and only if the arrays
        have the same shape and their values are equal. The arrays need not have
        the same type to be considered equal.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.
            equal_nan (bool): If ``True``, NaNs are considered equal.
              Defaults to ``False``.

        Returns:
            array: A scalar boolean array.
      )pbdoc");
  m.def(
      "matmul",
      &mx::matmul,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def matmul(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Matrix multiplication.

        Perform the (possibly batched) matrix multiplication of two arrays. This function supports
        broadcasting for arrays with more than two dimensions.

        - If the first array is 1-D then a 1 is prepended to its shape to make it
          a matrix. Similarly if the second array is 1-D then a 1 is appended to its
          shape to make it a matrix. In either case the singleton dimension is removed
          from the result.
        - A batched matrix multiplication is performed if the arrays have more than
          2 dimensions.  The matrix dimensions for the matrix product are the last
          two dimensions of each input.
        - All but the last two dimensions of each input are broadcast with one another using
          standard numpy-style broadcasting semantics.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The matrix product of ``a`` and ``b``.
      )pbdoc");
  m.def(
      "square",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::square(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def square(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise square.

        Args:
            a (array): Input array.

        Returns:
            array: The square of ``a``.
      )pbdoc");
  m.def(
      "sqrt",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sqrt(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sqrt(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise square root.

        Args:
            a (array): Input array.

        Returns:
            array: The square root of ``a``.
      )pbdoc");
  m.def(
      "rsqrt",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::rsqrt(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def rsqrt(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise reciprocal and square root.

        Args:
            a (array): Input array.

        Returns:
            array: One over the square root of ``a``.
      )pbdoc");
  m.def(
      "reciprocal",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::reciprocal(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def reciprocal(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise reciprocal.

        Args:
            a (array): Input array.

        Returns:
            array: The reciprocal of ``a``.
      )pbdoc");
  m.def(
      "logical_not",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::logical_not(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logical_not(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise logical not.

        Args:
            a (array): Input array or scalar.

        Returns:
            array: The boolean array containing the logical not of ``a``.
      )pbdoc");
  m.def(
      "logical_and",
      [](const ScalarOrArray& a, const ScalarOrArray& b, mx::StreamOrDevice s) {
        return mx::logical_and(to_array(a), to_array(b), s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logical_and(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise logical and.

        Args:
            a (array): First input array or scalar.
            b (array): Second input array or scalar.

        Returns:
            array: The boolean array containing the logical and of ``a`` and ``b``.
    )pbdoc");

  m.def(
      "logical_or",
      [](const ScalarOrArray& a, const ScalarOrArray& b, mx::StreamOrDevice s) {
        return mx::logical_or(to_array(a), to_array(b), s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logical_or(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise logical or.

        Args:
            a (array): First input array or scalar.
            b (array): Second input array or scalar.

        Returns:
            array: The boolean array containing the logical or of ``a`` and ``b``.
    )pbdoc");
  m.def(
      "logaddexp",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::logaddexp(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logaddexp(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise log-add-exp.

        This is a numerically stable log-add-exp of two arrays with numpy-style
        broadcasting semantics. Either or both input arrays can also be scalars.

        The computation is is a numerically stable version of ``log(exp(a) + exp(b))``.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The log-add-exp of ``a`` and ``b``.
      )pbdoc");
  m.def(
      "exp",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::exp(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def exp(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise exponential.

        Args:
            a (array): Input array.

        Returns:
            array: The exponential of ``a``.
      )pbdoc");
  m.def(
      "expm1",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::expm1(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def expm1(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise exponential minus 1.

        Computes ``exp(x) - 1`` with greater precision for small ``x``.

        Args:
            a (array): Input array.

        Returns:
            array: The expm1 of ``a``.
      )pbdoc");
  m.def(
      "erf",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::erf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def erf(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise error function.

        .. math::
          \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} \, dt

        Args:
            a (array): Input array.

        Returns:
            array: The error function of ``a``.
      )pbdoc");
  m.def(
      "erfinv",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::erfinv(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def erfinv(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise inverse of :func:`erf`.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse error function of ``a``.
      )pbdoc");
  m.def(
      "sin",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sin(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sin(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise sine.

        Args:
            a (array): Input array.

        Returns:
            array: The sine of ``a``.
      )pbdoc");
  m.def(
      "cos",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::cos(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def cos(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise cosine.

        Args:
            a (array): Input array.

        Returns:
            array: The cosine of ``a``.
      )pbdoc");
  m.def(
      "tan",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::tan(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tan(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise tangent.

        Args:
            a (array): Input array.

        Returns:
            array: The tangent of ``a``.
      )pbdoc");
  m.def(
      "arcsin",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arcsin(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arcsin(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise inverse sine.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse sine of ``a``.
      )pbdoc");
  m.def(
      "arccos",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arccos(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arccos(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise inverse cosine.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse cosine of ``a``.
      )pbdoc");
  m.def(
      "arctan",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arctan(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arctan(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise inverse tangent.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse tangent of ``a``.
      )pbdoc");
  m.def(
      "arctan2",
      &mx::arctan2,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arctan2(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise inverse tangent of the ratio of two arrays.

        Args:
            a (array): Input array.
            b (array): Input array.

        Returns:
            array: The inverse tangent of the ratio of ``a`` and ``b``.
      )pbdoc");
  m.def(
      "sinh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sinh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sinh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise hyperbolic sine.

        Args:
            a (array): Input array.

        Returns:
            array: The hyperbolic sine of ``a``.
      )pbdoc");
  m.def(
      "cosh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::cosh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def cosh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise hyperbolic cosine.

        Args:
            a (array): Input array.

        Returns:
            array: The hyperbolic cosine of ``a``.
      )pbdoc");
  m.def(
      "tanh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::tanh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tanh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise hyperbolic tangent.

        Args:
            a (array): Input array.

        Returns:
            array: The hyperbolic tangent of ``a``.
      )pbdoc");
  m.def(
      "arcsinh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arcsinh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arcsinh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise inverse hyperbolic sine.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse hyperbolic sine of ``a``.
      )pbdoc");
  m.def(
      "arccosh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arccosh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arccosh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise inverse hyperbolic cosine.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse hyperbolic cosine of ``a``.
      )pbdoc");
  m.def(
      "arctanh",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::arctanh(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arctanh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise inverse hyperbolic tangent.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse hyperbolic tangent of ``a``.
      )pbdoc");
  m.def(
      "degrees",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::degrees(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def degrees(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
      Convert angles from radians to degrees.

      Args:
          a (array): Input array.

      Returns:
          array: The angles in degrees.
    )pbdoc");
  m.def(
      "radians",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::radians(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def radians(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
      Convert angles from degrees to radians.

      Args:
          a (array): Input array.

      Returns:
          array: The angles in radians.
    )pbdoc");
  m.def(
      "log",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::log(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def log(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise natural logarithm.

        Args:
            a (array): Input array.

        Returns:
            array: The natural logarithm of ``a``.
      )pbdoc");
  m.def(
      "log2",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::log2(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def log2(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise base-2 logarithm.

        Args:
            a (array): Input array.

        Returns:
            array: The base-2 logarithm of ``a``.
      )pbdoc");
  m.def(
      "log10",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::log10(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def log10(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise base-10 logarithm.

        Args:
            a (array): Input array.

        Returns:
            array: The base-10 logarithm of ``a``.
      )pbdoc");
  m.def(
      "log1p",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::log1p(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def log1p(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise natural log of one plus the array.

        Args:
            a (array): Input array.

        Returns:
            array: The natural logarithm of one plus ``a``.
      )pbdoc");
  m.def(
      "stop_gradient",
      &mx::stop_gradient,
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def stop_gradient(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Stop gradients from being computed.

        The operation is the identity but it prevents gradients from flowing
        through the array.

        Args:
            a (array): Input array.

        Returns:
            array:
              The unchanged input ``a`` but without gradient flowing
              through it.
      )pbdoc");
  m.def(
      "sigmoid",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::sigmoid(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sigmoid(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise logistic sigmoid.

        The logistic sigmoid function is:

        .. math::
          \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

        Args:
            a (array): Input array.

        Returns:
            array: The logistic sigmoid of ``a``.
      )pbdoc");
  m.def(
      "power",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::power(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def power(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise power operation.

        Raise the elements of a to the powers in elements of b with numpy-style
        broadcasting semantics. Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: Bases of ``a`` raised to powers in ``b``.
      )pbdoc");
  m.def(
      "arange",
      [](Scalar start,
         Scalar stop,
         const std::optional<Scalar>& step,
         const std::optional<mx::Dtype>& dtype_,
         mx::StreamOrDevice s) {
        // Determine the final dtype based on input types
        mx::Dtype dtype = dtype_
            ? *dtype_
            : mx::promote_types(
                  scalar_to_dtype(start),
                  step ? mx::promote_types(
                             scalar_to_dtype(stop), scalar_to_dtype(*step))
                       : scalar_to_dtype(stop));
        return mx::arange(
            scalar_to_double(start),
            scalar_to_double(stop),
            step ? scalar_to_double(*step) : 1.0,
            dtype,
            s);
      },
      "start"_a.noconvert(),
      "stop"_a.noconvert(),
      "step"_a.noconvert() = nb::none(),
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arange(start : Union[int, float], stop : Union[int, float], step : Union[None, int, float], dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
      Generates ranges of numbers.

      Generate numbers in the half-open interval ``[start, stop)`` in
      increments of ``step``.

      Args:
          start (float or int, optional): Starting value which defaults to ``0``.
          stop (float or int): Stopping value.
          step (float or int, optional): Increment which defaults to ``1``.
          dtype (Dtype, optional): Specifies the data type of the output. If unspecified will default to ``float32`` if any of ``start``, ``stop``, or ``step`` are ``float``. Otherwise will default to ``int32``.

      Returns:
          array: The range of values.

      Note:
        Following the Numpy convention the actual increment used to
        generate numbers is ``dtype(start + step) - dtype(start)``.
        This can lead to unexpected results for example if `start + step`
        is a fractional value and the `dtype` is integral.
      )pbdoc");
  m.def(
      "arange",
      [](Scalar stop,
         const std::optional<Scalar>& step,
         const std::optional<mx::Dtype>& dtype_,
         mx::StreamOrDevice s) {
        mx::Dtype dtype = dtype_ ? *dtype_
            : step
            ? mx::promote_types(scalar_to_dtype(stop), scalar_to_dtype(*step))
            : scalar_to_dtype(stop);
        return mx::arange(
            0.0,
            scalar_to_double(stop),
            step ? scalar_to_double(*step) : 1.0,
            dtype,
            s);
      },
      "stop"_a.noconvert(),
      "step"_a.noconvert() = nb::none(),
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arange(stop : Union[int, float], step : Union[None, int, float] = None, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def(
      "linspace",
      [](Scalar start,
         Scalar stop,
         int num,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        return mx::linspace(
            scalar_to_double(start),
            scalar_to_double(stop),
            num,
            dtype.value_or(mx::float32),
            s);
      },
      "start"_a,
      "stop"_a,
      "num"_a = 50,
      "dtype"_a.none() = mx::float32,
      "stream"_a = nb::none(),
      nb::sig(
          "def linspace(start, stop, num: Optional[int] = 50, dtype: Optional[Dtype] = float32, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Generate ``num`` evenly spaced numbers over interval ``[start, stop]``.

        Args:
            start (scalar): Starting value.
            stop (scalar): Stopping value.
            num (int, optional): Number of samples, defaults to ``50``.
            dtype (Dtype, optional): Specifies the data type of the output,
              default to ``float32``.

        Returns:
            array: The range of values.
      )pbdoc");
  m.def(
      "take",
      [](const mx::array& a,
         const std::variant<nb::int_, mx::array>& indices,
         const std::optional<int>& axis,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<nb::int_>(&indices); pv) {
          auto idx = nb::cast<int>(*pv);
          return axis ? mx::take(a, idx, axis.value(), s) : mx::take(a, idx, s);
        } else {
          auto indices_ = std::get<mx::array>(indices);
          return axis ? mx::take(a, indices_, axis.value(), s)
                      : mx::take(a, indices_, s);
        }
      },
      nb::arg(),
      "indices"_a,
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def take(a: array, /, indices: Union[int, array], axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Take elements along an axis.

        The elements are taken from ``indices`` along the specified axis.
        If the axis is not specified the array is treated as a flattened
        1-D array prior to performing the take.

        As an example, if the ``axis=1`` this is equivalent to ``a[:, indices, ...]``.

        Args:
            a (array): Input array.
            indices (int or array): Integer index or input array with integral type.
            axis (int, optional): Axis along which to perform the take. If unspecified
              the array is treated as a flattened 1-D vector.

        Returns:
            array: The indexed values of ``a``.
      )pbdoc");
  m.def(
      "take_along_axis",
      [](const mx::array& a,
         const mx::array& indices,
         const std::optional<int>& axis,
         mx::StreamOrDevice s) {
        if (axis.has_value()) {
          return mx::take_along_axis(a, indices, axis.value(), s);
        } else {
          return mx::take_along_axis(mx::reshape(a, {-1}, s), indices, 0, s);
        }
      },
      nb::arg(),
      "indices"_a,
      "axis"_a.none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def take_along_axis(a: array, /, indices: array, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Take values along an axis at the specified indices.

        Args:
            a (array): Input array.
            indices (array): Indices array. These should be broadcastable with
              the input array excluding the `axis` dimension.
            axis (int or None): Axis in the input to take the values from. If
              ``axis == None`` the array is flattened to 1D prior to the indexing
              operation.

        Returns:
            array: The output array.
      )pbdoc");
  m.def(
      "put_along_axis",
      [](const mx::array& a,
         const mx::array& indices,
         const mx::array& values,
         const std::optional<int>& axis,
         mx::StreamOrDevice s) {
        if (axis.has_value()) {
          return mx::put_along_axis(a, indices, values, axis.value(), s);
        } else {
          return mx::reshape(
              mx::put_along_axis(
                  mx::reshape(a, {-1}, s), indices, values, 0, s),
              a.shape(),
              s);
        }
      },
      nb::arg(),
      "indices"_a,
      "values"_a,
      "axis"_a.none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def put_along_axis(a: array, /, indices: array, values: array, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Put values along an axis at the specified indices.

        Args:
            a (array): Destination array.
            indices (array): Indices array. These should be broadcastable with
              the input array excluding the `axis` dimension.
            values (array): Values array. These should be broadcastable with
              the indices.

            axis (int or None): Axis in the destination to put the values to. If
              ``axis == None`` the destination is flattened prior to the put
              operation.

        Returns:
            array: The output array.
      )pbdoc");
  m.def(
      "full",
      [](const std::variant<int, mx::Shape>& shape,
         const ScalarOrArray& vals,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&shape); pv) {
          return mx::full({*pv}, to_array(vals, dtype), s);
        } else {
          return mx::full(std::get<mx::Shape>(shape), to_array(vals, dtype), s);
        }
      },
      "shape"_a,
      "vals"_a,
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def full(shape: Union[int, Sequence[int]], vals: Union[scalar, array], dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Construct an array with the given value.

        Constructs an array of size ``shape`` filled with ``vals``. If ``vals``
        is an :obj:`array` it must be broadcastable to the given ``shape``.

        Args:
            shape (int or list(int)): The shape of the output array.
            vals (float or int or array): Values to fill the array with.
            dtype (Dtype, optional): Data type of the output array. If
              unspecified the output type is inferred from ``vals``.

        Returns:
            array: The output array with the specified shape and values.
      )pbdoc");
  m.def(
      "zeros",
      [](const std::variant<int, mx::Shape>& shape,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        auto t = dtype.value_or(mx::float32);
        if (auto pv = std::get_if<int>(&shape); pv) {
          return mx::zeros({*pv}, t, s);
        } else {
          return mx::zeros(std::get<mx::Shape>(shape), t, s);
        }
      },
      "shape"_a,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def zeros(shape: Union[int, Sequence[int]], dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Construct an array of zeros.

        Args:
            shape (int or list(int)): The shape of the output array.
            dtype (Dtype, optional): Data type of the output array. If
              unspecified the output type defaults to ``float32``.

        Returns:
            array: The array of zeros with the specified shape.
      )pbdoc");
  m.def(
      "zeros_like",
      &mx::zeros_like,
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def zeros_like(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        An array of zeros like the input.

        Args:
            a (array): The input to take the shape and type from.

        Returns:
            array: The output array filled with zeros.
      )pbdoc");
  m.def(
      "ones",
      [](const std::variant<int, mx::Shape>& shape,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        auto t = dtype.value_or(mx::float32);
        if (auto pv = std::get_if<int>(&shape); pv) {
          return mx::ones({*pv}, t, s);
        } else {
          return mx::ones(std::get<mx::Shape>(shape), t, s);
        }
      },
      "shape"_a,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def ones(shape: Union[int, Sequence[int]], dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Construct an array of ones.

        Args:
            shape (int or list(int)): The shape of the output array.
            dtype (Dtype, optional): Data type of the output array. If
              unspecified the output type defaults to ``float32``.

        Returns:
            array: The array of ones with the specified shape.
      )pbdoc");
  m.def(
      "ones_like",
      &mx::ones_like,
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def ones_like(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        An array of ones like the input.

        Args:
            a (array): The input to take the shape and type from.

        Returns:
            array: The output array filled with ones.
      )pbdoc");
  m.def(
      "eye",
      [](int n,
         std::optional<int> m,
         int k,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        return mx::eye(n, m.value_or(n), k, dtype.value_or(mx::float32), s);
      },
      "n"_a,
      "m"_a = nb::none(),
      "k"_a = 0,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def eye(n: int, m: Optional[int] = None, k: int = 0, dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Create an identity matrix or a general diagonal matrix.

        Args:
            n (int): The number of rows in the output.
            m (int, optional): The number of columns in the output. Defaults to n.
            k (int, optional): Index of the diagonal. Defaults to 0 (main diagonal).
            dtype (Dtype, optional): Data type of the output array. Defaults to float32.
            stream (Stream, optional): Stream or device. Defaults to None.

        Returns:
            array: An array where all elements are equal to zero, except for the k-th diagonal, whose values are equal to one.
      )pbdoc");
  m.def(
      "identity",
      [](int n, std::optional<mx::Dtype> dtype, mx::StreamOrDevice s) {
        return mx::identity(n, dtype.value_or(mx::float32), s);
      },
      "n"_a,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def identity(n: int, dtype: Optional[Dtype] = float32, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Create a square identity matrix.

        Args:
            n (int): The number of rows and columns in the output.
            dtype (Dtype, optional): Data type of the output array. Defaults to float32.
            stream (Stream, optional): Stream or device. Defaults to None.

        Returns:
            array: An identity matrix of size n x n.
      )pbdoc");
  m.def(
      "tri",
      [](int n,
         std::optional<int> m,
         int k,
         std::optional<mx::Dtype> type,
         mx::StreamOrDevice s) {
        return mx::tri(n, m.value_or(n), k, type.value_or(mx::float32), s);
      },
      "n"_a,
      "m"_a = nb::none(),
      "k"_a = 0,
      "dtype"_a.none() = mx::float32,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tri(n: int, m: int, k: int, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        An array with ones at and below the given diagonal and zeros elsewhere.

        Args:
          n (int): The number of rows in the output.
          m (int, optional): The number of cols in the output. Defaults to ``None``.
          k (int, optional): The diagonal of the 2-D array. Defaults to ``0``.
          dtype (Dtype, optional): Data type of the output array. Defaults to ``float32``.
          stream (Stream, optional): Stream or device. Defaults to ``None``.

        Returns:
          array: Array with its lower triangle filled with ones and zeros elsewhere
      )pbdoc");
  m.def(
      "tril",
      &mx::tril,
      "x"_a,
      "k"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tril(x: array, k: int, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Zeros the array above the given diagonal.

        Args:
          x (array): input array.
          k (int, optional): The diagonal of the 2-D array. Defaults to ``0``.
          stream (Stream, optional): Stream or device. Defaults to ``None``.

        Returns:
          array: Array zeroed above the given diagonal
      )pbdoc");
  m.def(
      "triu",
      &mx::triu,
      "x"_a,
      "k"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def triu(x: array, k: int, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Zeros the array below the given diagonal.

        Args:
          x (array): input array.
          k (int, optional): The diagonal of the 2-D array. Defaults to ``0``.
          stream (Stream, optional): Stream or device. Defaults to ``None``.

        Returns:
          array: Array zeroed below the given diagonal
    )pbdoc");
  m.def(
      "allclose",
      &mx::allclose,
      nb::arg(),
      nb::arg(),
      "rtol"_a = 1e-5,
      "atol"_a = 1e-8,
      nb::kw_only(),
      "equal_nan"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def allclose(a: array, b: array, /, rtol: float = 1e-05, atol: float = 1e-08, *, equal_nan: bool = False, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Approximate comparison of two arrays.

        Infinite values are considered equal if they have the same sign, NaN values are not equal unless ``equal_nan`` is ``True``.

        The arrays are considered equal if:

        .. code-block::

         all(abs(a - b) <= (atol + rtol * abs(b)))

        Note unlike :func:`array_equal`, this function supports numpy-style
        broadcasting.

        Args:
            a (array): Input array.
            b (array): Input array.
            rtol (float): Relative tolerance.
            atol (float): Absolute tolerance.
            equal_nan (bool): If ``True``, NaNs are considered equal.
              Defaults to ``False``.

        Returns:
            array: The boolean output scalar indicating if the arrays are close.
      )pbdoc");
  m.def(
      "isclose",
      &mx::isclose,
      nb::arg(),
      nb::arg(),
      "rtol"_a = 1e-5,
      "atol"_a = 1e-8,
      nb::kw_only(),
      "equal_nan"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def isclose(a: array, b: array, /, rtol: float = 1e-05, atol: float = 1e-08, *, equal_nan: bool = False, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Returns a boolean array where two arrays are element-wise equal within a tolerance.

        Infinite values are considered equal if they have the same sign, NaN values are
        not equal unless ``equal_nan`` is ``True``.

        Two values are considered equal if:

        .. code-block::

         abs(a - b) <= (atol + rtol * abs(b))

        Note unlike :func:`array_equal`, this function supports numpy-style
        broadcasting.

        Args:
            a (array): Input array.
            b (array): Input array.
            rtol (float): Relative tolerance.
            atol (float): Absolute tolerance.
            equal_nan (bool): If ``True``, NaNs are considered equal.
              Defaults to ``False``.

        Returns:
            array: The boolean output scalar indicating if the arrays are close.
      )pbdoc");
  m.def(
      "all",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::all(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def all(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        An `and` reduction over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the corresponding axes reduced.
      )pbdoc");
  m.def(
      "any",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::any(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def any(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        An `or` reduction over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the corresponding axes reduced.
      )pbdoc");
  m.def(
      "minimum",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::minimum(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def minimum(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise minimum.

        Take the element-wise min of two arrays with numpy-style broadcasting
        semantics. Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The min of ``a`` and ``b``.
      )pbdoc");
  m.def(
      "maximum",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::maximum(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def maximum(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise maximum.

        Take the element-wise max of two arrays with numpy-style broadcasting
        semantics. Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The max of ``a`` and ``b``.
      )pbdoc");
  m.def(
      "floor",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::floor(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def floor(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise floor.

        Args:
            a (array): Input array.

        Returns:
            array: The floor of ``a``.
      )pbdoc");
  m.def(
      "ceil",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::ceil(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def ceil(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise ceil.

        Args:
            a (array): Input array.

        Returns:
            array: The ceil of ``a``.
      )pbdoc");
  m.def(
      "isnan",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isnan(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def isnan(a: array, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Return a boolean array indicating which elements are NaN.

        Args:
            a (array): Input array.

        Returns:
            array: The boolean array indicating which elements are NaN.
      )pbdoc");
  m.def(
      "isinf",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isinf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def isinf(a: array, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Return a boolean array indicating which elements are +/- inifnity.

        Args:
            a (array): Input array.

        Returns:
            array: The boolean array indicating which elements are +/- infinity.
      )pbdoc");
  m.def(
      "isfinite",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isfinite(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def isfinite(a: array, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Return a boolean array indicating which elements are finite.

        An element is finite if it is not infinite or NaN.

        Args:
            a (array): Input array.

        Returns:
            array: The boolean array indicating which elements are finite.
      )pbdoc");
  m.def(
      "isposinf",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isposinf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def isposinf(a: array, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Return a boolean array indicating which elements are positive infinity.

        Args:
            a (array): Input array.
            stream (Union[None, Stream, Device]): Optional stream or device.

        Returns:
            array: The boolean array indicating which elements are positive infinity.
      )pbdoc");
  m.def(
      "isneginf",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::isneginf(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def isneginf(a: array, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Return a boolean array indicating which elements are negative infinity.

        Args:
            a (array): Input array.
            stream (Union[None, Stream, Device]): Optional stream or device.

        Returns:
            array: The boolean array indicating which elements are negative infinity.
      )pbdoc");
  m.def(
      "moveaxis",
      &mx::moveaxis,
      nb::arg(),
      "source"_a,
      "destination"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def moveaxis(a: array, /, source: int, destination: int, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Move an axis to a new position.

        Args:
            a (array): Input array.
            source (int): Specifies the source axis.
            destination (int): Specifies the destination axis.

        Returns:
            array: The array with the axis moved.
      )pbdoc");
  m.def(
      "swapaxes",
      &mx::swapaxes,
      nb::arg(),
      "axis1"_a,
      "axis2"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def swapaxes(a: array, /, axis1 : int, axis2: int, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Swap two axes of an array.

        Args:
            a (array): Input array.
            axis1 (int): Specifies the first axis.
            axis2 (int): Specifies the second axis.

        Returns:
            array: The array with swapped axes.
      )pbdoc");
  m.def(
      "transpose",
      [](const mx::array& a,
         const std::optional<std::vector<int>>& axes,
         mx::StreamOrDevice s) {
        if (axes.has_value()) {
          return mx::transpose(a, *axes, s);
        } else {
          return mx::transpose(a, s);
        }
      },
      nb::arg(),
      "axes"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def transpose(a: array, /, axes: Optional[Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Transpose the dimensions of the array.

        Args:
            a (array): Input array.
            axes (list(int), optional): Specifies the source axis for each axis
              in the new array. The default is to reverse the axes.

        Returns:
            array: The transposed array.
      )pbdoc");
  m.def(
      "permute_dims",
      [](const mx::array& a,
         const std::optional<std::vector<int>>& axes,
         mx::StreamOrDevice s) {
        if (axes.has_value()) {
          return mx::transpose(a, *axes, s);
        } else {
          return mx::transpose(a, s);
        }
      },
      nb::arg(),
      "axes"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def permute_dims(a: array, /, axes: Optional[Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        See :func:`transpose`.
      )pbdoc");
  m.def(
      "sum",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::sum(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      "array"_a,
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sum(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Sum reduce the array over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the corresponding axes reduced.
      )pbdoc");
  m.def(
      "prod",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::prod(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def prod(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        An product reduction over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the corresponding axes reduced.
      )pbdoc");
  m.def(
      "min",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::min(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def min(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        A `min` reduction over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the corresponding axes reduced.
      )pbdoc");
  m.def(
      "max",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::max(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def max(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        A `max` reduction over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the corresponding axes reduced.
      )pbdoc");
  m.def(
      "logsumexp",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::logsumexp(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def logsumexp(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        A `log-sum-exp` reduction over the given axes.

        The log-sum-exp reduction is a numerically stable version of:

        .. code-block::

          log(sum(exp(a), axis))

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the corresponding axes reduced.
      )pbdoc");
  m.def(
      "mean",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        return mx::mean(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def mean(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Compute the mean(s) over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array of means.
      )pbdoc");
  m.def(
      "var",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         int ddof,
         mx::StreamOrDevice s) {
        return mx::var(a, get_reduce_axes(axis, a.ndim()), keepdims, ddof, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      "ddof"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def var(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, ddof: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Compute the variance(s) over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.
            ddof (int, optional): The divisor to compute the variance
              is ``N - ddof``, defaults to 0.

        Returns:
            array: The output array of variances.
      )pbdoc");
  m.def(
      "std",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool keepdims,
         int ddof,
         mx::StreamOrDevice s) {
        return mx::std(a, get_reduce_axes(axis, a.ndim()), keepdims, ddof, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      "ddof"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def std(a: array, /, axis: Union[None, int, Sequence[int]] = None, keepdims: bool = False, ddof: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Compute the standard deviation(s) over the given axes.

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or
              axes to reduce over. If unspecified this defaults
              to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.
            ddof (int, optional): The divisor to compute the variance
              is ``N - ddof``, defaults to 0.

        Returns:
            array: The output array of standard deviations.
      )pbdoc");
  m.def(
      "split",
      [](const mx::array& a,
         const std::variant<int, mx::Shape>& indices_or_sections,
         int axis,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&indices_or_sections); pv) {
          return mx::split(a, *pv, axis, s);
        } else {
          return mx::split(
              a, std::get<mx::Shape>(indices_or_sections), axis, s);
        }
      },
      nb::arg(),
      "indices_or_sections"_a,
      "axis"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def split(a: array, /, indices_or_sections: Union[int, Sequence[int]], axis: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Split an array along a given axis.

        Args:
            a (array): Input array.
            indices_or_sections (int or list(int)): If ``indices_or_sections``
              is an integer the array is split into that many sections of equal
              size. An error is raised if this is not possible. If ``indices_or_sections``
              is a list, the list contains the indices of the start of each subarray
              along the given axis.
            axis (int, optional): Axis to split along, defaults to `0`.

        Returns:
            list(array): A list of split arrays.
      )pbdoc");
  m.def(
      "argmin",
      [](const mx::array& a,
         std::optional<int> axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::argmin(a, *axis, keepdims, s);
        } else {
          return mx::argmin(a, keepdims, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def argmin(a: array, /, axis: Union[None, int] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Indices of the minimum values along the axis.

        Args:
            a (array): Input array.
            axis (int, optional): Optional axis to reduce over. If unspecified
              this defaults to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The ``uint32`` array with the indices of the minimum values.
      )pbdoc");
  m.def(
      "argmax",
      [](const mx::array& a,
         std::optional<int> axis,
         bool keepdims,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::argmax(a, *axis, keepdims, s);
        } else {
          return mx::argmax(a, keepdims, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      "keepdims"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def argmax(a: array, /, axis: Union[None, int] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Indices of the maximum values along the axis.

        Args:
            a (array): Input array.
            axis (int, optional): Optional axis to reduce over. If unspecified
              this defaults to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The ``uint32`` array with the indices of the maximum values.
      )pbdoc");
  m.def(
      "sort",
      [](const mx::array& a, std::optional<int> axis, mx::StreamOrDevice s) {
        if (axis) {
          return mx::sort(a, *axis, s);
        } else {
          return mx::sort(a, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def sort(a: array, /, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Returns a sorted copy of the array.

        Args:
            a (array): Input array.
            axis (int or None, optional): Optional axis to sort over.
              If ``None``, this sorts over the flattened array.
              If unspecified, it defaults to -1 (sorting over the last axis).

        Returns:
            array: The sorted array.
      )pbdoc");
  m.def(
      "argsort",
      [](const mx::array& a, std::optional<int> axis, mx::StreamOrDevice s) {
        if (axis) {
          return mx::argsort(a, *axis, s);
        } else {
          return mx::argsort(a, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def argsort(a: array, /, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Returns the indices that sort the array.

        Args:
            a (array): Input array.
            axis (int or None, optional): Optional axis to sort over.
              If ``None``, this sorts over the flattened array.
              If unspecified, it defaults to -1 (sorting over the last axis).

        Returns:
            array: The ``uint32`` array containing indices that sort the input.
      )pbdoc");
  m.def(
      "partition",
      [](const mx::array& a,
         int kth,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::partition(a, kth, *axis, s);
        } else {
          return mx::partition(a, kth, s);
        }
      },
      nb::arg(),
      "kth"_a,
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def partition(a: array, /, kth: int, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Returns a partitioned copy of the array such that the smaller ``kth``
        elements are first.

        The ordering of the elements in partitions is undefined.

        Args:
            a (array): Input array.
            kth (int): Element at the ``kth`` index will be in its sorted
              position in the output. All elements before the kth index will
              be less or equal to the ``kth`` element and all elements after
              will be greater or equal to the ``kth`` element in the output.
            axis (int or None, optional): Optional axis to partition over.
              If ``None``, this partitions over the flattened array.
              If unspecified, it defaults to ``-1``.

        Returns:
            array: The partitioned array.
      )pbdoc");
  m.def(
      "argpartition",
      [](const mx::array& a,
         int kth,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::argpartition(a, kth, *axis, s);
        } else {
          return mx::argpartition(a, kth, s);
        }
      },
      nb::arg(),
      "kth"_a,
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def argpartition(a: array, /, kth: int, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Returns the indices that partition the array.

        The ordering of the elements within a partition in given by the indices
        is undefined.

        Args:
            a (array): Input array.
            kth (int): Element index at the ``kth`` position in the output will
              give the sorted position. All indices before the ``kth`` position
              will be of elements less or equal to the element at the ``kth``
              index and all indices after will be of elements greater or equal
              to the element at the ``kth`` index.
            axis (int or None, optional): Optional axis to partition over.
              If ``None``, this partitions over the flattened array.
              If unspecified, it defaults to ``-1``.

        Returns:
            array: The ``uint32`` array containing indices that partition the input.
      )pbdoc");
  m.def(
      "topk",
      [](const mx::array& a,
         int k,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::topk(a, k, *axis, s);
        } else {
          return mx::topk(a, k, s);
        }
      },
      nb::arg(),
      "k"_a,
      "axis"_a.none() = -1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def topk(a: array, /, k: int, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Returns the ``k`` largest elements from the input along a given axis.

        The elements will not necessarily be in sorted order.

        Args:
            a (array): Input array.
            k (int): ``k`` top elements to be returned
            axis (int or None, optional): Optional axis to select over.
              If ``None``, this selects the top ``k`` elements over the
              flattened array. If unspecified, it defaults to ``-1``.

        Returns:
            array: The top ``k`` elements from the input.
      )pbdoc");
  m.def(
      "broadcast_to",
      [](const ScalarOrArray& a, const mx::Shape& shape, mx::StreamOrDevice s) {
        return mx::broadcast_to(to_array(a), shape, s);
      },
      nb::arg(),
      "shape"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def broadcast_to(a: Union[scalar, array], /, shape: Sequence[int], *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Broadcast an array to the given shape.

        The broadcasting semantics are the same as Numpy.

        Args:
            a (array): Input array.
            shape (list(int)): The shape to broadcast to.

        Returns:
            array: The output array with the new shape.
      )pbdoc");
  m.def(
      "broadcast_arrays",
      [](const nb::args& args, mx::StreamOrDevice s) {
        return broadcast_arrays(nb::cast<std::vector<mx::array>>(args), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def broadcast_arrays(*arrays: array, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, ...]"),
      R"pbdoc(
        Broadcast arrays against one another.

        The broadcasting semantics are the same as Numpy.

        Args:
            *arrays (array): The input arrays.

        Returns:
            tuple(array): The output arrays with the broadcasted shape.
      )pbdoc");
  m.def(
      "softmax",
      [](const mx::array& a,
         const IntOrVec& axis,
         bool precise,
         mx::StreamOrDevice s) {
        return mx::softmax(a, get_reduce_axes(axis, a.ndim()), precise, s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "precise"_a = false,
      "stream"_a = nb::none(),
      nb::sig(
          "def softmax(a: array, /, axis: Union[None, int, Sequence[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Perform the softmax along the given axis.

        This operation is a numerically stable version of:

        .. code-block::

          exp(a) / sum(exp(a), axis, keepdims=True)

        Args:
            a (array): Input array.
            axis (int or list(int), optional): Optional axis or axes to compute
             the softmax over. If unspecified this performs the softmax over
             the full array.

        Returns:
            array: The output of the softmax.
      )pbdoc");
  m.def(
      "concatenate",
      [](const std::vector<mx::array>& arrays,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::concatenate(arrays, *axis, s);
        } else {
          return mx::concatenate(arrays, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def concatenate(arrays: list[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Concatenate the arrays along the given axis.

        Args:
            arrays (list(array)): Input :obj:`list` or :obj:`tuple` of arrays.
            axis (int, optional): Optional axis to concatenate along. If
              unspecified defaults to ``0``.

        Returns:
            array: The concatenated array.
      )pbdoc");
  m.def(
      "concat",
      [](const std::vector<mx::array>& arrays,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::concatenate(arrays, *axis, s);
        } else {
          return mx::concatenate(arrays, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def concat(arrays: list[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        See :func:`concatenate`.
      )pbdoc");
  m.def(
      "stack",
      [](const std::vector<mx::array>& arrays,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis.has_value()) {
          return mx::stack(arrays, axis.value(), s);
        } else {
          return mx::stack(arrays, s);
        }
      },
      nb::arg(),
      "axis"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def stack(arrays: list[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Stacks the arrays along a new axis.

        Args:
            arrays (list(array)): A list of arrays to stack.
            axis (int, optional): The axis in the result array along which the
              input arrays are stacked. Defaults to ``0``.
            stream (Stream, optional): Stream or device. Defaults to ``None``.

        Returns:
            array: The resulting stacked array.
      )pbdoc");
  m.def(
      "meshgrid",
      [](nb::args arrays_,
         bool sparse,
         std::string indexing,
         mx::StreamOrDevice s) {
        std::vector<mx::array> arrays =
            nb::cast<std::vector<mx::array>>(arrays_);
        return mx::meshgrid(arrays, sparse, indexing, s);
      },
      "arrays"_a,
      "sparse"_a = false,
      "indexing"_a = "xy",
      "stream"_a = nb::none(),
      nb::sig(
          "def meshgrid(*arrays: array, sparse: Optional[bool] = False, indexing: Optional[str] = 'xy', stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Generate multidimensional coordinate grids from 1-D coordinate arrays

        Args:
            *arrays (array): Input arrays.
            sparse (bool, optional): If ``True``, a sparse grid is returned in which each output
              array has a single non-zero element. If ``False``, a dense grid is returned.
              Defaults to ``False``.
            indexing (str, optional): Cartesian ('xy') or matrix ('ij') indexing of the output arrays.
              Defaults to ``'xy'``.

        Returns:
            list(array): The output arrays.
      )pbdoc");
  m.def(
      "repeat",
      [](const mx::array& array,
         int repeats,
         std::optional<int> axis,
         mx::StreamOrDevice s) {
        if (axis.has_value()) {
          return mx::repeat(array, repeats, axis.value(), s);
        } else {
          return mx::repeat(array, repeats, s);
        }
      },
      nb::arg(),
      "repeats"_a,
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def repeat(array: array, repeats: int, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Repeat an array along a specified axis.

        Args:
            array (array): Input array.
            repeats (int): The number of repetitions for each element.
            axis (int, optional): The axis in which to repeat the array along. If
              unspecified it uses the flattened array of the input and repeats
              along axis 0.
            stream (Stream, optional): Stream or device. Defaults to ``None``.

        Returns:
            array: The resulting repeated array.
      )pbdoc");
  m.def(
      "clip",
      [](const mx::array& a,
         const std::optional<ScalarOrArray>& min,
         const std::optional<ScalarOrArray>& max,
         mx::StreamOrDevice s) {
        std::optional<mx::array> min_ = std::nullopt;
        std::optional<mx::array> max_ = std::nullopt;
        if (min) {
          min_ = to_arrays(a, min.value()).second;
        }
        if (max) {
          max_ = to_arrays(a, max.value()).second;
        }
        return mx::clip(a, min_, max_, s);
      },
      nb::arg(),
      "a_min"_a.none(),
      "a_max"_a.none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def clip(a: array, /, a_min: Union[scalar, array, None], a_max: Union[scalar, array, None], *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Clip the values of the array between the given minimum and maximum.

        If either ``a_min`` or ``a_max`` are ``None``, then corresponding edge
        is ignored. At least one of ``a_min`` and ``a_max`` cannot be ``None``.
        The input ``a`` and the limits must broadcast with one another.

        Args:
            a (array): Input array.
            a_min (scalar or array or None): Minimum value to clip to.
            a_max (scalar or array or None): Maximum value to clip to.

        Returns:
            array: The clipped array.
      )pbdoc");
  m.def(
      "pad",
      [](const mx::array& a,
         const std::variant<
             int,
             std::tuple<int>,
             std::pair<int, int>,
             std::vector<std::pair<int, int>>>& pad_width,
         const std::string mode,
         const ScalarOrArray& constant_value,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&pad_width); pv) {
          return mx::pad(a, *pv, to_array(constant_value), mode, s);
        } else if (auto pv = std::get_if<std::tuple<int>>(&pad_width); pv) {
          return mx::pad(
              a, std::get<0>(*pv), to_array(constant_value), mode, s);
        } else if (auto pv = std::get_if<std::pair<int, int>>(&pad_width); pv) {
          return mx::pad(a, *pv, to_array(constant_value), mode, s);
        } else {
          auto v = std::get<std::vector<std::pair<int, int>>>(pad_width);
          if (v.size() == 1) {
            return mx::pad(a, v[0], to_array(constant_value), mode, s);
          } else {
            return mx::pad(a, v, to_array(constant_value), mode, s);
          }
        }
      },
      nb::arg(),
      "pad_width"_a,
      "mode"_a = "constant",
      "constant_values"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def pad(a: array, pad_width: Union[int, tuple[int], tuple[int, int], list[tuple[int, int]]], mode: Literal['constant', 'edge'] = 'constant', constant_values: Union[scalar, array] = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Pad an array with a constant value

        Args:
            a (array): Input array.
            pad_width (int, tuple(int), tuple(int, int) or list(tuple(int, int))): Number of padded
              values to add to the edges of each axis:``((before_1, after_1),
              (before_2, after_2), ..., (before_N, after_N))``. If a single pair
              of integers is passed then ``(before_i, after_i)`` are all the same.
              If a single integer or tuple with a single integer is passed then
              all axes are extended by the same number on each side.
            mode: Padding mode. One of the following strings:
              "constant" (default): Pads with a constant value.
              "edge": Pads with the edge values of array.
            constant_value (array or scalar, optional): Optional constant value
              to pad the edges of the array with.

        Returns:
            array: The padded array.
      )pbdoc");
  m.def(
      "as_strided",
      [](const mx::array& a,
         std::optional<mx::Shape> shape,
         std::optional<mx::Strides> strides,
         size_t offset,
         mx::StreamOrDevice s) {
        auto a_shape = (shape) ? *shape : a.shape();
        mx::Strides a_strides;
        if (strides) {
          a_strides = *strides;
        } else {
          a_strides = mx::Strides(a_shape.size(), 1);
          for (int i = a_shape.size() - 1; i > 0; i--) {
            a_strides[i - 1] = a_shape[i] * a_strides[i];
          }
        }
        return mx::as_strided(a, a_shape, a_strides, offset, s);
      },
      nb::arg(),
      "shape"_a = nb::none(),
      "strides"_a = nb::none(),
      "offset"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def as_strided(a: array, /, shape: Optional[Sequence[int]] = None, strides: Optional[Sequence[int]] = None, offset: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Create a view into the array with the given shape and strides.

        The resulting array will always be as if the provided array was row
        contiguous regardless of the provided arrays storage order and current
        strides.

        .. note::
           Note that this function should be used with caution as it changes
           the shape and strides of the array directly. This can lead to the
           resulting array pointing to invalid memory locations which can
           result into crashes.

        Args:
          a (array): Input array
          shape (list(int), optional): The shape of the resulting array. If
            None it defaults to ``a.shape()``.
          strides (list(int), optional): The strides of the resulting array. If
            None it defaults to the reverse exclusive cumulative product of
            ``a.shape()``.
          offset (int): Skip that many elements from the beginning of the input
            array.

        Returns:
          array: The output array which is the strided view of the input.
      )pbdoc");
  m.def(
      "cumsum",
      [](const mx::array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::cumsum(a, *axis, reverse, inclusive, s);
        } else {
          return mx::cumsum(mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig(
          "def cumsum(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Return the cumulative sum of the elements along the given axis.

        Args:
          a (array): Input array
          axis (int, optional): Optional axis to compute the cumulative sum
            over. If unspecified the cumulative sum of the flattened array is
            returned.
          reverse (bool): Perform the cumulative sum in reverse.
          inclusive (bool): The i-th element of the output includes the i-th
            element of the input.

        Returns:
          array: The output array.
      )pbdoc");
  m.def(
      "cumprod",
      [](const mx::array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::cumprod(a, *axis, reverse, inclusive, s);
        } else {
          return mx::cumprod(mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig(
          "def cumprod(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Return the cumulative product of the elements along the given axis.

        Args:
          a (array): Input array
          axis (int, optional): Optional axis to compute the cumulative product
            over. If unspecified the cumulative product of the flattened array is
            returned.
          reverse (bool): Perform the cumulative product in reverse.
          inclusive (bool): The i-th element of the output includes the i-th
            element of the input.

        Returns:
          array: The output array.
      )pbdoc");
  m.def(
      "cummax",
      [](const mx::array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::cummax(a, *axis, reverse, inclusive, s);
        } else {
          return mx::cummax(mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig(
          "def cummax(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Return the cumulative maximum of the elements along the given axis.

        Args:
          a (array): Input array
          axis (int, optional): Optional axis to compute the cumulative maximum
            over. If unspecified the cumulative maximum of the flattened array is
            returned.
          reverse (bool): Perform the cumulative maximum in reverse.
          inclusive (bool): The i-th element of the output includes the i-th
            element of the input.

        Returns:
          array: The output array.
      )pbdoc");
  m.def(
      "cummin",
      [](const mx::array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         mx::StreamOrDevice s) {
        if (axis) {
          return mx::cummin(a, *axis, reverse, inclusive, s);
        } else {
          return mx::cummin(mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
        }
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = nb::none(),
      nb::sig(
          "def cummin(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Return the cumulative minimum of the elements along the given axis.

        Args:
          a (array): Input array
          axis (int, optional): Optional axis to compute the cumulative minimum
            over. If unspecified the cumulative minimum of the flattened array is
            returned.
          reverse (bool): Perform the cumulative minimum in reverse.
          inclusive (bool): The i-th element of the output includes the i-th
            element of the input.

        Returns:
          array: The output array.
      )pbdoc");
  m.def(
      "conj",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::conjugate(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conj(a: array, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Return the elementwise complex conjugate of the input.
        Alias for `mx.conjugate`.

        Args:
          a (array): Input array

        Returns:
          array: The output array.
      )pbdoc");
  m.def(
      "conjugate",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::conjugate(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conjugate(a: array, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Return the elementwise complex conjugate of the input.
        Alias for `mx.conj`.

        Args:
          a (array): Input array

        Returns:
          array: The output array.
      )pbdoc");
  m.def(
      "convolve",
      [](const mx::array& a,
         const mx::array& v,
         const std::string& mode,
         mx::StreamOrDevice s) {
        if (a.ndim() != 1 || v.ndim() != 1) {
          throw std::invalid_argument("[convolve] Inputs must be 1D.");
        }

        if (a.size() == 0 || v.size() == 0) {
          throw std::invalid_argument("[convolve] Inputs cannot be empty.");
        }

        mx::array in = a.size() < v.size() ? v : a;
        mx::array wt = a.size() < v.size() ? a : v;
        wt = mx::slice(wt, {wt.shape(0) - 1}, {-wt.shape(0) - 1}, {-1}, s);

        in = mx::reshape(in, {1, -1, 1}, s);
        wt = mx::reshape(wt, {1, -1, 1}, s);

        int padding = 0;

        if (mode == "full") {
          padding = wt.size() - 1;
        } else if (mode == "valid") {
          padding = 0;
        } else if (mode == "same") {
          // Odd sizes use symmetric padding
          if (wt.size() % 2) {
            padding = wt.size() / 2;
          } else { // Even sizes use asymmetric padding
            int pad_l = wt.size() / 2;
            int pad_r = std::max(0, pad_l - 1);
            in = mx::pad(
                in,
                {{0, 0}, {pad_l, pad_r}, {0, 0}},
                mx::array(0),
                "constant",
                s);
          }

        } else {
          throw std::invalid_argument("[convolve] Invalid mode.");
        }

        mx::array out = mx::conv1d(
            in,
            wt,
            /*stride = */ 1,
            /*padding = */ padding,
            /*dilation = */ 1,
            /*groups = */ 1,
            s);

        return mx::reshape(out, {-1}, s);
      },
      nb::arg(),
      nb::arg(),
      "mode"_a = "full",
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          R"(def convolve(a: array, v: array, /, mode: str = "full", *, stream: Union[None, Stream, Device] = None) -> array)"),
      R"pbdoc(
        The discrete convolution of 1D arrays.

        If ``v`` is longer than ``a``, then they are swapped.
        The conv filter is flipped following signal processing convention.

        Args:
            a (array): 1D Input array.
            v (array): 1D Input array.
            mode (str, optional): {'full', 'valid', 'same'}

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv1d",
      &mx::conv1d,
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv1d(input: array, weight: array, /, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        1D convolution over an input with several channels

        Args:
            input (array): Input array of shape ``(N, H, C_in)``.
            weight (array): Weight array of shape ``(C_out, H, C_in)``.
            stride (int, optional): Kernel stride. Default: ``1``.
            padding (int, optional): Input padding. Default: ``0``.
            dilation (int, optional): Kernel dilation. Default: ``1``.
            groups (int, optional): Input feature groups. Default: ``1``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv2d",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::pair<int, int>>& stride,
         const std::variant<int, std::pair<int, int>>& padding,
         const std::variant<int, std::pair<int, int>>& dilation,
         int groups,
         mx::StreamOrDevice s) {
        std::pair<int, int> stride_pair{1, 1};
        std::pair<int, int> padding_pair{0, 0};
        std::pair<int, int> dilation_pair{1, 1};

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_pair = std::pair<int, int>{*pv, *pv};
        } else {
          stride_pair = std::get<std::pair<int, int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_pair = std::pair<int, int>{*pv, *pv};
        } else {
          padding_pair = std::get<std::pair<int, int>>(padding);
        }

        if (auto pv = std::get_if<int>(&dilation); pv) {
          dilation_pair = std::pair<int, int>{*pv, *pv};
        } else {
          dilation_pair = std::get<std::pair<int, int>>(dilation);
        }

        return mx::conv2d(
            input, weight, stride_pair, padding_pair, dilation_pair, groups, s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv2d(input: array, weight: array, /, stride: Union[int, tuple[int, int]] = 1, padding: Union[int, tuple[int, int]] = 0, dilation: Union[int, tuple[int, int]] = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        2D convolution over an input with several channels

        Args:
            input (array): Input array of shape ``(N, H, W, C_in)``.
            weight (array): Weight array of shape ``(C_out, H, W, C_in)``.
            stride (int or tuple(int), optional): :obj:`tuple` of size 2 with
                kernel strides. All spatial dimensions get the same stride if
                only one number is specified. Default: ``1``.
            padding (int or tuple(int), optional): :obj:`tuple` of size 2 with
                symmetric input padding. All spatial dimensions get the same
                padding if only one number is specified. Default: ``0``.
            dilation (int or tuple(int), optional): :obj:`tuple` of size 2 with
                kernel dilation. All spatial dimensions get the same dilation
                if only one number is specified. Default: ``1``
            groups (int, optional): input feature groups. Default: ``1``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv3d",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::tuple<int, int, int>>& stride,
         const std::variant<int, std::tuple<int, int, int>>& padding,
         const std::variant<int, std::tuple<int, int, int>>& dilation,
         int groups,
         mx::StreamOrDevice s) {
        std::tuple<int, int, int> stride_tuple{1, 1, 1};
        std::tuple<int, int, int> padding_tuple{0, 0, 0};
        std::tuple<int, int, int> dilation_tuple{1, 1, 1};

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          stride_tuple = std::get<std::tuple<int, int, int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          padding_tuple = std::get<std::tuple<int, int, int>>(padding);
        }

        if (auto pv = std::get_if<int>(&dilation); pv) {
          dilation_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          dilation_tuple = std::get<std::tuple<int, int, int>>(dilation);
        }

        return mx::conv3d(
            input,
            weight,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
            groups,
            s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv3d(input: array, weight: array, /, stride: Union[int, tuple[int, int, int]] = 1, padding: Union[int, tuple[int, int, int]] = 0, dilation: Union[int, tuple[int, int, int]] = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        3D convolution over an input with several channels

        Note: Only the default ``groups=1`` is currently supported.

        Args:
            input (array): Input array of shape ``(N, D, H, W, C_in)``.
            weight (array): Weight array of shape ``(C_out, D, H, W, C_in)``.
            stride (int or tuple(int), optional): :obj:`tuple` of size 3 with
                kernel strides. All spatial dimensions get the same stride if
                only one number is specified. Default: ``1``.
            padding (int or tuple(int), optional): :obj:`tuple` of size 3 with
                symmetric input padding. All spatial dimensions get the same
                padding if only one number is specified. Default: ``0``.
            dilation (int or tuple(int), optional): :obj:`tuple` of size 3 with
                kernel dilation. All spatial dimensions get the same dilation
                if only one number is specified. Default: ``1``
            groups (int, optional): input feature groups. Default: ``1``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv_transpose1d",
      &mx::conv_transpose1d,
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv_transpose1d(input: array, weight: array, /, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        1D transposed convolution over an input with several channels

        Args:
            input (array): Input array of shape ``(N, H, C_in)``.
            weight (array): Weight array of shape ``(C_out, H, C_in)``.
            stride (int, optional): Kernel stride. Default: ``1``.
            padding (int, optional): Input padding. Default: ``0``.
            dilation (int, optional): Kernel dilation. Default: ``1``.
            groups (int, optional): Input feature groups. Default: ``1``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv_transpose2d",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::pair<int, int>>& stride,
         const std::variant<int, std::pair<int, int>>& padding,
         const std::variant<int, std::pair<int, int>>& dilation,
         int groups,
         mx::StreamOrDevice s) {
        std::pair<int, int> stride_pair{1, 1};
        std::pair<int, int> padding_pair{0, 0};
        std::pair<int, int> dilation_pair{1, 1};

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_pair = std::pair<int, int>{*pv, *pv};
        } else {
          stride_pair = std::get<std::pair<int, int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_pair = std::pair<int, int>{*pv, *pv};
        } else {
          padding_pair = std::get<std::pair<int, int>>(padding);
        }

        if (auto pv = std::get_if<int>(&dilation); pv) {
          dilation_pair = std::pair<int, int>{*pv, *pv};
        } else {
          dilation_pair = std::get<std::pair<int, int>>(dilation);
        }

        return mx::conv_transpose2d(
            input, weight, stride_pair, padding_pair, dilation_pair, groups, s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv_transpose2d(input: array, weight: array, /, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        2D transposed convolution over an input with several channels

        Note: Only the default ``groups=1`` is currently supported.

        Args:
            input (array): Input array of shape ``(N, H, W, C_in)``.
            weight (array): Weight array of shape ``(C_out, H, W, C_in)``.
            stride (int or tuple(int), optional): :obj:`tuple` of size 2 with
                kernel strides. All spatial dimensions get the same stride if
                only one number is specified. Default: ``1``.
            padding (int or tuple(int), optional): :obj:`tuple` of size 2 with
                symmetric input padding. All spatial dimensions get the same
                padding if only one number is specified. Default: ``0``.
            dilation (int or tuple(int), optional): :obj:`tuple` of size 2 with
                kernel dilation. All spatial dimensions get the same dilation
                if only one number is specified. Default: ``1``
            groups (int, optional): input feature groups. Default: ``1``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv_transpose3d",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::tuple<int, int, int>>& stride,
         const std::variant<int, std::tuple<int, int, int>>& padding,
         const std::variant<int, std::tuple<int, int, int>>& dilation,
         int groups,
         mx::StreamOrDevice s) {
        std::tuple<int, int, int> stride_tuple{1, 1, 1};
        std::tuple<int, int, int> padding_tuple{0, 0, 0};
        std::tuple<int, int, int> dilation_tuple{1, 1, 1};

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          stride_tuple = std::get<std::tuple<int, int, int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          padding_tuple = std::get<std::tuple<int, int, int>>(padding);
        }

        if (auto pv = std::get_if<int>(&dilation); pv) {
          dilation_tuple = std::tuple<int, int, int>{*pv, *pv, *pv};
        } else {
          dilation_tuple = std::get<std::tuple<int, int, int>>(dilation);
        }

        return mx::conv_transpose3d(
            input,
            weight,
            stride_tuple,
            padding_tuple,
            dilation_tuple,
            groups,
            s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv_transpose3d(input: array, weight: array, /, stride: Union[int, Tuple[int, int, int]] = 1, padding: Union[int, Tuple[int, int, int]] = 0, dilation: Union[int, Tuple[int, int, int]] = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        3D transposed convolution over an input with several channels

        Note: Only the default ``groups=1`` is currently supported.

        Args:
            input (array): Input array of shape ``(N, D, H, W, C_in)``.
            weight (array): Weight array of shape ``(C_out, D, H, W, C_in)``.
            stride (int or tuple(int), optional): :obj:`tuple` of size 3 with
                kernel strides. All spatial dimensions get the same stride if
                only one number is specified. Default: ``1``.
            padding (int or tuple(int), optional): :obj:`tuple` of size 3 with
                symmetric input padding. All spatial dimensions get the same
                padding if only one number is specified. Default: ``0``.
            dilation (int or tuple(int), optional): :obj:`tuple` of size 3 with
                kernel dilation. All spatial dimensions get the same dilation
                if only one number is specified. Default: ``1``
            groups (int, optional): input feature groups. Default: ``1``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv_general",
      [](const mx::array& input,
         const mx::array& weight,
         const std::variant<int, std::vector<int>>& stride,
         const std::variant<
             int,
             std::vector<int>,
             std::pair<std::vector<int>, std::vector<int>>>& padding,
         const std::variant<int, std::vector<int>>& kernel_dilation,
         const std::variant<int, std::vector<int>>& input_dilation,
         int groups,
         bool flip,
         mx::StreamOrDevice s) {
        std::vector<int> stride_vec;
        std::vector<int> padding_lo_vec;
        std::vector<int> padding_hi_vec;
        std::vector<int> kernel_dilation_vec;
        std::vector<int> input_dilation_vec;

        if (auto pv = std::get_if<int>(&stride); pv) {
          stride_vec.push_back(*pv);
        } else {
          stride_vec = std::get<std::vector<int>>(stride);
        }

        if (auto pv = std::get_if<int>(&padding); pv) {
          padding_lo_vec.push_back(*pv);
          padding_hi_vec.push_back(*pv);
        } else if (auto pv = std::get_if<std::vector<int>>(&padding); pv) {
          padding_lo_vec = *pv;
          padding_hi_vec = *pv;
        } else {
          auto [pl, ph] =
              std::get<std::pair<std::vector<int>, std::vector<int>>>(padding);
          padding_lo_vec = pl;
          padding_hi_vec = ph;
        }

        if (auto pv = std::get_if<int>(&kernel_dilation); pv) {
          kernel_dilation_vec.push_back(*pv);
        } else {
          kernel_dilation_vec = std::get<std::vector<int>>(kernel_dilation);
        }

        if (auto pv = std::get_if<int>(&input_dilation); pv) {
          input_dilation_vec.push_back(*pv);
        } else {
          input_dilation_vec = std::get<std::vector<int>>(input_dilation);
        }

        return mx::conv_general(
            /* array input = */ std::move(input),
            /* array weight = */ std::move(weight),
            /* std::vector<int> stride = */ std::move(stride_vec),
            /* std::vector<int> padding_lo = */ std::move(padding_lo_vec),
            /* std::vector<int> padding_hi = */ std::move(padding_hi_vec),
            /* std::vector<int> kernel_dilation = */
            std::move(kernel_dilation_vec),
            /* std::vector<int> input_dilation = */
            std::move(input_dilation_vec),
            /* int groups = */ groups,
            /* bool flip = */ flip,
            s);
      },
      nb::arg(),
      nb::arg(),
      "stride"_a = 1,
      "padding"_a = 0,
      "kernel_dilation"_a = 1,
      "input_dilation"_a = 1,
      "groups"_a = 1,
      "flip"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def conv_general(input: array, weight: array, /, stride: Union[int, Sequence[int]] = 1, padding: Union[int, Sequence[int], tuple[Sequence[int], Sequence[int]]] = 0, kernel_dilation: Union[int, Sequence[int]] = 1, input_dilation: Union[int, Sequence[int]] = 1, groups: int = 1, flip: bool = False, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        General convolution over an input with several channels

        Args:
            input (array): Input array of shape ``(N, ..., C_in)``.
            weight (array): Weight array of shape ``(C_out, ..., C_in)``.
            stride (int or list(int), optional): :obj:`list` with kernel strides.
                All spatial dimensions get the same stride if
                only one number is specified. Default: ``1``.
            padding (int, list(int), or tuple(list(int), list(int)), optional):
                :obj:`list` with input padding. All spatial dimensions get the same
                padding if only one number is specified. Default: ``0``.
            kernel_dilation (int or list(int), optional): :obj:`list` with
                kernel dilation. All spatial dimensions get the same dilation
                if only one number is specified. Default: ``1``
            input_dilation (int or list(int), optional): :obj:`list` with
                input dilation. All spatial dimensions get the same dilation
                if only one number is specified. Default: ``1``
            groups (int, optional): Input feature groups. Default: ``1``.
            flip (bool, optional): Flip the order in which the spatial dimensions of
                the weights are processed. Performs the cross-correlation operator when
                ``flip`` is ``False`` and the convolution operator otherwise.
                Default: ``False``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "save",
      &mlx_save_helper,
      "file"_a,
      "arr"_a,
      nb::sig("def save(file: str, arr: array) -> None"),
      R"pbdoc(
        Save the array to a binary file in ``.npy`` format.

        Args:
            file (str): File to which the array is saved
            arr (array): Array to be saved.
      )pbdoc");
  m.def(
      "savez",
      [](nb::object file, nb::args args, const nb::kwargs& kwargs) {
        mlx_savez_helper(file, args, kwargs, /* compressed= */ false);
      },
      "file"_a,
      "args"_a,
      "kwargs"_a,
      R"pbdoc(
        Save several arrays to a binary file in uncompressed ``.npz``
        format.

        .. code-block:: python

            import mlx.core as mx

            x = mx.ones((10, 10))
            mx.savez("my_path.npz", x=x)

            import mlx.nn as nn
            from mlx.utils import tree_flatten

            model = nn.TransformerEncoder(6, 128, 4)
            flat_params = tree_flatten(model.parameters())
            mx.savez("model.npz", **dict(flat_params))

        Args:
            file (file, str): Path to file to which the arrays are saved.
            *args (arrays): Arrays to be saved.
            **kwargs (arrays): Arrays to be saved. Each array will be saved
              with the associated keyword as the output file name.
      )pbdoc");
  m.def(
      "savez_compressed",
      [](nb::object file, nb::args args, const nb::kwargs& kwargs) {
        mlx_savez_helper(file, args, kwargs, /*compressed=*/true);
      },
      nb::arg(),
      "args"_a,
      "kwargs"_a,
      nb::sig("def savez_compressed(file: str, *args, **kwargs)"),
      R"pbdoc(
        Save several arrays to a binary file in compressed ``.npz`` format.

        Args:
            file (file, str): Path to file to which the arrays are saved.
            *args (arrays): Arrays to be saved.
            **kwargs (arrays): Arrays to be saved. Each array will be saved
              with the associated keyword as the output file name.
      )pbdoc");
  m.def(
      "load",
      &mlx_load_helper,
      nb::arg(),
      "format"_a = nb::none(),
      "return_metadata"_a = false,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def load(file: str, /, format: Optional[str] = None, return_metadata: bool = False, *, stream: Union[None, Stream, Device] = None) -> Union[array, dict[str, array]]"),
      R"pbdoc(
        Load array(s) from a binary file.

        The supported formats are ``.npy``, ``.npz``, ``.safetensors``, and
        ``.gguf``.

        Args:
            file (file, str): File in which the array is saved.
            format (str, optional): Format of the file. If ``None``, the
              format is inferred from the file extension. Supported formats:
              ``npy``, ``npz``, and ``safetensors``. Default: ``None``.
            return_metadata (bool, optional): Load the metadata for formats
              which support matadata. The metadata will be returned as an
              additional dictionary. Default: ``False``.
        Returns:
            array or dict:
                A single array if loading from a ``.npy`` file or a dict
                mapping names to arrays if loading from a ``.npz`` or
                ``.safetensors`` file. If ``return_metadata` is ``True`` an
                additional dictionary of metadata will be returned.

        Warning:

          When loading unsupported quantization formats from GGUF, tensors
          will automatically cast to ``mx.float16``
      )pbdoc");
  m.def(
      "save_safetensors",
      &mlx_save_safetensor_helper,
      "file"_a,
      "arrays"_a,
      "metadata"_a = nb::none(),
      nb::sig(
          "def save_safetensors(file: str, arrays: dict[str, array], metadata: Optional[dict[str, str]] = None)"),
      R"pbdoc(
        Save array(s) to a binary file in ``.safetensors`` format.

        See the `Safetensors documentation
        <https://huggingface.co/docs/safetensors/index>`_ for more
        information on the format.

        Args:
            file (file, str): File in which the array is saved.
            arrays (dict(str, array)): The dictionary of names to arrays to
            be saved. metadata (dict(str, str), optional): The dictionary of
            metadata to be saved.
      )pbdoc");
  m.def(
      "save_gguf",
      &mlx_save_gguf_helper,
      "file"_a,
      "arrays"_a,
      "metadata"_a = nb::none(),
      nb::sig(
          "def save_gguf(file: str, arrays: dict[str, array], metadata: dict[str, Union[array, str, list[str]]])"),
      R"pbdoc(
        Save array(s) to a binary file in ``.gguf`` format.

        See the `GGUF documentation
        <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>`_ for
        more information on the format.

        Args:
            file (file, str): File in which the array is saved.
            arrays (dict(str, array)): The dictionary of names to arrays to
              be saved.
            metadata (dict(str, Union[array, str, list(str)])): The dictionary
               of metadata to be saved. The values can be a scalar or 1D
               obj:`array`, a :obj:`str`, or a :obj:`list` of :obj:`str`.
      )pbdoc");
  m.def(
      "where",
      [](const ScalarOrArray& condition,
         const ScalarOrArray& x_,
         const ScalarOrArray& y_,
         mx::StreamOrDevice s) {
        auto [x, y] = to_arrays(x_, y_);
        return mx::where(to_array(condition), x, y, s);
      },
      "condition"_a,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def where(condition: Union[scalar, array], x: Union[scalar, array], y: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Select from ``x`` or ``y`` according to ``condition``.

        The condition and input arrays must be the same shape or
        broadcastable with each another.

        Args:
          condition (array): The condition array.
          x (array): The input selected from where condition is ``True``.
          y (array): The input selected from where condition is ``False``.

        Returns:
            array: The output containing elements selected from
            ``x`` and ``y``.
      )pbdoc");
  m.def(
      "nan_to_num",
      [](const ScalarOrArray& a,
         float nan,
         std::optional<float>& posinf,
         std::optional<float>& neginf,
         mx::StreamOrDevice s) {
        return mx::nan_to_num(to_array(a), nan, posinf, neginf, s);
      },
      nb::arg(),
      "nan"_a = 0.0f,
      "posinf"_a = nb::none(),
      "neginf"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def nan_to_num(a: Union[scalar, array], nan: float = 0, posinf: Optional[float] = None, neginf: Optional[float] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Replace NaN and Inf values with finite numbers.

        Args:
            a (array): Input array
            nan (float, optional): Value to replace NaN with. Default: ``0``.
            posinf (float, optional): Value to replace positive infinities
              with. If ``None``, defaults to largest finite value for the
              given data type. Default: ``None``.
            neginf (float, optional): Value to replace negative infinities
              with. If ``None``, defaults to the negative of the largest
              finite value for the given data type. Default: ``None``.

        Returns:
            array: Output array with NaN and Inf replaced.
    )pbdoc");
  m.def(
      "round",
      [](const ScalarOrArray& a, int decimals, mx::StreamOrDevice s) {
        return mx::round(to_array(a), decimals, s);
      },
      nb::arg(),
      "decimals"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def round(a: array, /, decimals: int = 0, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Round to the given number of decimals.

        Basically performs:

        .. code-block:: python

          s = 10**decimals
          x = round(x * s) / s

        Args:
          a (array): Input array
          decimals (int): Number of decimal places to round to. (default: 0)

        Returns:
          array: An array of the same type as ``a`` rounded to the
          given number of decimals.
      )pbdoc");
  m.def(
      "quantized_matmul",
      &mx::quantized_matmul,
      nb::arg(),
      nb::arg(),
      "scales"_a,
      "biases"_a,
      "transpose"_a = true,
      "group_size"_a = 64,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def quantized_matmul(x: array, w: array, /, scales: array, biases: array, transpose: bool = True, group_size: int = 64, bits: int = 4, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Perform the matrix multiplication with the quantized matrix ``w``. The
        quantization uses one floating point scale and bias per ``group_size`` of
        elements. Each element in ``w`` takes ``bits`` bits and is packed in an
        unsigned 32 bit integer.

        Args:
          x (array): Input array
          w (array): Quantized matrix packed in unsigned integers
          scales (array): The scales to use per ``group_size`` elements of ``w``
          biases (array): The biases to use per ``group_size`` elements of ``w``
          transpose (bool, optional): Defines whether to multiply with the
            transposed ``w`` or not, namely whether we are performing
            ``x @ w.T`` or ``x @ w``. Default: ``True``.
          group_size (int, optional): The size of the group in ``w`` that
            shares a scale and bias. Default: ``64``.
          bits (int, optional): The number of bits occupied by each element in
            ``w``. Default: ``4``.

        Returns:
          array: The result of the multiplication of ``x`` with ``w``.
      )pbdoc");
  m.def(
      "quantize",
      &mx::quantize,
      nb::arg(),
      "group_size"_a = 64,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def quantize(w: array, /, group_size: int = 64, bits : int = 4, *, stream: Union[None, Stream, Device] = None) -> tuple[array, array, array]"),
      R"pbdoc(
        Quantize the matrix ``w`` using ``bits`` bits per element.

        Note, every ``group_size`` elements in a row of ``w`` are quantized
        together. Hence, number of columns of ``w`` should be divisible by
        ``group_size``. In particular, the rows of ``w`` are divided into groups of
        size ``group_size`` which are quantized together.

        .. warning::

          ``quantize`` currently only supports 2D inputs with dimensions which are multiples of 32

        Formally, for a group of :math:`g` consecutive elements :math:`w_1` to
        :math:`w_g` in a row of ``w`` we compute the quantized representation
        of each element :math:`\hat{w_i}` as follows

        .. math::

          \begin{aligned}
            \alpha &= \max_i w_i \\
            \beta &= \min_i w_i \\
            s &= \frac{\alpha - \beta}{2^b - 1} \\
            \hat{w_i} &= \textrm{round}\left( \frac{w_i - \beta}{s}\right).
          \end{aligned}

        After the above computation, :math:`\hat{w_i}` fits in :math:`b` bits
        and is packed in an unsigned 32-bit integer from the lower to upper
        bits. For instance, for 4-bit quantization we fit 8 elements in an
        unsigned 32 bit integer where the 1st element occupies the 4 least
        significant bits, the 2nd bits 4-7 etc.

        In order to be able to dequantize the elements of ``w`` we also need to
        save :math:`s` and :math:`\beta` which are the returned ``scales`` and
        ``biases`` respectively.

        Args:
          w (array): Matrix to be quantized
          group_size (int, optional): The size of the group in ``w`` that shares a
            scale and bias. Default: ``64``.
          bits (int, optional): The number of bits occupied by each element of
            ``w`` in the returned quantized matrix. Default: ``4``.

        Returns:
          tuple: A tuple containing

          * w_q (array): The quantized version of ``w``
          * scales (array): The scale to multiply each element with, namely :math:`s`
          * biases (array): The biases to add to each element, namely :math:`\beta`
      )pbdoc");
  m.def(
      "dequantize",
      &mx::dequantize,
      nb::arg(),
      "scales"_a,
      "biases"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def dequantize(w: array, /, scales: array, biases: array, group_size: int = 64, bits: int = 4, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Dequantize the matrix ``w`` using the provided ``scales`` and
        ``biases`` and the ``group_size`` and ``bits`` configuration.

        Formally, given the notation in :func:`quantize`, we compute
        :math:`w_i` from :math:`\hat{w_i}` and corresponding :math:`s` and
        :math:`\beta` as follows

        .. math::

          w_i = s \hat{w_i} - \beta

        Args:
          w (array): Matrix to be quantized
          scales (array): The scales to use per ``group_size`` elements of ``w``
          biases (array): The biases to use per ``group_size`` elements of ``w``
          group_size (int, optional): The size of the group in ``w`` that shares a
            scale and bias. Default: ``64``.
          bits (int, optional): The number of bits occupied by each element in
            ``w``. Default: ``4``.

        Returns:
          array: The dequantized version of ``w``
      )pbdoc");
  m.def(
      "gather_qmm",
      &mx::gather_qmm,
      nb::arg(),
      nb::arg(),
      "scales"_a,
      "biases"_a,
      "lhs_indices"_a = nb::none(),
      "rhs_indices"_a = nb::none(),
      "transpose"_a = true,
      "group_size"_a = 64,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def gather_qmm(x: array, w: array, /, scales: array, biases: array, lhs_indices: Optional[array] = None, rhs_indices: Optional[array] = None, transpose: bool = True, group_size: int = 64, bits: int = 4, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Perform quantized matrix multiplication with matrix-level gather.

        This operation is the quantized equivalent to :func:`gather_mm`.
        Similar to :func:`gather_mm`, the indices ``lhs_indices`` and
        ``rhs_indices`` contain flat indices along the batch dimensions (i.e.
        all but the last two dimensions) of ``x`` and ``w`` respectively.

        Note that ``scales`` and ``biases`` must have the same batch dimensions
        as ``w`` since they represent the same quantized matrix.

        Args:
          x (array): Input array
          w (array): Quantized matrix packed in unsigned integers
          scales (array): The scales to use per ``group_size`` elements of ``w``
          biases (array): The biases to use per ``group_size`` elements of ``w``
          lhs_indices (array, optional): Integer indices for ``x``. Default: ``None``.
          rhs_indices (array, optional): Integer indices for ``w``. Default: ``None``.
          transpose (bool, optional): Defines whether to multiply with the
            transposed ``w`` or not, namely whether we are performing
            ``x @ w.T`` or ``x @ w``. Default: ``True``.
          group_size (int, optional): The size of the group in ``w`` that
            shares a scale and bias. Default: ``64``.
          bits (int, optional): The number of bits occupied by each element in
            ``w``. Default: ``4``.

        Returns:
          array: The result of the multiplication of ``x`` with ``w``
            after gathering using ``lhs_indices`` and ``rhs_indices``.
      )pbdoc");
  m.def(
      "tensordot",
      [](const mx::array& a,
         const mx::array& b,
         const std::variant<int, std::vector<std::vector<int>>>& axes,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&axes); pv) {
          return mx::tensordot(a, b, *pv, s);
        } else {
          auto& x = std::get<std::vector<std::vector<int>>>(axes);
          if (x.size() != 2) {
            throw std::invalid_argument(
                "[tensordot] axes must be a list of two lists.");
          }
          return mx::tensordot(a, b, x[0], x[1], s);
        }
      },
      nb::arg(),
      nb::arg(),
      "axes"_a = 2,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tensordot(a: array, b: array, /, axes: Union[int, list[Sequence[int]]] = 2, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Compute the tensor dot product along the specified axes.

        Args:
          a (array): Input array
          b (array): Input array
          axes (int or list(list(int)), optional): The number of dimensions to
            sum over. If an integer is provided, then sum over the last
            ``axes`` dimensions of ``a`` and the first ``axes`` dimensions of
            ``b``. If a list of lists is provided, then sum over the
            corresponding dimensions of ``a`` and ``b``. Default: 2.

        Returns:
          array: The tensor dot product.
      )pbdoc");
  m.def(
      "inner",
      &mx::inner,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def inner(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
      Ordinary inner product of vectors for 1-D arrays, in higher dimensions a sum product over the last axes.

      Args:
        a (array): Input array
        b (array): Input array

      Returns:
        array: The inner product.
    )pbdoc");
  m.def(
      "outer",
      &mx::outer,
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def outer(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
      Compute the outer product of two 1-D arrays, if the array's passed are not 1-D a flatten op will be run beforehand.

      Args:
        a (array): Input array
        b (array): Input array

      Returns:
        array: The outer product.
    )pbdoc");
  m.def(
      "tile",
      [](const mx::array& a,
         const std::variant<int, std::vector<int>>& reps,
         mx::StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&reps); pv) {
          return mx::tile(a, {*pv}, s);
        } else {
          return mx::tile(a, std::get<std::vector<int>>(reps), s);
        }
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tile(a: array, reps: Union[int, Sequence[int]], /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
      Construct an array by repeating ``a`` the number of times given by ``reps``.

      Args:
        a (array): Input array
        reps (int or list(int)): The number of times to repeat ``a`` along each axis.

      Returns:
        array: The tiled array.
    )pbdoc");
  m.def(
      "addmm",
      &mx::addmm,
      nb::arg(),
      nb::arg(),
      nb::arg(),
      "alpha"_a = 1.0f,
      "beta"_a = 1.0f,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def addmm(c: array, a: array, b: array, /, alpha: float = 1.0, beta: float = 1.0,  *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Matrix multiplication with addition and optional scaling.

        Perform the (possibly batched) matrix multiplication of two arrays and add to the result
        with optional scaling factors.

        Args:
            c (array): Input array or scalar.
            a (array): Input array or scalar.
            b (array): Input array or scalar.
            alpha (float, optional): Scaling factor for the
                matrix product of ``a`` and ``b`` (default: ``1``)
            beta (float, optional): Scaling factor for ``c`` (default: ``1``)

        Returns:
            array: ``alpha * (a @ b)  + beta * c``
      )pbdoc");
  m.def(
      "block_masked_mm",
      &mx::block_masked_mm,
      nb::arg(),
      nb::arg(),
      "block_size"_a = 64,
      "mask_out"_a = nb::none(),
      "mask_lhs"_a = nb::none(),
      "mask_rhs"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def block_masked_mm(a: array, b: array, /, block_size: int = 64, mask_out: Optional[array] = None, mask_lhs: Optional[array] = None, mask_rhs: Optional[array] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Matrix multiplication with block masking.

        Perform the (possibly batched) matrix multiplication of two arrays and with blocks
        of size ``block_size x block_size`` optionally masked out.

        Assuming ``a`` with shape (..., `M`, `K`) and b with shape (..., `K`, `N`)

        * ``lhs_mask`` must have shape (..., :math:`\lceil` `M` / ``block_size`` :math:`\rceil`, :math:`\lceil` `K` / ``block_size`` :math:`\rceil`)

        * ``rhs_mask`` must have shape (..., :math:`\lceil` `K` / ``block_size`` :math:`\rceil`, :math:`\lceil` `N` / ``block_size`` :math:`\rceil`)

        * ``out_mask`` must have shape (..., :math:`\lceil` `M` / ``block_size`` :math:`\rceil`, :math:`\lceil` `N` / ``block_size`` :math:`\rceil`)

        Note: Only ``block_size=64`` and ``block_size=32`` are currently supported

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.
            block_size (int): Size of blocks to be masked. Must be ``32`` or ``64``. Default: ``64``.
            mask_out (array, optional): Mask for output. Default: ``None``.
            mask_lhs (array, optional): Mask for ``a``. Default: ``None``.
            mask_rhs (array, optional): Mask for ``b``. Default: ``None``.

        Returns:
            array: The output array.
      )pbdoc");
  m.def(
      "gather_mm",
      &mx::gather_mm,
      nb::arg(),
      nb::arg(),
      "lhs_indices"_a = nb::none(),
      "rhs_indices"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def gather_mm(a: array, b: array, /, lhs_indices: array, rhs_indices: array, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Matrix multiplication with matrix-level gather.

        Performs a gather of the operands with the given indices followed by a
        (possibly batched) matrix multiplication of two arrays.  This operation
        is more efficient than explicitly applying a :func:`take` followed by a
        :func:`matmul`.

        The indices ``lhs_indices`` and ``rhs_indices`` contain flat indices
        along the batch dimensions (i.e. all but the last two dimensions) of
        ``a`` and ``b`` respectively.

        For ``a`` with shape ``(A1, A2, ..., AS, M, K)``, ``lhs_indices``
        contains indices from the range ``[0, A1 * A2 * ... * AS)``

        For ``b`` with shape ``(B1, B2, ..., BS, M, K)``, ``rhs_indices``
        contains indices from the range ``[0, B1 * B2 * ... * BS)``

        Args:
            a (array): Input array.
            b (array): Input array.
            lhs_indices (array, optional): Integer indices for ``a``. Default: ``None``
            rhs_indices (array, optional): Integer indices for ``b``. Default: ``None``

        Returns:
            array: The output array.
      )pbdoc");
  m.def(
      "diagonal",
      &mx::diagonal,
      "a"_a,
      "offset"_a = 0,
      "axis1"_a = 0,
      "axis2"_a = 1,
      "stream"_a = nb::none(),
      nb::sig(
          "def diagonal(a: array, offset: int = 0, axis1: int = 0, axis2: int = 1, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Return specified diagonals.

        If ``a`` is 2-D, then a 1-D array containing the diagonal at the given
        ``offset`` is returned.

        If ``a`` has more than two dimensions, then ``axis1`` and ``axis2``
        determine the 2D subarrays from which diagonals are extracted. The new
        shape is the original shape with ``axis1`` and ``axis2`` removed and a
        new dimension inserted at the end corresponding to the diagonal.

        Args:
          a (array): Input array
          offset (int, optional): Offset of the diagonal from the main diagonal.
            Can be positive or negative. Default: ``0``.
          axis1 (int, optional): The first axis of the 2-D sub-arrays from which
              the diagonals should be taken. Default: ``0``.
          axis2 (int, optional): The second axis of the 2-D sub-arrays from which
              the diagonals should be taken. Default: ``1``.

        Returns:
            array: The diagonals of the array.
      )pbdoc");
  m.def(
      "diag",
      &mx::diag,
      nb::arg(),
      "k"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def diag(a: array, /, k: int = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Extract a diagonal or construct a diagonal matrix.
        If ``a`` is 1-D then a diagonal matrix is constructed with ``a`` on the
        :math:`k`-th diagonal. If ``a`` is 2-D then the :math:`k`-th diagonal is
        returned.

        Args:
            a (array): 1-D or 2-D input array.
            k (int, optional): The diagonal to extract or construct.
                Default: ``0``.

        Returns:
            array: The extracted diagonal or the constructed diagonal matrix.
        )pbdoc");
  m.def(
      "trace",
      [](const mx::array& a,
         int offset,
         int axis1,
         int axis2,
         std::optional<mx::Dtype> dtype,
         mx::StreamOrDevice s) {
        if (!dtype.has_value()) {
          return mx::trace(a, offset, axis1, axis2, s);
        }
        return mx::trace(a, offset, axis1, axis2, dtype.value(), s);
      },
      nb::arg(),
      "offset"_a = 0,
      "axis1"_a = 0,
      "axis2"_a = 1,
      "dtype"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def trace(a: array, /, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Return the sum along a specified diagonal in the given array.

        Args:
          a (array): Input array
          offset (int, optional): Offset of the diagonal from the main diagonal.
            Can be positive or negative. Default: ``0``.
          axis1 (int, optional): The first axis of the 2-D sub-arrays from which
              the diagonals should be taken. Default: ``0``.
          axis2 (int, optional): The second axis of the 2-D sub-arrays from which
              the diagonals should be taken. Default: ``1``.
          dtype (Dtype, optional): Data type of the output array. If
              unspecified the output type is inferred from the input array.

        Returns:
            array: Sum of specified diagonal.
        )pbdoc");
  m.def(
      "atleast_1d",
      [](const nb::args& arys, mx::StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(mx::atleast_1d(nb::cast<mx::array>(arys[0]), s));
        }
        return nb::cast(
            mx::atleast_1d(nb::cast<std::vector<mx::array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig(
          "def atleast_1d(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, list[array]]"),
      R"pbdoc(
        Convert all arrays to have at least one dimension.

        Args:
            *arys: Input arrays.
            stream (Union[None, Stream, Device], optional): The stream to execute the operation on.

        Returns:
            array or list(array): An array or list of arrays with at least one dimension.
        )pbdoc");
  m.def(
      "atleast_2d",
      [](const nb::args& arys, mx::StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(mx::atleast_2d(nb::cast<mx::array>(arys[0]), s));
        }
        return nb::cast(
            mx::atleast_2d(nb::cast<std::vector<mx::array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig(
          "def atleast_2d(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, list[array]]"),
      R"pbdoc(
        Convert all arrays to have at least two dimensions.

        Args:
            *arys: Input arrays.
            stream (Union[None, Stream, Device], optional): The stream to execute the operation on.

        Returns:
            array or list(array): An array or list of arrays with at least two dimensions.
        )pbdoc");
  m.def(
      "atleast_3d",
      [](const nb::args& arys, mx::StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(mx::atleast_3d(nb::cast<mx::array>(arys[0]), s));
        }
        return nb::cast(
            mx::atleast_3d(nb::cast<std::vector<mx::array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig(
          "def atleast_3d(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, list[array]]"),
      R"pbdoc(
        Convert all arrays to have at least three dimensions.

        Args:
            *arys: Input arrays.
            stream (Union[None, Stream, Device], optional): The stream to execute the operation on.

        Returns:
            array or list(array): An array or list of arrays with at least three dimensions.
        )pbdoc");
  m.def(
      "issubdtype",
      [](const nb::object& d1, const nb::object& d2) {
        auto dispatch_second = [](const auto& t1, const auto& d2) {
          if (nb::isinstance<mx::Dtype>(d2)) {
            return mx::issubdtype(t1, nb::cast<mx::Dtype>(d2));
          } else if (nb::isinstance<mx::Dtype::Category>(d2)) {
            return mx::issubdtype(t1, nb::cast<mx::Dtype::Category>(d2));
          } else {
            throw std::invalid_argument(
                "[issubdtype] Received invalid type for second input.");
          }
        };
        if (nb::isinstance<mx::Dtype>(d1)) {
          return dispatch_second(nb::cast<mx::Dtype>(d1), d2);
        } else if (nb::isinstance<mx::Dtype::Category>(d1)) {
          return dispatch_second(nb::cast<mx::Dtype::Category>(d1), d2);
        } else {
          throw std::invalid_argument(
              "[issubdtype] Received invalid type for first input.");
        }
      },
      ""_a,
      ""_a,
      nb::sig(
          "def issubdtype(arg1: Union[Dtype, DtypeCategory], arg2: Union[Dtype, DtypeCategory]) -> bool"),
      R"pbdoc(
        Check if a :obj:`Dtype` or :obj:`DtypeCategory` is a subtype
        of another.

        Args:
            arg1 (Union[Dtype, DtypeCategory]: First dtype or category.
            arg2 (Union[Dtype, DtypeCategory]: Second dtype or category.

        Returns:
            bool:
               A boolean indicating if the first input is a subtype of the
               second input.

        Example:

          >>> ints = mx.array([1, 2, 3], dtype=mx.int32)
          >>> mx.issubdtype(ints.dtype, mx.integer)
          True
          >>> mx.issubdtype(ints.dtype, mx.floating)
          False

          >>> floats = mx.array([1, 2, 3], dtype=mx.float32)
          >>> mx.issubdtype(floats.dtype, mx.integer)
          False
          >>> mx.issubdtype(floats.dtype, mx.floating)
          True

          Similar types of different sizes are not subdtypes of each other:

          >>> mx.issubdtype(mx.float64, mx.float32)
          False
          >>> mx.issubdtype(mx.float32, mx.float64)
          False

          but both are subtypes of `floating`:

          >>> mx.issubdtype(mx.float64, mx.floating)
          True
          >>> mx.issubdtype(mx.float32, mx.floating)
          True

          For convenience, dtype-like objects are allowed too:

          >>> mx.issubdtype(mx.float32, mx.inexact)
          True
          >>> mx.issubdtype(mx.signedinteger, mx.floating)
          False
      )pbdoc");
  m.def(
      "bitwise_and",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::bitwise_and(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def bitwise_and(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise bitwise and.

        Take the bitwise and of two arrays with numpy-style broadcasting
        semantics. Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The bitwise and ``a & b``.
      )pbdoc");
  m.def(
      "bitwise_or",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::bitwise_or(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def bitwise_or(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise bitwise or.

        Take the bitwise or of two arrays with numpy-style broadcasting
        semantics. Either or both input arrays can also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The bitwise or``a | b``.
      )pbdoc");
  m.def(
      "bitwise_xor",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::bitwise_xor(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def bitwise_xor(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise bitwise xor.

        Take the bitwise exclusive or of two arrays with numpy-style
        broadcasting semantics. Either or both input arrays can also be
        scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The bitwise xor ``a ^ b``.
      )pbdoc");
  m.def(
      "left_shift",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::left_shift(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def left_shift(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise left shift.

        Shift the bits of the first input to the left by the second using
        numpy-style broadcasting semantics. Either or both input arrays can
        also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The bitwise left shift ``a << b``.
      )pbdoc");
  m.def(
      "right_shift",
      [](const ScalarOrArray& a_,
         const ScalarOrArray& b_,
         mx::StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return mx::right_shift(a, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def right_shift(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Element-wise right shift.

        Shift the bits of the first input to the right by the second using
        numpy-style broadcasting semantics. Either or both input arrays can
        also be scalars.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.

        Returns:
            array: The bitwise right shift ``a >> b``.
      )pbdoc");
  m.def(
      "view",
      [](const ScalarOrArray& a, const mx::Dtype& dtype, mx::StreamOrDevice s) {
        return mx::view(to_array(a), dtype, s);
      },
      nb::arg(),
      "dtype"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def view(a: Union[scalar, array], dtype: Dtype, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        View the array as a different type.

        The output shape changes along the last axis if the input array's
        type and the input ``dtype`` do not have the same size.

        Note: the view op does not imply that the input and output arrays share
        their underlying data. The view only gaurantees that the binary
        representation of each element (or group of elements) is the same.

        Args:
            a (array): Input array or scalar.
            dtype (Dtype): The data type to change to.

        Returns:
            array: The array with the new type.
      )pbdoc");
  m.def(
      "hadamard_transform",
      &mx::hadamard_transform,
      nb::arg(),
      "scale"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def hadamard_transform(a: array, scale: Optional[float] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Perform the Walsh-Hadamard transform along the final axis.

        Equivalent to:

        .. code-block:: python

           from scipy.linalg import hadamard

           y = (hadamard(len(x)) @ x) * scale

        Supports sizes ``n = m*2^k`` for ``m`` in ``(1, 12, 20, 28)`` and ``2^k
        <= 8192`` for float32 and ``2^k <= 16384`` for float16/bfloat16.

        Args:
            a (array): Input array or scalar.
            scale (float): Scale the output by this factor.
              Defaults to ``1/sqrt(a.shape[-1])`` so that the Hadamard matrix is orthonormal.

        Returns:
            array: The transformed array.
      )pbdoc");
  m.def(
      "einsum_path",
      [](const std::string& equation, const nb::args& operands) {
        auto arrays_list = nb::cast<std::vector<mx::array>>(operands);
        auto [path, str] = mx::einsum_path(equation, arrays_list);
        // Convert to list of tuples
        std::vector<nb::tuple> tuple_path;
        for (auto& p : path) {
          tuple_path.push_back(nb::tuple(nb::cast(p)));
        }
        return std::make_pair(tuple_path, str);
      },
      "subscripts"_a,
      "operands"_a,
      nb::sig("def einsum_path(subscripts: str, *operands)"),
      R"pbdoc(

      Compute the contraction order for the given Einstein summation.

      Args:
        subscripts (str): The Einstein summation convention equation.
        *operands (array): The input arrays.

      Returns:
        tuple(list(tuple(int, int)), str):
          The einsum path and a string containing information about the
          chosen path.
    )pbdoc");
  m.def(
      "einsum",
      [](const std::string& subscripts,
         const nb::args& operands,
         mx::StreamOrDevice s) {
        auto arrays_list = nb::cast<std::vector<mx::array>>(operands);
        return mx::einsum(subscripts, arrays_list, s);
      },
      "subscripts"_a,
      "operands"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def einsum(subscripts: str, *operands, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(

      Perform the Einstein summation convention on the operands.

      Args:
        subscripts (str): The Einstein summation convention equation.
        *operands (array): The input arrays.

      Returns:
        array: The output array.
    )pbdoc");
  m.def(
      "roll",
      [](const mx::array& a,
         const std::variant<int, mx::Shape>& shift,
         const IntOrVec& axis,
         mx::StreamOrDevice s) {
        return std::visit(
            [&](auto sh, auto ax) -> mx::array {
              if constexpr (std::is_same_v<decltype(ax), std::monostate>) {
                return mx::roll(a, sh, s);
              } else {
                return mx::roll(a, sh, ax, s);
              }
            },
            shift,
            axis);
      },
      nb::arg(),
      "shift"_a,
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def roll(a: array, shift: Union[int, Tuple[int]], axis: Union[None, int, Tuple[int]] = None, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Roll array elements along a given axis.

        Elements that are rolled beyond the end of the array are introduced at
        the beggining and vice-versa.

        If the axis is not provided the array is flattened, rolled and then the
        shape is restored.

        Args:
          a (array): Input array
          shift (int or tuple(int)): The number of places by which elements
            are shifted. If positive the array is rolled to the right, if
            negative it is rolled to the left. If an int is provided but the
            axis is a tuple then the same value is used for all axes.
          axis (int or tuple(int), optional): The axis or axes along which to
            roll the elements.
      )pbdoc");
  m.def(
      "real",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::real(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def real(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Returns the real part of a complex array.

        Args:
            a (array): Input array.

        Returns:
            array: The real part of ``a``.
      )pbdoc");
  m.def(
      "imag",
      [](const ScalarOrArray& a, mx::StreamOrDevice s) {
        return mx::imag(to_array(a), s);
      },
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def imag(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Returns the imaginary part of a complex array.

        Args:
            a (array): Input array.

        Returns:
            array: The imaginary part of ``a``.
      )pbdoc");
}
