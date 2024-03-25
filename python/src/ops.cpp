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

#include "mlx/ops.h"
#include "mlx/utils.h"
#include "python/src/load.h"
#include "python/src/utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

using Scalar = std::variant<int, double>;

Dtype scalar_to_dtype(Scalar scalar) {
  if (std::holds_alternative<int>(scalar)) {
    return int32;
  } else {
    return float32;
  }
}

double scalar_to_double(Scalar s) {
  if (std::holds_alternative<double>(s)) {
    return std::get<double>(s);
  } else {
    return static_cast<double>(std::get<int>(s));
  }
}

void init_ops(nb::module_& m) {
  m.def(
      "reshape",
      &reshape,
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
      [](const array& a,
         int start_axis,
         int end_axis,
         const StreamOrDevice& s) { return flatten(a, start_axis, end_axis); },
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
      "squeeze",
      [](const array& a, const IntOrVec& v, const StreamOrDevice& s) {
        if (std::holds_alternative<std::monostate>(v)) {
          return squeeze(a, s);
        } else if (auto pv = std::get_if<int>(&v); pv) {
          return squeeze(a, *pv, s);
        } else {
          return squeeze(a, std::get<std::vector<int>>(v), s);
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
      [](const array& a,
         const std::variant<int, std::vector<int>>& v,
         StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&v); pv) {
          return expand_dims(a, *pv, s);
        } else {
          return expand_dims(a, std::get<std::vector<int>>(v), s);
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
      &mlx::core::abs,
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
      &sign,
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
      &negative,
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return add(a, b, s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return subtract(a, b, s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return multiply(a, b, s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return divide(a, b, s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return divmod(a, b, s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return floor_divide(a, b, s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return remainder(a, b, s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return equal(a, b, s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return not_equal(a, b, s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return less(a, b, s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return less_equal(a, b, s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return greater(a, b, s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return greater_equal(a, b, s);
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
         StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return array_equal(a, b, equal_nan, s);
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
      &matmul,
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
      &square,
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
      &mlx::core::sqrt,
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
      &rsqrt,
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
      &reciprocal,
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
      [](const ScalarOrArray& a, StreamOrDevice s) {
        return logical_not(to_array(a), s);
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
      [](const ScalarOrArray& a, const ScalarOrArray& b, StreamOrDevice s) {
        return logical_and(to_array(a), to_array(b), s);
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
      [](const ScalarOrArray& a, const ScalarOrArray& b, StreamOrDevice s) {
        return logical_or(to_array(a), to_array(b), s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return logaddexp(a, b, s);
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
      &mlx::core::exp,
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
      "erf",
      &mlx::core::erf,
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
      &mlx::core::erfinv,
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
      &mlx::core::sin,
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
      &mlx::core::cos,
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
      &mlx::core::tan,
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
      &mlx::core::arcsin,
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
      &mlx::core::arccos,
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
      &mlx::core::arctan,
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
      "sinh",
      &mlx::core::sinh,
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
      &mlx::core::cosh,
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
      &mlx::core::tanh,
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
      &mlx::core::arcsinh,
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
      &mlx::core::arccosh,
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
      &mlx::core::arctanh,
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
      "log",
      &mlx::core::log,
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
      &mlx::core::log2,
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
      &mlx::core::log10,
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
      &mlx::core::log1p,
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
      &stop_gradient,
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
            array: The unchanged input ``a`` but without gradient flowing
              through it.
      )pbdoc");
  m.def(
      "sigmoid",
      &sigmoid,
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return power(a, b, s);
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
         const std::optional<Dtype>& dtype_,
         StreamOrDevice s) {
        // Determine the final dtype based on input types
        Dtype dtype = dtype_
            ? *dtype_
            : promote_types(
                  scalar_to_dtype(start),
                  step ? promote_types(
                             scalar_to_dtype(stop), scalar_to_dtype(*step))
                       : scalar_to_dtype(stop));
        return arange(
            scalar_to_double(start),
            scalar_to_double(stop),
            step ? scalar_to_double(*step) : 1.0,
            dtype,
            s);
      },
      "start"_a,
      "stop"_a,
      "step"_a = nb::none(),
      nb::kw_only(),
      "dtype"_a = nb::none(),
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
          dtype (Dtype, optional): Specifies the data type of the output.
            If unspecified will default to ``float32`` if any of ``start``,
            ``stop``, or ``step`` are ``float``. Otherwise will default to
            ``int32``.

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
         const std::optional<Dtype>& dtype_,
         StreamOrDevice s) {
        Dtype dtype = dtype_ ? *dtype_
            : step
            ? promote_types(scalar_to_dtype(stop), scalar_to_dtype(*step))
            : scalar_to_dtype(stop);
        return arange(
            0.0,
            scalar_to_double(stop),
            step ? scalar_to_double(*step) : 1.0,
            dtype,
            s);
      },
      "stop"_a,
      "step"_a = nb::none(),
      nb::kw_only(),
      "dtype"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def arange(stop : Union[int, float], step : Union[None, int, float], dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array"));
  m.def(
      "linspace",
      [](Scalar start,
         Scalar stop,
         int num,
         std::optional<Dtype> dtype,
         StreamOrDevice s) {
        return linspace(
            scalar_to_double(start),
            scalar_to_double(stop),
            num,
            dtype.value_or(float32),
            s);
      },
      "start"_a,
      "stop"_a,
      "num"_a = 50,
      "dtype"_a.none() = float32,
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
      [](const array& a,
         const array& indices,
         const std::optional<int>& axis,
         StreamOrDevice s) {
        if (axis.has_value()) {
          return take(a, indices, axis.value(), s);
        } else {
          return take(a, indices, s);
        }
      },
      nb::arg(),
      "indices"_a,
      "axis"_a = nb::none(),
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def take(a: array, /, indices: array, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Take elements along an axis.

        The elements are taken from ``indices`` along the specified axis.
        If the axis is not specified the array is treated as a flattened
        1-D array prior to performing the take.

        As an example, if the ``axis=1`` this is equivalent to ``a[:, indices, ...]``.

        Args:
            a (array): Input array.
            indices (array): Input array with integral type.
            axis (int, optional): Axis along which to perform the take. If unspecified
              the array is treated as a flattened 1-D vector.

        Returns:
            array: The indexed values of ``a``.
      )pbdoc");
  m.def(
      "take_along_axis",
      [](const array& a,
         const array& indices,
         const std::optional<int>& axis,
         StreamOrDevice s) {
        if (axis.has_value()) {
          return take_along_axis(a, indices, axis.value(), s);
        } else {
          return take_along_axis(reshape(a, {-1}, s), indices, 0, s);
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
            array: The output array with the specified shape and values.
      )pbdoc");
  m.def(
      "full",
      [](const std::variant<int, std::vector<int>>& shape,
         const ScalarOrArray& vals,
         std::optional<Dtype> dtype,
         StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&shape); pv) {
          return full({*pv}, to_array(vals, dtype), s);
        } else {
          return full(
              std::get<std::vector<int>>(shape), to_array(vals, dtype), s);
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
      [](const std::variant<int, std::vector<int>>& shape,
         std::optional<Dtype> dtype,
         StreamOrDevice s) {
        auto t = dtype.value_or(float32);
        if (auto pv = std::get_if<int>(&shape); pv) {
          return zeros({*pv}, t, s);
        } else {
          return zeros(std::get<std::vector<int>>(shape), t, s);
        }
      },
      "shape"_a,
      "dtype"_a.none() = float32,
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
      &zeros_like,
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
      [](const std::variant<int, std::vector<int>>& shape,
         std::optional<Dtype> dtype,
         StreamOrDevice s) {
        auto t = dtype.value_or(float32);
        if (auto pv = std::get_if<int>(&shape); pv) {
          return ones({*pv}, t, s);
        } else {
          return ones(std::get<std::vector<int>>(shape), t, s);
        }
      },
      "shape"_a,
      "dtype"_a.none() = float32,
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
      &ones_like,
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
         std::optional<Dtype> dtype,
         StreamOrDevice s) {
        return eye(n, m.value_or(n), k, dtype.value_or(float32), s);
      },
      "n"_a,
      "m"_a = nb::none(),
      "k"_a = 0,
      "dtype"_a.none() = float32,
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
      [](int n, std::optional<Dtype> dtype, StreamOrDevice s) {
        return identity(n, dtype.value_or(float32), s);
      },
      "n"_a,
      "dtype"_a.none() = float32,
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
         std::optional<Dtype> type,
         StreamOrDevice s) {
        return tri(n, m.value_or(n), k, type.value_or(float32), s);
      },
      "n"_a,
      "m"_a = nb::none(),
      "k"_a = 0,
      "dtype"_a.none() = float32,
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
      &tril,
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
      &triu,
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
      &allclose,
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
      &isclose,
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
      [](const array& a,
         const IntOrVec& axis,
         bool keepdims,
         StreamOrDevice s) {
        return all(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
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
      [](const array& a,
         const IntOrVec& axis,
         bool keepdims,
         StreamOrDevice s) {
        return any(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return minimum(a, b, s);
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
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return maximum(a, b, s);
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
      &mlx::core::floor,
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
      &mlx::core::ceil,
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
      &mlx::core::isnan,
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
      &mlx::core::isinf,
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
      "isposinf",
      &isposinf,
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
      &isneginf,
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
      &moveaxis,
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
      &swapaxes,
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
      [](const array& a,
         const std::optional<std::vector<int>>& axes,
         StreamOrDevice s) {
        if (axes.has_value()) {
          return transpose(a, get_reduce_axes(axes.value(), a.ndim()), s);
        } else {
          return transpose(a, s);
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
      "sum",
      [](const array& a,
         const IntOrVec& axis,
         bool keepdims,
         StreamOrDevice s) {
        return sum(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
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
      [](const array& a,
         const IntOrVec& axis,
         bool keepdims,
         StreamOrDevice s) {
        return prod(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
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
      [](const array& a,
         const IntOrVec& axis,
         bool keepdims,
         StreamOrDevice s) {
        return min(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
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
      [](const array& a,
         const IntOrVec& axis,
         bool keepdims,
         StreamOrDevice s) {
        return max(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
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
      [](const array& a,
         const IntOrVec& axis,
         bool keepdims,
         StreamOrDevice s) {
        return logsumexp(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
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
      [](const array& a,
         const IntOrVec& axis,
         bool keepdims,
         StreamOrDevice s) {
        return mean(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
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
      [](const array& a,
         const IntOrVec& axis,
         bool keepdims,
         int ddof,
         StreamOrDevice s) {
        return var(a, get_reduce_axes(axis, a.ndim()), keepdims, ddof, s);
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
      "split",
      [](const array& a,
         const std::variant<int, std::vector<int>>& indices_or_sections,
         int axis,
         StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&indices_or_sections); pv) {
          return split(a, *pv, axis, s);
        } else {
          return split(
              a, std::get<std::vector<int>>(indices_or_sections), axis, s);
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
      [](const array& a,
         std::optional<int> axis,
         bool keepdims,
         StreamOrDevice s) {
        if (axis) {
          return argmin(a, *axis, keepdims, s);
        } else {
          return argmin(a, keepdims, s);
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
      [](const array& a,
         std::optional<int> axis,
         bool keepdims,
         StreamOrDevice s) {
        if (axis) {
          return argmax(a, *axis, keepdims, s);
        } else {
          return argmax(a, keepdims, s);
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
      [](const array& a, std::optional<int> axis, StreamOrDevice s) {
        if (axis) {
          return sort(a, *axis, s);
        } else {
          return sort(a, s);
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
      [](const array& a, std::optional<int> axis, StreamOrDevice s) {
        if (axis) {
          return argsort(a, *axis, s);
        } else {
          return argsort(a, s);
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
      [](const array& a, int kth, std::optional<int> axis, StreamOrDevice s) {
        if (axis) {
          return partition(a, kth, *axis, s);
        } else {
          return partition(a, kth, s);
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
      [](const array& a, int kth, std::optional<int> axis, StreamOrDevice s) {
        if (axis) {
          return argpartition(a, kth, *axis, s);
        } else {
          return argpartition(a, kth, s);
        }
      },
      nb::arg(),
      "kth"_a,
      "axis"_a = -1,
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
            array: The `uint32`` array containing indices that partition the input.
      )pbdoc");
  m.def(
      "topk",
      [](const array& a, int k, std::optional<int> axis, StreamOrDevice s) {
        if (axis) {
          return topk(a, k, *axis, s);
        } else {
          return topk(a, k, s);
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
      [](const ScalarOrArray& a,
         const std::vector<int>& shape,
         StreamOrDevice s) { return broadcast_to(to_array(a), shape, s); },
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
      "softmax",
      [](const array& a, const IntOrVec& axis, StreamOrDevice s) {
        return softmax(a, get_reduce_axes(axis, a.ndim()), s);
      },
      nb::arg(),
      "axis"_a = nb::none(),
      nb::kw_only(),
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
      [](const std::vector<array>& arrays,
         std::optional<int> axis,
         StreamOrDevice s) {
        if (axis) {
          return concatenate(arrays, *axis, s);
        } else {
          return concatenate(arrays, s);
        }
      },
      nb::arg(),
      "axis"_a.none() = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def concatenate(arrays: List[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
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
      "stack",
      [](const std::vector<array>& arrays,
         std::optional<int> axis,
         StreamOrDevice s) {
        if (axis.has_value()) {
          return stack(arrays, axis.value(), s);
        } else {
          return stack(arrays, s);
        }
      },
      nb::arg(),
      "axis"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def stack(arrays: List[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
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
      "repeat",
      [](const array& array,
         int repeats,
         std::optional<int> axis,
         StreamOrDevice s) {
        if (axis.has_value()) {
          return repeat(array, repeats, axis.value(), s);
        } else {
          return repeat(array, repeats, s);
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
      [](const array& a,
         const std::optional<ScalarOrArray>& min,
         const std::optional<ScalarOrArray>& max,
         StreamOrDevice s) {
        std::optional<array> min_ = std::nullopt;
        std::optional<array> max_ = std::nullopt;
        if (min) {
          min_ = to_array(min.value());
        }
        if (max) {
          max_ = to_array(max.value());
        }
        return clip(a, min_, max_, s);
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
      [](const array& a,
         const std::variant<
             int,
             std::tuple<int>,
             std::pair<int, int>,
             std::vector<std::pair<int, int>>>& pad_width,
         const ScalarOrArray& constant_value,
         StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&pad_width); pv) {
          return pad(a, *pv, to_array(constant_value), s);
        } else if (auto pv = std::get_if<std::tuple<int>>(&pad_width); pv) {
          return pad(a, std::get<0>(*pv), to_array(constant_value), s);
        } else if (auto pv = std::get_if<std::pair<int, int>>(&pad_width); pv) {
          return pad(a, *pv, to_array(constant_value), s);
        } else {
          auto v = std::get<std::vector<std::pair<int, int>>>(pad_width);
          if (v.size() == 1) {
            return pad(a, v[0], to_array(constant_value), s);
          } else {
            return pad(a, v, to_array(constant_value), s);
          }
        }
      },
      nb::arg(),
      "pad_width"_a,
      "constant_values"_a = 0,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def pad(a: array, pad_with: Union[int, Tuple[int], Tuple[int, int], List[Tuple[int, int]]], constant_values: Union[scalar, array] = 0, *, stream: Union[None, Stream, Device] = None) -> array"),
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
            constant_value (array or scalar, optional): Optional constant value
              to pad the edges of the array with.

        Returns:
            array: The padded array.
      )pbdoc");
  m.def(
      "as_strided",
      [](const array& a,
         std::optional<std::vector<int>> shape,
         std::optional<std::vector<size_t>> strides,
         size_t offset,
         StreamOrDevice s) {
        std::vector<int> a_shape = (shape) ? *shape : a.shape();
        std::vector<size_t> a_strides;
        if (strides) {
          a_strides = *strides;
        } else {
          std::fill_n(std::back_inserter(a_strides), a_shape.size(), 1);
          for (int i = a_shape.size() - 1; i > 0; i--) {
            a_strides[i - 1] = a_shape[i] * a_strides[i];
          }
        }
        return as_strided(a, a_shape, a_strides, offset, s);
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
      [](const array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         StreamOrDevice s) {
        if (axis) {
          return cumsum(a, *axis, reverse, inclusive, s);
        } else {
          return cumsum(reshape(a, {-1}, s), 0, reverse, inclusive, s);
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
      )pbdoc");
  m.def(
      "cumprod",
      [](const array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         StreamOrDevice s) {
        if (axis) {
          return cumprod(a, *axis, reverse, inclusive, s);
        } else {
          return cumprod(reshape(a, {-1}, s), 0, reverse, inclusive, s);
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
      )pbdoc");
  m.def(
      "cummax",
      [](const array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         StreamOrDevice s) {
        if (axis) {
          return cummax(a, *axis, reverse, inclusive, s);
        } else {
          return cummax(reshape(a, {-1}, s), 0, reverse, inclusive, s);
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
      )pbdoc");
  m.def(
      "cummin",
      [](const array& a,
         std::optional<int> axis,
         bool reverse,
         bool inclusive,
         StreamOrDevice s) {
        if (axis) {
          return cummin(a, *axis, reverse, inclusive, s);
        } else {
          return cummin(reshape(a, {-1}, s), 0, reverse, inclusive, s);
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
      )pbdoc");
  m.def(
      "convolve",
      [](const array& a,
         const array& v,
         const std::string& mode,
         StreamOrDevice s) {
        if (a.ndim() != 1 || v.ndim() != 1) {
          throw std::invalid_argument("[convolve] Inputs must be 1D.");
        }

        if (a.size() == 0 || v.size() == 0) {
          throw std::invalid_argument("[convolve] Inputs cannot be empty.");
        }

        array in = a.size() < v.size() ? v : a;
        array wt = a.size() < v.size() ? a : v;
        wt = slice(wt, {wt.shape(0) - 1}, {-wt.shape(0) - 1}, {-1}, s);

        in = reshape(in, {1, -1, 1}, s);
        wt = reshape(wt, {1, -1, 1}, s);

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
            in = pad(in, {{0, 0}, {pad_l, pad_r}, {0, 0}}, array(0), s);
          }

        } else {
          throw std::invalid_argument("[convolve] Invalid mode.");
        }

        array out = conv1d(
            in,
            wt,
            /*stride = */ 1,
            /*padding = */ padding,
            /*dilation = */ 1,
            /*groups = */ 1,
            s);

        return reshape(out, {-1}, s);
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
      &conv1d,
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

        Note: Only the default ``groups=1`` is currently supported.

        Args:
            input (array): input array of shape (``N``, ``H``, ``C_in``)
            weight (array): weight array of shape (``C_out``, ``H``, ``C_in``)
            stride (int, optional): kernel stride. Default: ``1``.
            padding (int, optional): input padding. Default: ``0``.
            dilation (int, optional): kernel dilation. Default: ``1``.
            groups (int, optional): input feature groups. Default: ``1``.

        Returns:
            array: The convolved array.
      )pbdoc");
  m.def(
      "conv2d",
      [](const array& input,
         const array& weight,
         const std::variant<int, std::pair<int, int>>& stride,
         const std::variant<int, std::pair<int, int>>& padding,
         const std::variant<int, std::pair<int, int>>& dilation,
         int groups,
         StreamOrDevice s) {
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

        return conv2d(
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
          "def conv2d(input: array, weight: array, /, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        2D convolution over an input with several channels

        Note: Only the default ``groups=1`` is currently supported.

        Args:
            input (array): input array of shape ``(N, H, W, C_in)``
            weight (array): weight array of shape ``(C_out, H, W, C_in)``
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
      "conv_general",
      [](const array& input,
         const array& weight,
         const std::variant<int, std::vector<int>>& stride,
         const std::variant<
             int,
             std::vector<int>,
             std::pair<std::vector<int>, std::vector<int>>>& padding,
         const std::variant<int, std::vector<int>>& kernel_dilation,
         const std::variant<int, std::vector<int>>& input_dilation,
         int groups,
         bool flip,
         StreamOrDevice s) {
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

        return conv_general(
            /* const array& input = */ input,
            /* const array& weight = */ weight,
            /* std::vector<int> stride = */ stride_vec,
            /* std::vector<int> padding_lo = */ padding_lo_vec,
            /* std::vector<int> padding_hi = */ padding_lo_vec,
            /* std::vector<int> kernel_dilation = */ kernel_dilation_vec,
            /* std::vector<int> input_dilation = */ input_dilation_vec,
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
          "def conv_general(input: array, weight: array, /, stride: Union[int, Sequence[int]] = 1, padding: Union[int, Sequence[int], Tuple[Sequence[int], Sequence[int]]] = 0, kernel_dilation: Union[int, Sequence[int]] = 1, input_dilation: Union[int, Sequence[int]] = 1, groups: int = 1, flip: bool = false, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        General convolution over an input with several channels

        .. note::

           * Only 1d and 2d convolutions are supported at the moment
           * the default ``groups=1`` is currently supported.

        Args:
            input (array): Input array of shape ``(N, ..., C_in)``
            weight (array): Weight array of shape ``(C_out, ..., C_in)``
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
            args (arrays): Arrays to be saved.
            kwargs (arrays): Arrays to be saved. Each array will be saved
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
            args (arrays): Arrays to be saved.
            kwargs (arrays): Arrays to be saved. Each array will be saved
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
          "def load(file: str, /, format: Optional[str] = None, return_metadata: bool = False, *, stream: Union[None, Stream, Device] = None) -> Union[array, Dict[str, array]]"),
      R"pbdoc(
        Load array(s) from a binary file.

        The supported formats are ``.npy``, ``.npz``, ``.safetensors``, and
        ``.gguf``.

        Args:
            file (file, str): File in which the array is saved.
            format (str, optional): Format of the file. If ``None``, the
            format
              is inferred from the file extension. Supported formats:
              ``npy``,
              ``npz``, and ``safetensors``. Default: ``None``.
            return_metadata (bool, optional): Load the metadata for formats
            which
              support matadata. The metadata will be returned as an
              additional dictionary.
        Returns:
            result (array, dict):
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
          "def save_safetensors(file: str, arrays: Dict[str, array], metadata: Optional[Dict[str, str]] = None)"),
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
          "def save_gguf(file: str, arrays: Dict[str, array], metadata: Dict[str, Union[array, str, List[str]]])"),
      R"pbdoc(
        Save array(s) to a binary file in ``.gguf`` format.

        See the `GGUF documentation
        <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>`_ for
        more information on the format.

        Args:
            file (file, str): File in which the array is saved.
            arrays (dict(str, array)): The dictionary of names to arrays to
            be saved. metadata (dict(str, Union[array, str, list(str)])):
            The dictionary of
               metadata to be saved. The values can be a scalar or 1D
               obj:`array`, a :obj:`str`, or a :obj:`list` of :obj:`str`.
      )pbdoc");
  m.def(
      "where",
      [](const ScalarOrArray& condition,
         const ScalarOrArray& x_,
         const ScalarOrArray& y_,
         StreamOrDevice s) {
        auto [x, y] = to_arrays(x_, y_);
        return where(to_array(condition), x, y, s);
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
            result (array): The output containing elements selected from
            ``x`` and ``y``.
      )pbdoc");
  m.def(
      "round",
      [](const array& a, int decimals, StreamOrDevice s) {
        return round(a, decimals, s);
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
          result (array): An array of the same type as ``a`` rounded to the
          given number of decimals.
      )pbdoc");
  m.def(
      "quantized_matmul",
      &quantized_matmul,
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
            ``x @ w.T`` or ``x @ w``. (default: ``True``)
          group_size (int, optional): The size of the group in ``w`` that
            shares a scale and bias. (default: ``64``)
          bits (int, optional): The number of bits occupied by each element in
            ``w``. (default: ``4``)

        Returns:
          result (array): The result of the multiplication of ``x`` with ``w``.
      )pbdoc");
  m.def(
      "quantize",
      &quantize,
      nb::arg(),
      "group_size"_a = 64,
      "bits"_a = 4,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def quantize(w: array, /, group_size: int = 64, bits : int = 4, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array, array]"),
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
            scale and bias. (default: ``64``)
          bits (int, optional): The number of bits occupied by each element of
            ``w`` in the returned quantized matrix. (default: ``4``)

        Returns:
          (tuple): A tuple containing

            - w_q (array): The quantized version of ``w``
            - scales (array): The scale to multiply each element with, namely :math:`s`
            - biases (array): The biases to add to each element, namely :math:`\beta`
      )pbdoc");
  m.def(
      "dequantize",
      &dequantize,
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
            scale and bias. (default: ``64``)
          bits (int, optional): The number of bits occupied by each element in
            ``w``. (default: ``4``)

        Returns:
          result (array): The dequantized version of ``w``
      )pbdoc");
  m.def(
      "tensordot",
      [](const array& a,
         const array& b,
         const std::variant<int, std::vector<std::vector<int>>>& axes,
         StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&axes); pv) {
          return tensordot(a, b, *pv, s);
        } else {
          auto& x = std::get<std::vector<std::vector<int>>>(axes);
          if (x.size() != 2) {
            throw std::invalid_argument(
                "[tensordot] axes must be a list of two lists.");
          }
          return tensordot(a, b, x[0], x[1], s);
        }
      },
      nb::arg(),
      nb::arg(),
      "axes"_a = 2,
      nb::kw_only(),
      "stream"_a = nb::none(),
      nb::sig(
          "def tensordot(a: array, b: array, /, axes: Union[int, List[Sequence[int]]] = 2, *, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Compute the tensor dot product along the specified axes.

        Args:
          a (array): Input array
          b (array): Input array
          axes (int or list(list(int)), optional): The number of dimensions to
            sum over. If an integer is provided, then sum over the last
            ``axes`` dimensions of ``a`` and the first ``axes`` dimensions of
            ``b``. If a list of lists is provided, then sum over the
            corresponding dimensions of ``a`` and ``b``. (default: 2)

        Returns:
          result (array): The tensor dot product.
      )pbdoc");
  m.def(
      "inner",
      &inner,
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
        result (array): The inner product.
    )pbdoc");
  m.def(
      "outer",
      &outer,
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
        result (array): The outer product.
    )pbdoc");
  m.def(
      "tile",
      [](const array& a, const IntOrVec& reps, StreamOrDevice s) {
        if (auto pv = std::get_if<int>(&reps); pv) {
          return tile(a, {*pv}, s);
        } else {
          return tile(a, std::get<std::vector<int>>(reps), s);
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
        result (array): The tiled array.
    )pbdoc");
  m.def(
      "addmm",
      &addmm,
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
      "diagonal",
      &diagonal,
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
      &diag,
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
      "atleast_1d",
      [](const nb::args& arys, StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(atleast_1d(nb::cast<array>(arys[0]), s));
        }
        return nb::cast(atleast_1d(nb::cast<std::vector<array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig(
          "def atleast_1d(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, List[array]]"),
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
      [](const nb::args& arys, StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(atleast_2d(nb::cast<array>(arys[0]), s));
        }
        return nb::cast(atleast_2d(nb::cast<std::vector<array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig(
          "def atleast_2d(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, List[array]]"),
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
      [](const nb::args& arys, StreamOrDevice s) -> nb::object {
        if (arys.size() == 1) {
          return nb::cast(atleast_3d(nb::cast<array>(arys[0]), s));
        }
        return nb::cast(atleast_3d(nb::cast<std::vector<array>>(arys), s));
      },
      "arys"_a,
      "stream"_a = nb::none(),
      nb::sig(
          "def atleast_3d(*arys: array, stream: Union[None, Stream, Device] = None) -> Union[array, List[array]]"),
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
      nb::overload_cast<const Dtype&, const Dtype&>(&issubdtype),
      ""_a,
      ""_a,
      R"pbdoc(
        Check if a :obj:`Dtype` or :obj:`DtypeCategory` is a subtype
        of another.

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
      "issubdtype",
      nb::overload_cast<const Dtype&, const Dtype::Category&>(&issubdtype),
      ""_a,
      ""_a);
  m.def(
      "issubdtype",
      nb::overload_cast<const Dtype::Category&, const Dtype&>(&issubdtype),
      ""_a,
      ""_a);
  m.def(
      "issubdtype",
      nb::overload_cast<const Dtype::Category&, const Dtype::Category&>(
          &issubdtype),
      ""_a,
      ""_a);
}
