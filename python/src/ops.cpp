// Copyright © 2023 Apple Inc.

#include <numeric>
#include <ostream>
#include <variant>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlx/ops.h"
#include "mlx/utils.h"
#include "python/src/load.h"
#include "python/src/utils.h"

namespace py = pybind11;
using namespace py::literals;
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

void init_ops(py::module_& m) {
  py::options options;
  options.disable_function_signatures();

  m.def(
      "reshape",
      &reshape,
      "a"_a,
      py::pos_only(),
      "shape"_a,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        reshape(a: array, /, shape: List[int], *, stream: Union[None, Stream, Device] = None) -> array

        Reshape an array while preserving the size.

        Args:
            a (array): Input array.
            shape (tuple(int)): New shape.
            stream (Stream, optional): Stream or device. Defaults to ```None```
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
      "a"_a,
      py::pos_only(),
      "start_axis"_a = 0,
      "end_axis"_a = -1,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
      flatten(a: array, /, start_axis: int = 0, end_axis: int = -1, *, stream: Union[None, Stream, Device] = None) -> array

      Flatten an array.

      Args:
          a (array): Input array.
          start_axis (int, optional): The first dimension to flatten. Defaults to ``0``.
          end_axis (int, optional): The last dimension to flatten. Defaults to ``-1``.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

      Returns:
          array: The flattened array.
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
      "a"_a,
      py::pos_only(),
      "axis"_a = none,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        squeeze(a: array, /, axis: Union[None, int, List[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array

        Remove length one axes from an array.

        Args:
            a (array): Input array.
            axis (int or tuple(int), optional): Axes to remove. Defaults
            to ```None``` in which case all size one axes are removed.

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
      "a"_a,
      py::pos_only(),
      "axis"_a,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        expand_dims(a: array, /, axis: Union[int, List[int]], *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        abs(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise absolute value.

        Args:
            a (array): Input array.

        Returns:
            array: The absolute value of ``a``.
      )pbdoc");
  m.def(
      "sign",
      &sign,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        sign(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise sign.

        Args:
            a (array): Input array.

        Returns:
            array: The sign of ``a``.
      )pbdoc");
  m.def(
      "negative",
      &negative,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        negative(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        add(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        subtract(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        multiply(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        divide(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array

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
      "floor_divide",
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return floor_divide(a, b, s);
      },
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        floor_divide(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        remainder(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        not_equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        less(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        less_equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        greater(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        greater_equal(a: Union[scalar, array], b: Union[scalar, array], stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "equal_nan"_a = false,
      "stream"_a = none,
      R"pbdoc(
        array_equal(a: Union[scalar, array], b: Union[scalar, array], equal_nan: bool = False, stream: Union[None, Stream, Device] = None) -> array

        Array equality check.

        Compare two arrays for equality. Returns ``True`` if and only if the arrays
        have the same shape and their values are equal. The arrays need not have
        the same type to be considered equal.

        Args:
            a (array): Input array or scalar.
            b (array): Input array or scalar.
            equal_nan (bool): If ``True``, NaNs are treated as equal.
              Defaults to ``False``.

        Returns:
            array: A scalar boolean array.
      )pbdoc");
  m.def(
      "matmul",
      &matmul,
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        matmul(a: array, b: array, /, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        square(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise square.

        Args:
            a (array): Input array.

        Returns:
            array: The square of ``a``.
      )pbdoc");
  m.def(
      "sqrt",
      &mlx::core::sqrt,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        sqrt(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise square root.

        Args:
            a (array): Input array.

        Returns:
            array: The square root of ``a``.
      )pbdoc");
  m.def(
      "rsqrt",
      &rsqrt,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        rsqrt(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise reciprocal and square root.

        Args:
            a (array): Input array.

        Returns:
            array: One over the square root of ``a``.
      )pbdoc");
  m.def(
      "reciprocal",
      &reciprocal,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        reciprocal(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        logical_not(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise logical not.

        Args:
            a (array): Input array or scalar.

        Returns:
            array: The boolean array containing the logical not of ``a``.
      )pbdoc");
  m.def(
      "logaddexp",
      [](const ScalarOrArray& a_, const ScalarOrArray& b_, StreamOrDevice s) {
        auto [a, b] = to_arrays(a_, b_);
        return logaddexp(a, b, s);
      },
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        logaddexp(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        exp(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise exponential.

        Args:
            a (array): Input array.

        Returns:
            array: The exponential of ``a``.
      )pbdoc");
  m.def(
      "erf",
      &mlx::core::erf,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        erf(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise error function.

        .. math::
          \mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^t e^{-t^2} \, dx

        Args:
            a (array): Input array.

        Returns:
            array: The error function of ``a``.
      )pbdoc");
  m.def(
      "erfinv",
      &mlx::core::erfinv,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        erfinv(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise inverse of :func:`erf`.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse error function of ``a``.
      )pbdoc");
  m.def(
      "sin",
      &mlx::core::sin,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        sin(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise sine.

        Args:
            a (array): Input array.

        Returns:
            array: The sine of ``a``.
      )pbdoc");
  m.def(
      "cos",
      &mlx::core::cos,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        cos(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise cosine.

        Args:
            a (array): Input array.

        Returns:
            array: The cosine of ``a``.
      )pbdoc");
  m.def(
      "tan",
      &mlx::core::tan,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        tan(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise tangent.

        Args:
            a (array): Input array.

        Returns:
            array: The tangent of ``a``.
      )pbdoc");
  m.def(
      "arcsin",
      &mlx::core::arcsin,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        arcsin(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise inverse sine.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse sine of ``a``.
      )pbdoc");
  m.def(
      "arccos",
      &mlx::core::arccos,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        arccos(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise inverse cosine.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse cosine of ``a``.
      )pbdoc");
  m.def(
      "arctan",
      &mlx::core::arctan,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        arctan(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise inverse tangent.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse tangent of ``a``.
      )pbdoc");
  m.def(
      "sinh",
      &mlx::core::sinh,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        sinh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise hyperbolic sine.

        Args:
            a (array): Input array.

        Returns:
            array: The hyperbolic sine of ``a``.
      )pbdoc");
  m.def(
      "cosh",
      &mlx::core::cosh,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        cosh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise hyperbolic cosine.

        Args:
            a (array): Input array.

        Returns:
            array: The hyperbolic cosine of ``a``.
      )pbdoc");
  m.def(
      "tanh",
      &mlx::core::tanh,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        tanh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise hyperbolic tangent.

        Args:
            a (array): Input array.

        Returns:
            array: The hyperbolic tangent of ``a``.
      )pbdoc");
  m.def(
      "arcsinh",
      &mlx::core::arcsinh,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        arcsinh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise inverse hyperbolic sine.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse hyperbolic sine of ``a``.
      )pbdoc");
  m.def(
      "arccosh",
      &mlx::core::arccosh,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        arccosh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise inverse hyperbolic cosine.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse hyperbolic cosine of ``a``.
      )pbdoc");
  m.def(
      "arctanh",
      &mlx::core::arctanh,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        arctanh(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise inverse hyperbolic tangent.

        Args:
            a (array): Input array.

        Returns:
            array: The inverse hyperbolic tangent of ``a``.
      )pbdoc");
  m.def(
      "log",
      &mlx::core::log,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        log(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise natural logarithm.

        Args:
            a (array): Input array.

        Returns:
            array: The natural logarithm of ``a``.
      )pbdoc");
  m.def(
      "log2",
      &mlx::core::log2,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        log2(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise base-2 logarithm.

        Args:
            a (array): Input array.

        Returns:
            array: The base-2 logarithm of ``a``.
      )pbdoc");
  m.def(
      "log10",
      &mlx::core::log10,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        log10(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise base-10 logarithm.

        Args:
            a (array): Input array.

        Returns:
            array: The base-10 logarithm of ``a``.
      )pbdoc");
  m.def(
      "log1p",
      &mlx::core::log1p,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        log1p(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise natural log of one plus the array.

        Args:
            a (array): Input array.

        Returns:
            array: The natural logarithm of one plus ``a``.
      )pbdoc");
  m.def(
      "stop_gradient",
      &stop_gradient,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        stop_gradient(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        sigmoid(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        power(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array

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
      [](Scalar stop, std::optional<Dtype> dtype_, StreamOrDevice s) {
        Dtype dtype =
            dtype_.has_value() ? dtype_.value() : scalar_to_dtype(stop);

        return arange(0.0, scalar_to_double(stop), 1.0, dtype, s);
      },
      "stop"_a,
      "dtype"_a = none,
      "stream"_a = none);
  m.def(
      "arange",
      [](Scalar start,
         Scalar stop,
         std::optional<Dtype> dtype_,
         StreamOrDevice s) {
        Dtype dtype = dtype_.has_value()
            ? dtype_.value()
            : promote_types(scalar_to_dtype(start), scalar_to_dtype(stop));
        return arange(
            scalar_to_double(start), scalar_to_double(stop), dtype, s);
      },
      "start"_a,
      "stop"_a,
      "dtype"_a = none,
      "stream"_a = none);
  m.def(
      "arange",
      [](Scalar stop,
         Scalar step,
         std::optional<Dtype> dtype_,
         StreamOrDevice s) {
        Dtype dtype = dtype_.has_value()
            ? dtype_.value()
            : promote_types(scalar_to_dtype(stop), scalar_to_dtype(step));

        return arange(
            0.0, scalar_to_double(stop), scalar_to_double(step), dtype, s);
      },
      "stop"_a,
      "step"_a,
      "dtype"_a = none,
      "stream"_a = none);
  m.def(
      "arange",
      [](Scalar start,
         Scalar stop,
         Scalar step,
         std::optional<Dtype> dtype_,
         StreamOrDevice s) {
        // Determine the final dtype based on input types
        Dtype dtype = dtype_.has_value()
            ? dtype_.value()
            : promote_types(
                  scalar_to_dtype(start),
                  promote_types(scalar_to_dtype(stop), scalar_to_dtype(step)));

        return arange(
            scalar_to_double(start),
            scalar_to_double(stop),
            scalar_to_double(step),
            dtype,
            s);
      },
      "start"_a,
      "stop"_a,
      "step"_a,
      "dtype"_a = none,
      "stream"_a = none,
      R"pbdoc(
      arange(start, stop, step, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array

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
      "linspace",
      [](Scalar start, Scalar stop, int num, Dtype dtype, StreamOrDevice s) {
        return linspace(
            scalar_to_double(start), scalar_to_double(stop), num, dtype, s);
      },
      "start"_a,
      "stop"_a,
      "num"_a = 50,
      "dtype"_a = float32,
      "stream"_a = none,
      R"pbdoc(
      linspace(start, stop, num: Optional[int] = 50, dtype: Optional[Dtype] = float32, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "indices"_a,
      "axis"_a = std::nullopt,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        take(a: array, /, indices: array, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array

        Take elements along an axis.

        The elements are taken from ``indices`` along the specified axis.
        If the axis is not specified the array is treated as a flattened
        1-D array prior to performing the take.

        As an example, if the ``axis=1`` this is equialent to ``a[:, indices, ...]``.

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
      "a"_a,
      py::pos_only(),
      "indices"_a,
      "axis"_a,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        take_along_axis(a: array, /, indices: array, axis: Optional[int] = None, *, stream: Union[None, Stream, Device] = None) -> array

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
      "dtype"_a = std::nullopt,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        full(shape: Union[int, List[int]], vals: Union[scalar, array], dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array

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
      "dtype"_a = std::nullopt,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        zeros(shape: Union[int, List[int]], dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        zeros_like(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

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
      "dtype"_a = std::nullopt,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        ones(shape: Union[int, List[int]], dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        ones_like(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

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
      "m"_a = py::none(),
      "k"_a = 0,
      "dtype"_a = std::nullopt,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
      eye(n: int, m: Optional[int] = None, k: int = 0, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array

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
      "dtype"_a = std::nullopt,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
      identity(n: int, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array

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
      [](int n, std::optional<int> m, int k, Dtype dtype, StreamOrDevice s) {
        return tri(n, m.value_or(n), k, float32, s);
      },
      "n"_a,
      "m"_a = none,
      "k"_a = 0,
      "dtype"_a = float32,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        tri(n: int, m: int, k: int, dtype: Optional[Dtype] = None, *, stream: Union[None, Stream, Device] = None) -> array

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
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
      tril(x: array, k: int, *, stream: Union[None, Stream, Device] = None) -> array

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
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
      triu(x: array, k: int, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      "rtol"_a = 1e-5,
      "atol"_a = 1e-8,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        allclose(a: array, b: array, /, rtol: float = 1e-05, atol: float = 1e-08, *, stream: Union[None, Stream, Device] = None) -> array

        Approximate comparison of two arrays.

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
      "a"_a,
      py::pos_only(),
      "axis"_a = none,
      "keepdims"_a = false,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        all(a: array, /, axis: Union[None, int, List[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "axis"_a = none,
      "keepdims"_a = false,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        any(a: array, /, axis: Union[None, int, List[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        minimum(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "b"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        maximum(a: Union[scalar, array], b: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        floor(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise floor.

        Args:
            a (array): Input array.

        Returns:
            array: The floor of ``a``.
      )pbdoc");
  m.def(
      "ceil",
      &mlx::core::ceil,
      "a"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        ceil(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

        Element-wise ceil.

        Args:
            a (array): Input array.

        Returns:
            array: The ceil of ``a``.
      )pbdoc");
  m.def(
      "moveaxis",
      &moveaxis,
      "a"_a,
      py::pos_only(),
      "source"_a,
      "destiantion"_a,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        moveaxis(a: array, /, source: int, destination: int, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "axis1"_a,
      "axis2"_a,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        swapaxes(a: array, /, axis1 : int, axis2: int, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "axes"_a = std::nullopt,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        transpose(a: array, /, axes: Optional[List[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array

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
      py::pos_only(),
      "axis"_a = none,
      "keepdims"_a = false,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        sum(a: array, /, axis: Union[None, int, List[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "axis"_a = none,
      "keepdims"_a = false,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        prod(a: array, /, axis: Union[None, int, List[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "axis"_a = none,
      "keepdims"_a = false,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        min(a: array, /, axis: Union[None, int, List[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array

        An `min` reduction over the given axes.

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
      "a"_a,
      py::pos_only(),
      "axis"_a = none,
      "keepdims"_a = false,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        max(a: array, /, axis: Union[None, int, List[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array

        An `max` reduction over the given axes.

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
      "a"_a,
      py::pos_only(),
      "axis"_a = none,
      "keepdims"_a = false,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        logsumexp(a: array, /, axis: Union[None, int, List[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "axis"_a = none,
      "keepdims"_a = false,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        mean(a: array, /, axis: Union[None, int, List[int]] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "axis"_a = none,
      "keepdims"_a = false,
      "ddof"_a = 0,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        var(a: array, /, axis: Union[None, int, List[int]] = None, keepdims: bool = False, ddof: int = 0, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "indices_or_sections"_a,
      "axis"_a = 0,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        split(a: array, /, indices_or_sections: Union[int, List[int]], axis: int = 0, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "axis"_a = std::nullopt,
      "keepdims"_a = false,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        argmin(a: array, /, axis: Union[None, int] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array

        Indices of the minimum values along the axis.

        Args:
            a (array): Input array.
            axis (int, optional): Optional axis to reduce over. If unspecified
              this defaults to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the indices of the minimum values.
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
      "a"_a,
      py::pos_only(),
      "axis"_a = std::nullopt,
      "keepdims"_a = false,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        argmax(a: array, /, axis: Union[None, int] = None, keepdims: bool = False, *, stream: Union[None, Stream, Device] = None) -> array

        Indices of the maximum values along the axis.

        Args:
            a (array): Input array.
            axis (int, optional): Optional axis to reduce over. If unspecified
              this defaults to reducing over the entire array.
            keepdims (bool, optional): Keep reduced axes as
              singleton dimensions, defaults to `False`.

        Returns:
            array: The output array with the indices of the minimum values.
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
      "a"_a,
      py::pos_only(),
      "axis"_a = -1,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        sort(a: array, /, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "axis"_a = -1,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        argsort(a: array, /, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array

        Returns the indices that sort the array.

        Args:
            a (array): Input array.
            axis (int or None, optional): Optional axis to sort over.
              If ``None``, this sorts over the flattened array.
              If unspecified, it defaults to -1 (sorting over the last axis).

        Returns:
            array: The indices that sort the input array.
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
      "a"_a,
      py::pos_only(),
      "kth"_a,
      "axis"_a = -1,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        partition(a: array, /, kth: int, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "kth"_a,
      "axis"_a = -1,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        argpartition(a: array, /, kth: int, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array

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
            axis (int or None, optional): Optional axis to partiton over.
              If ``None``, this partitions over the flattened array.
              If unspecified, it defaults to ``-1``.

        Returns:
            array: The indices that partition the input array.
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
      "a"_a,
      py::pos_only(),
      "k"_a,
      "axis"_a = -1,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        topk(a: array, /, k: int, axis: Union[None, int] = -1, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "shape"_a,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        broadcast_to(a: Union[scalar, array], /, shape: List[int], *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "axis"_a = none,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        softmax(a: array, /, axis: Union[None, int, List[int]] = None, *, stream: Union[None, Stream, Device] = None) -> array

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
      "arrays"_a,
      py::pos_only(),
      "axis"_a = 0,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        concatenate(arrays: List[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array

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
      "arrays"_a,
      py::pos_only(),
      "axis"_a = 0,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
      stack(arrays: List[array], axis: Optional[int] = 0, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "a_min"_a,
      "a_max"_a,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
      clip(a: array, /, a_min: Union[scalar, array, None], a_max: Union[scalar, array, None], *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "pad_width"_a,
      "constant_values"_a = 0,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        pad(a: array, pad_with: Union[int, Tuple[int], Tuple[int, int], List[Tuple[int, int]]], constant_values: Union[scalar, array] = 0, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "shape"_a = none,
      "strides"_a = none,
      "offset"_a = 0,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        as_strided(a: array, /, shape: Optional[List[int]] = None, strides: Optional[List[int]] = None, offset: int = 0, *, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "axis"_a = std::nullopt,
      py::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = none,
      R"pbdoc(
        cumsum(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "axis"_a = std::nullopt,
      py::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = none,
      R"pbdoc(
        cumprod(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "axis"_a = std::nullopt,
      py::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = none,
      R"pbdoc(
        cummax(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      py::pos_only(),
      "axis"_a = std::nullopt,
      py::kw_only(),
      "reverse"_a = false,
      "inclusive"_a = true,
      "stream"_a = none,
      R"pbdoc(
        cummin(a: array, /, axis: Optional[int] = None, *, reverse: bool = False, inclusive: bool = True, stream: Union[None, Stream, Device] = None) -> array

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
      "a"_a,
      "v"_a,
      py::pos_only(),
      "mode"_a = "full",
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        convolve(a: array, v: array, /, mode: str = "full", *, stream: Union[None, Stream, Device] = None) -> array

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
      "input"_a,
      "weight"_a,
      py::pos_only(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        conv1d(input: array, weight: array, /, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, *, stream: Union[None, Stream, Device] = None) -> array

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
      "input"_a,
      "weight"_a,
      py::pos_only(),
      "stride"_a = 1,
      "padding"_a = 0,
      "dilation"_a = 1,
      "groups"_a = 1,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        conv2d(input: array, weight: array, /, stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0, dilation: Union[int, Tuple[int, int]] = 1, groups: Union[int, Tuple[int, int]] = 1, *, stream: Union[None, Stream, Device] = None) -> array

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
      "save",
      &mlx_save_helper,
      "file"_a,
      "arr"_a,
      py::pos_only(),
      "retain_graph"_a = std::nullopt,
      py::kw_only(),
      R"pbdoc(
        save(file: str, arr: array, / , retain_graph: Optional[bool] = None)

        Save the array to a binary file in ``.npy`` format.

        Args:
            file (str): File to which the array is saved
            arr (array): Array to be saved.
            retain_graph (bool, optional): Optional argument to retain graph
              during array evaluation before saving. If not provided the graph
              is retained if we are during a function transformation. Default:
              None

      )pbdoc");
  m.def(
      "savez",
      [](py::object file, py::args args, const py::kwargs& kwargs) {
        mlx_savez_helper(file, args, kwargs, /*compressed=*/false);
      },
      "file"_a,
      py::pos_only(),
      py::kw_only(),
      R"pbdoc(
        savez(file: str, *args, **kwargs)

        Save several arrays to a binary file in uncompressed ``.npz`` format.

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
      [](py::object file, py::args args, const py::kwargs& kwargs) {
        mlx_savez_helper(file, args, kwargs, /*compressed=*/true);
      },
      "file"_a,
      py::pos_only(),
      py::kw_only(),
      R"pbdoc(
        savez_compressed(file: str, *args, **kwargs)

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
      "file"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        load(file: str, /, *, stream: Union[None, Stream, Device] = None) -> Union[array, Dict[str, array]]

        Load array(s) from a binary file in ``.npy`` or ``.npz`` format.

        Args:
            file (file, str): File in which the array is saved

        Returns:
            result (array, dict): The loaded array if ``.npy`` file or a dict mapping name to array if ``.npz`` file
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
      "x"_a,
      "y"_a,
      py::pos_only(),
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        where(condition: Union[scalar, array], x: Union[scalar, array], y: Union[scalar, array], /, *, stream: Union[None, Stream, Device] = None) -> array

        Select from ``x`` or ``y`` according to ``condition``.

        The condition and input arrays must be the same shape or broadcastable
        with each another.

        Args:
          condition (array): The condition array.
          x (array): The input selected from where condition is ``True``.
          y (array): The input selected from where condition is ``False``.

        Returns:
            result (array): The output containing elements selected from ``x`` and ``y``.
      )pbdoc");
  m.def(
      "round",
      [](const array& a, int decimals, StreamOrDevice s) {
        return round(a, decimals, s);
      },
      "a"_a,
      py::pos_only(),
      "decimals"_a = 0,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        round(a: array, /, decimals: int = 0, stream: Union[None, Stream, Device] = None) -> array

        Round to the given number of decimals.

        Bascially performs:

        .. code-block:: python

          s = 10**decimals
          x = round(x * s) / s

        Args:
          a (array): Input array
          decimals (int): Number of decimal places to round to. (default: 0)

        Returns:
          result (array): An array of the same type as ``a`` rounded to the given number of decimals.
      )pbdoc");
  m.def(
      "quantized_matmul",
      &quantized_matmul,
      "x"_a,
      "w"_a,
      py::pos_only(),
      "scales"_a,
      "biases"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        quantized_matmul(x: array, w: array, scales: array, biases: array, /, group_size: int = 64, bits: int = 4, *, stream: Union[None, Stream, Device] = None) -> array

        Perform the matrix multiplication with the quantized matrix ``w``. The
        quantization uses one floating point scale and bias per ``group_size`` of
        elements. Each element in ``w`` takes ``bits`` bits and is packed in an
        unsigned 32 bit integer.

        Args:
          x (array): Input array
          w (array): Quantized matrix packed in unsigned integers
          scales (array): The scales to use per ``group_size`` elements of ``w``
          biases (array): The biases to use per ``group_size`` elements of ``w``
          group_size (int, optional): The size of the group in ``w`` that
            shares a scale and bias. (default: 64)
          bits (int, optional): The number of bits occupied by each element in
            ``w``. (default: 4)

        Returns:
          result (array): The result of the multiplication of ``x`` with ``w``.
      )pbdoc");
  m.def(
      "quantize",
      &quantize,
      "w"_a,
      py::pos_only(),
      "group_size"_a = 64,
      "bits"_a = 4,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        quantize(w: array, /, group_size: int = 64, bits : int = 4, *, stream: Union[None, Stream, Device] = None) -> Tuple[array, array, array]

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
            scale and bias. (default: 64)
          bits (int, optional): The number of bits occupied by each element of
            ``w`` in the returned quantized matrix. (default: 4)

        Returns:
          (tuple): A tuple containing

            - w_q (array): The quantized version of ``w``
            - scales (array): The scale to multiply each element with, namely :math:`s`
            - biases (array): The biases to add to each element, namely :math:`\beta`
      )pbdoc");
  m.def(
      "dequantize",
      &dequantize,
      "w"_a,
      py::pos_only(),
      "scales"_a,
      "biases"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      py::kw_only(),
      "stream"_a = none,
      R"pbdoc(
        dequantize(w: array, /, scales: array, biases: array, group_size: int = 64, bits: int = 4, *, stream: Union[None, Stream, Device] = None) -> array

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
            scale and bias. (default: 64)
          bits (int, optional): The number of bits occupied by each element in
            ``w``. (default: 4)

        Returns:
          result (array): The dequantized version of ``w``
      )pbdoc");
}
