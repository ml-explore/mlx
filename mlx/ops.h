// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx::core {

/** Creation operations */

/**
 * A 1D array of numbers starting at `start` (optional),
 * stopping at stop, stepping by `step` (optional). */
array arange(
    double start,
    double stop,
    double step,
    Dtype dtype,
    StreamOrDevice s = {});
array arange(double start, double stop, double step, StreamOrDevice s = {});
array arange(double start, double stop, Dtype dtype, StreamOrDevice s = {});
array arange(double start, double stop, StreamOrDevice s = {});
array arange(double stop, Dtype dtype, StreamOrDevice s = {});
array arange(double stop, StreamOrDevice s = {});

array arange(int start, int stop, int step, StreamOrDevice s = {});
array arange(int start, int stop, StreamOrDevice s = {});
array arange(int stop, StreamOrDevice s = {});

/** A 1D array of `num` evenly spaced numbers in the range `[start, stop]` */
array linspace(
    double start,
    double stop,
    int num = 50,
    Dtype dtype = float32,
    StreamOrDevice s = {});

/** Convert an array to the given data type. */
array astype(const array& a, Dtype dtype, StreamOrDevice s = {});

/** Create a view of an array with the given shape and strides. */
array as_strided(
    const array& a,
    std::vector<int> shape,
    std::vector<size_t> strides,
    size_t offset,
    StreamOrDevice s = {});

/** Copy another array. */
array copy(const array& a, StreamOrDevice s = {});

/** Fill an array of the given shape with the given value(s). */
array full(
    const std::vector<int>& shape,
    const array& vals,
    Dtype dtype,
    StreamOrDevice s = {});
array full(
    const std::vector<int>& shape,
    const array& vals,
    StreamOrDevice s = {});
template <typename T>
array full(
    const std::vector<int>& shape,
    T val,
    Dtype dtype,
    StreamOrDevice s = {}) {
  return full(shape, array(val, dtype), to_stream(s));
}
template <typename T>
array full(const std::vector<int>& shape, T val, StreamOrDevice s = {}) {
  return full(shape, array(val), to_stream(s));
}

/** Fill an array of the given shape with zeros. */
array zeros(const std::vector<int>& shape, Dtype dtype, StreamOrDevice s = {});
inline array zeros(const std::vector<int>& shape, StreamOrDevice s = {}) {
  return zeros(shape, float32, s);
}
array zeros_like(const array& a, StreamOrDevice s = {});

/** Fill an array of the given shape with ones. */
array ones(const std::vector<int>& shape, Dtype dtype, StreamOrDevice s = {});
inline array ones(const std::vector<int>& shape, StreamOrDevice s = {}) {
  return ones(shape, float32, s);
}
array ones_like(const array& a, StreamOrDevice s = {});

/** Fill an array of the given shape (n,m) with ones in the specified diagonal
 * k, and zeros everywhere else. */
array eye(int n, int m, int k, Dtype dtype, StreamOrDevice s = {});
inline array eye(int n, Dtype dtype, StreamOrDevice s = {}) {
  return eye(n, n, 0, dtype, s);
}
inline array eye(int n, int m, StreamOrDevice s = {}) {
  return eye(n, m, 0, float32, s);
}
inline array eye(int n, int m, int k, StreamOrDevice s = {}) {
  return eye(n, m, k, float32, s);
}
inline array eye(int n, StreamOrDevice s = {}) {
  return eye(n, n, 0, float32, s);
}

/** Create a square matrix of shape (n,n) of zeros, and ones in the major
 * diagonal. */
array identity(int n, Dtype dtype, StreamOrDevice s = {});
inline array identity(int n, StreamOrDevice s = {}) {
  return identity(n, float32, s);
}

array tri(int n, int m, int k, Dtype type, StreamOrDevice s = {});
inline array tri(int n, Dtype type, StreamOrDevice s = {}) {
  return tri(n, n, 0, type, s);
}

array tril(array x, int k = 0, StreamOrDevice s = {});
array triu(array x, int k = 0, StreamOrDevice s = {});

/** array manipulation */

/** Reshape an array to the given shape. */
array reshape(const array& a, std::vector<int> shape, StreamOrDevice s = {});

/** Flatten the dimensions in the range `[start_axis, end_axis]` . */
array flatten(
    const array& a,
    int start_axis,
    int end_axis = -1,
    StreamOrDevice s = {});

/** Flatten the array to 1D. */
array flatten(const array& a, StreamOrDevice s = {});

/** Remove singleton dimensions at the given axes. */
array squeeze(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s = {});

/** Remove singleton dimensions at the given axis. */
inline array squeeze(const array& a, int axis, StreamOrDevice s = {}) {
  return squeeze(a, std::vector<int>{axis}, s);
}

/** Remove all singleton dimensions. */
array squeeze(const array& a, StreamOrDevice s = {});

/** Add a singleton dimension at the given axes. */
array expand_dims(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s = {});

/** Add a singleton dimension at the given axis. */
inline array expand_dims(const array& a, int axis, StreamOrDevice s = {}) {
  return expand_dims(a, std::vector<int>{axis}, s);
}

/** Slice an array. */
array slice(
    const array& a,
    std::vector<int> start,
    std::vector<int> stop,
    std::vector<int> strides,
    StreamOrDevice s = {});

/** Slice an array with a stride of 1 in each dimension. */
array slice(
    const array& a,
    const std::vector<int>& start,
    const std::vector<int>& stop,
    StreamOrDevice s = {});

/** Split an array into sub-arrays along a given axis. */
std::vector<array>
split(const array& a, int num_splits, int axis, StreamOrDevice s = {});
std::vector<array> split(const array& a, int num_splits, StreamOrDevice s = {});
std::vector<array> split(
    const array& a,
    const std::vector<int>& indices,
    int axis,
    StreamOrDevice s = {});
std::vector<array>
split(const array& a, const std::vector<int>& indices, StreamOrDevice s = {});

/**
 * Clip (limit) the values in an array.
 */
array clip(
    const array& a,
    const std::optional<array>& a_min = std::nullopt,
    const std::optional<array>& a_max = std::nullopt,
    StreamOrDevice s = {});

/** Concatenate arrays along a given axis. */
array concatenate(
    const std::vector<array>& arrays,
    int axis,
    StreamOrDevice s = {});
array concatenate(const std::vector<array>& arrays, StreamOrDevice s = {});

/** Stack arrays along a new axis. */
array stack(const std::vector<array>& arrays, int axis, StreamOrDevice s = {});
array stack(const std::vector<array>& arrays, StreamOrDevice s = {});

/** Repeat an array along an axis. */
array repeat(const array& arr, int repeats, int axis, StreamOrDevice s = {});
array repeat(const array& arr, int repeats, StreamOrDevice s = {});

array tile(const array& arr, std::vector<int> reps, StreamOrDevice s = {});

/** Permutes the dimensions according to the given axes. */
array transpose(const array& a, std::vector<int> axes, StreamOrDevice s = {});
inline array transpose(
    const array& a,
    std::initializer_list<int> axes,
    StreamOrDevice s = {}) {
  return transpose(a, std::vector<int>(axes), s);
}

/** Swap two axes of an array. */
array swapaxes(const array& a, int axis1, int axis2, StreamOrDevice s = {});

/** Move an axis of an array. */
array moveaxis(
    const array& a,
    int source,
    int destination,
    StreamOrDevice s = {});

/** Pad an array with a constant value */
array pad(
    const array& a,
    const std::vector<int>& axes,
    const std::vector<int>& low_pad_size,
    const std::vector<int>& high_pad_size,
    const array& pad_value = array(0),
    StreamOrDevice s = {});

/** Pad an array with a constant value along all axes */
array pad(
    const array& a,
    const std::vector<std::pair<int, int>>& pad_width,
    const array& pad_value = array(0),
    StreamOrDevice s = {});
array pad(
    const array& a,
    const std::pair<int, int>& pad_width,
    const array& pad_value = array(0),
    StreamOrDevice s = {});
array pad(
    const array& a,
    int pad_width,
    const array& pad_value = array(0),
    StreamOrDevice s = {});

/** Permutes the dimensions in reverse order. */
array transpose(const array& a, StreamOrDevice s = {});

/** Broadcast an array to a given shape. */
array broadcast_to(
    const array& a,
    const std::vector<int>& shape,
    StreamOrDevice s = {});

/** Broadcast a vector of arrays against one another. */
std::vector<array> broadcast_arrays(
    const std::vector<array>& inputs,
    StreamOrDevice s = {});

/** Comparison operations */

/** Returns the bool array with (a == b) element-wise. */
array equal(const array& a, const array& b, StreamOrDevice s = {});
inline array operator==(const array& a, const array& b) {
  return equal(a, b);
}
template <typename T>
array operator==(T a, const array& b) {
  return equal(array(a), b);
}
template <typename T>
array operator==(const array& a, T b) {
  return equal(a, array(b));
}

/** Returns the bool array with (a != b) element-wise. */
array not_equal(const array& a, const array& b, StreamOrDevice s = {});
inline array operator!=(const array& a, const array& b) {
  return not_equal(a, b);
}
template <typename T>
array operator!=(T a, const array& b) {
  return not_equal(array(a), b);
}
template <typename T>
array operator!=(const array& a, T b) {
  return not_equal(a, array(b));
}

/** Returns bool array with (a > b) element-wise. */
array greater(const array& a, const array& b, StreamOrDevice s = {});
inline array operator>(const array& a, const array& b) {
  return greater(a, b);
}
template <typename T>
array operator>(T a, const array& b) {
  return greater(array(a), b);
}
template <typename T>
array operator>(const array& a, T b) {
  return greater(a, array(b));
}

/** Returns bool array with (a >= b) element-wise. */
array greater_equal(const array& a, const array& b, StreamOrDevice s = {});
inline array operator>=(const array& a, const array& b) {
  return greater_equal(a, b);
}
template <typename T>
array operator>=(T a, const array& b) {
  return greater_equal(array(a), b);
}
template <typename T>
array operator>=(const array& a, T b) {
  return greater_equal(a, array(b));
}

/** Returns bool array with (a < b) element-wise. */
array less(const array& a, const array& b, StreamOrDevice s = {});
inline array operator<(const array& a, const array& b) {
  return less(a, b);
}
template <typename T>
array operator<(T a, const array& b) {
  return less(array(a), b);
}
template <typename T>
array operator<(const array& a, T b) {
  return less(a, array(b));
}

/** Returns bool array with (a <= b) element-wise. */
array less_equal(const array& a, const array& b, StreamOrDevice s = {});
inline array operator<=(const array& a, const array& b) {
  return less_equal(a, b);
}
template <typename T>
array operator<=(T a, const array& b) {
  return less_equal(array(a), b);
}
template <typename T>
array operator<=(const array& a, T b) {
  return less_equal(a, array(b));
}

/** True if two arrays have the same shape and elements. */
array array_equal(
    const array& a,
    const array& b,
    bool equal_nan,
    StreamOrDevice s = {});
inline array
array_equal(const array& a, const array& b, StreamOrDevice s = {}) {
  return array_equal(a, b, false, s);
}

array isnan(const array& a, StreamOrDevice s = {});

array isinf(const array& a, StreamOrDevice s = {});

array isposinf(const array& a, StreamOrDevice s = {});

array isneginf(const array& a, StreamOrDevice s = {});

/** Select from x or y depending on condition. */
array where(
    const array& condition,
    const array& x,
    const array& y,
    StreamOrDevice s = {});

/** Reduction operations */

/** True if all elements in the array are true (or non-zero). **/
array all(const array& a, bool keepdims, StreamOrDevice s = {});
inline array all(const array& a, StreamOrDevice s = {}) {
  return all(a, false, to_stream(s));
}

/** True if the two arrays are equal within the specified tolerance. */
array allclose(
    const array& a,
    const array& b,
    double rtol = 1e-5,
    double atol = 1e-8,
    bool equal_nan = false,
    StreamOrDevice s = {});

/** Returns a boolean array where two arrays are element-wise equal within the
 * specified tolerance. */
array isclose(
    const array& a,
    const array& b,
    double rtol = 1e-5,
    double atol = 1e-8,
    bool equal_nan = false,
    StreamOrDevice s = {});

/**
 *  Reduces the input along the given axes. An output value is true
 *  if all the corresponding inputs are true.
 **/
array all(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/**
 *  Reduces the input along the given axis. An output value is true
 *  if all the corresponding inputs are true.
 **/
array all(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

/** True if any elements in the array are true (or non-zero). **/
array any(const array& a, bool keepdims, StreamOrDevice s = {});
inline array any(const array& a, StreamOrDevice s = {}) {
  return any(a, false, to_stream(s));
}

/**
 *  Reduces the input along the given axes. An output value is true
 *  if any of the corresponding inputs are true.
 **/
array any(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/**
 *  Reduces the input along the given axis. An output value is true
 *  if any of the corresponding inputs are true.
 **/
array any(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

/** Sums the elements of an array. */
array sum(const array& a, bool keepdims, StreamOrDevice s = {});
inline array sum(const array& a, StreamOrDevice s = {}) {
  return sum(a, false, to_stream(s));
}

/** Sums the elements of an array along the given axes. */
array sum(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/** Sums the elements of an array along the given axis. */
array sum(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

/** Computes the mean of the elements of an array. */
array mean(const array& a, bool keepdims, StreamOrDevice s = {});
inline array mean(const array& a, StreamOrDevice s = {}) {
  return mean(a, false, to_stream(s));
}

/** Computes the mean of the elements of an array along the given axes */
array mean(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/** Computes the mean of the elements of an array along the given axis */
array mean(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

/** Computes the mean of the elements of an array. */
array var(const array& a, bool keepdims, int ddof = 0, StreamOrDevice s = {});
inline array var(const array& a, StreamOrDevice s = {}) {
  return var(a, false, 0, to_stream(s));
}

/** Computes the var of the elements of an array along the given axes */
array var(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    int ddof = 0,
    StreamOrDevice s = {});

/** Computes the var of the elements of an array along the given axis */
array var(
    const array& a,
    int axis,
    bool keepdims = false,
    int ddof = 0,
    StreamOrDevice s = {});

/** The product of all elements of the array. */
array prod(const array& a, bool keepdims, StreamOrDevice s = {});
inline array prod(const array& a, StreamOrDevice s = {}) {
  return prod(a, false, to_stream(s));
}

/** The product of the elements of an array along the given axes. */
array prod(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/** The product of the elements of an array along the given axis. */
array prod(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

/** The maximum of all elements of the array. */
array max(const array& a, bool keepdims, StreamOrDevice s = {});
inline array max(const array& a, StreamOrDevice s = {}) {
  return max(a, false, to_stream(s));
}

/** The maximum of the elements of an array along the given axes. */
array max(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/** The maximum of the elements of an array along the given axis. */
array max(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

/** The minimum of all elements of the array. */
array min(const array& a, bool keepdims, StreamOrDevice s = {});
inline array min(const array& a, StreamOrDevice s = {}) {
  return min(a, false, to_stream(s));
}

/** The minimum of the elements of an array along the given axes. */
array min(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/** The minimum of the elements of an array along the given axis. */
array min(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

/** Returns the index of the minimum value in the array. */
array argmin(const array& a, bool keepdims, StreamOrDevice s = {});
inline array argmin(const array& a, StreamOrDevice s = {}) {
  return argmin(a, false, s);
}

/** Returns the indices of the minimum values along a given axis. */
array argmin(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

/** Returns the index of the maximum value in the array. */
array argmax(const array& a, bool keepdims, StreamOrDevice s = {});
inline array argmax(const array& a, StreamOrDevice s = {}) {
  return argmax(a, false, s);
}

/** Returns the indices of the maximum values along a given axis. */
array argmax(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

/** Returns a sorted copy of the flattened array. */
array sort(const array& a, StreamOrDevice s = {});

/** Returns a sorted copy of the array along a given axis. */
array sort(const array& a, int axis, StreamOrDevice s = {});

/** Returns indices that sort the flattened array. */
array argsort(const array& a, StreamOrDevice s = {});

/** Returns indices that sort the array along a given axis. */
array argsort(const array& a, int axis, StreamOrDevice s = {});

/**
 * Returns a partitioned copy of the flattened array
 * such that the smaller kth elements are first.
 **/
array partition(const array& a, int kth, StreamOrDevice s = {});

/**
 * Returns a partitioned copy of the array along a given axis
 * such that the smaller kth elements are first.
 **/
array partition(const array& a, int kth, int axis, StreamOrDevice s = {});

/**
 * Returns indices that partition the flattened array
 * such that the smaller kth elements are first.
 **/
array argpartition(const array& a, int kth, StreamOrDevice s = {});

/**
 * Returns indices that partition the array along a given axis
 * such that the smaller kth elements are first.
 **/
array argpartition(const array& a, int kth, int axis, StreamOrDevice s = {});

/** Returns topk elements of the flattened array. */
array topk(const array& a, int k, StreamOrDevice s = {});

/** Returns topk elements of the array along a given axis. */
array topk(const array& a, int k, int axis, StreamOrDevice s = {});

/** The logsumexp of all elements of the array. */
array logsumexp(const array& a, bool keepdims, StreamOrDevice s = {});
inline array logsumexp(const array& a, StreamOrDevice s = {}) {
  return logsumexp(a, false, to_stream(s));
}

/** The logsumexp of the elements of an array along the given axes. */
array logsumexp(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/** The logsumexp of the elements of an array along the given axis. */
array logsumexp(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

/** Simple arithmetic operations */

/** Absolute value of elements in an array. */
array abs(const array& a, StreamOrDevice s = {});

/** Negate an array. */
array negative(const array& a, StreamOrDevice s = {});
array operator-(const array& a);

/** The sign of the elements in an array. */
array sign(const array& a, StreamOrDevice s = {});

/** Logical not of an array */
array logical_not(const array& a, StreamOrDevice s = {});

/** Logical and of two arrays */
array logical_and(const array& a, const array& b, StreamOrDevice s = {});
array operator&&(const array& a, const array& b);

/** Logical or of two arrays */
array logical_or(const array& a, const array& b, StreamOrDevice s = {});
array operator||(const array& a, const array& b);

/** The reciprocal (1/x) of the elements in an array. */
array reciprocal(const array& a, StreamOrDevice s = {});

/** Add two arrays. */
array add(const array& a, const array& b, StreamOrDevice s = {});
array operator+(const array& a, const array& b);
template <typename T>
array operator+(T a, const array& b) {
  return add(array(a), b);
}
template <typename T>
array operator+(const array& a, T b) {
  return add(a, array(b));
}

/** Subtract two arrays. */
array subtract(const array& a, const array& b, StreamOrDevice s = {});
array operator-(const array& a, const array& b);
template <typename T>
array operator-(T a, const array& b) {
  return subtract(array(a), b);
}
template <typename T>
array operator-(const array& a, T b) {
  return subtract(a, array(b));
}

/** Multiply two arrays. */
array multiply(const array& a, const array& b, StreamOrDevice s = {});
array operator*(const array& a, const array& b);
template <typename T>
array operator*(T a, const array& b) {
  return multiply(array(a), b);
}
template <typename T>
array operator*(const array& a, T b) {
  return multiply(a, array(b));
}

/** Divide two arrays. */
array divide(const array& a, const array& b, StreamOrDevice s = {});
array operator/(const array& a, const array& b);
array operator/(double a, const array& b);
array operator/(const array& a, double b);

/** Compute the element-wise quotient and remainder. */
std::vector<array>
divmod(const array& a, const array& b, StreamOrDevice s = {});

/** Compute integer division. Equivalent to doing floor(a / x). */
array floor_divide(const array& a, const array& b, StreamOrDevice s = {});

/** Compute the element-wise remainder of division */
array remainder(const array& a, const array& b, StreamOrDevice s = {});
array operator%(const array& a, const array& b);
template <typename T>
array operator%(T a, const array& b) {
  return remainder(array(a), b);
}
template <typename T>
array operator%(const array& a, T b) {
  return remainder(a, array(b));
}

/** Element-wise maximum between two arrays. */
array maximum(const array& a, const array& b, StreamOrDevice s = {});

/** Element-wise minimum between two arrays. */
array minimum(const array& a, const array& b, StreamOrDevice s = {});

/** Floor the element of an array. **/
array floor(const array& a, StreamOrDevice s = {});

/** Ceil the element of an array. **/
array ceil(const array& a, StreamOrDevice s = {});

/** Square the elements of an array. */
array square(const array& a, StreamOrDevice s = {});

/** Exponential of the elements of an array. */
array exp(const array& a, StreamOrDevice s = {});

/** Sine of the elements of an array */
array sin(const array& a, StreamOrDevice s = {});

/** Cosine of the elements of an array */
array cos(const array& a, StreamOrDevice s = {});

/** Tangent of the elements of an array */
array tan(const array& a, StreamOrDevice s = {});

/** Arc Sine of the elements of an array */
array arcsin(const array& a, StreamOrDevice s = {});

/** Arc Cosine of the elements of an array */
array arccos(const array& a, StreamOrDevice s = {});

/** Arc Tangent of the elements of an array */
array arctan(const array& a, StreamOrDevice s = {});

/** Hyperbolic Sine of the elements of an array */
array sinh(const array& a, StreamOrDevice s = {});

/** Hyperbolic Cosine of the elements of an array */
array cosh(const array& a, StreamOrDevice s = {});

/** Hyperbolic Tangent of the elements of an array */
array tanh(const array& a, StreamOrDevice s = {});

/** Inverse Hyperbolic Sine of the elements of an array */
array arcsinh(const array& a, StreamOrDevice s = {});

/** Inverse Hyperbolic Cosine of the elements of an array */
array arccosh(const array& a, StreamOrDevice s = {});

/** Inverse Hyperbolic Tangent of the elements of an array */
array arctanh(const array& a, StreamOrDevice s = {});

/** Natural logarithm of the elements of an array. */
array log(const array& a, StreamOrDevice s = {});

/** Log base 2 of the elements of an array. */
array log2(const array& a, StreamOrDevice s = {});

/** Log base 10 of the elements of an array. */
array log10(const array& a, StreamOrDevice s = {});

/** Natural logarithm of one plus elements in the array: `log(1 + a)`. */
array log1p(const array& a, StreamOrDevice s = {});

/** Log-add-exp of one elements in the array: `log(exp(a) + exp(b))`. */
array logaddexp(const array& a, const array& b, StreamOrDevice s = {});

/** Element-wise logistic sigmoid of the array: `1 / (1 + exp(-x)`. */
array sigmoid(const array& a, StreamOrDevice s = {});

/** Computes the error function of the elements of an array. */
array erf(const array& a, StreamOrDevice s = {});

/** Computes the inverse error function of the elements of an array. */
array erfinv(const array& a, StreamOrDevice s = {});

/** Stop the flow of gradients. */
array stop_gradient(const array& a, StreamOrDevice s = {});

/** Round a floating point number */
array round(const array& a, int decimals, StreamOrDevice s = {});
inline array round(const array& a, StreamOrDevice s = {}) {
  return round(a, 0, s);
}

/** Matrix-matrix multiplication. */
array matmul(const array& a, const array& b, StreamOrDevice s = {});

/** Gather array entries given indices and slices */
array gather(
    const array& a,
    const std::vector<array>& indices,
    const std::vector<int>& axes,
    const std::vector<int>& slice_sizes,
    StreamOrDevice s = {});
inline array gather(
    const array& a,
    const array& indices,
    int axis,
    const std::vector<int>& slice_sizes,
    StreamOrDevice s = {}) {
  return gather(a, {indices}, std::vector<int>{axis}, slice_sizes, s);
}

/** Take array slices at the given indices of the specified axis. */
array take(
    const array& a,
    const array& indices,
    int axis,
    StreamOrDevice s = {});

/** Take array entries at the given indices treating the array as flattened. */
array take(const array& a, const array& indices, StreamOrDevice s = {});

/** Take array entries given indices along the axis */
array take_along_axis(
    const array& a,
    const array& indices,
    int axis,
    StreamOrDevice s = {});

/** Scatter updates to given linear indices */
array scatter(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
inline array scatter(
    const array& a,
    const array& indices,
    const array& updates,
    int axis,
    StreamOrDevice s = {}) {
  return scatter(a, {indices}, updates, std::vector<int>{axis}, s);
}

/** Scatter and add updates to given indices */
array scatter_add(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
inline array scatter_add(
    const array& a,
    const array& indices,
    const array& updates,
    int axis,
    StreamOrDevice s = {}) {
  return scatter_add(a, {indices}, updates, std::vector<int>{axis}, s);
}

/** Scatter and prod updates to given indices */
array scatter_prod(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
inline array scatter_prod(
    const array& a,
    const array& indices,
    const array& updates,
    int axis,
    StreamOrDevice s = {}) {
  return scatter_prod(a, {indices}, updates, std::vector<int>{axis}, s);
}

/** Scatter and max updates to given linear indices */
array scatter_max(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
inline array scatter_max(
    const array& a,
    const array& indices,
    const array& updates,
    int axis,
    StreamOrDevice s = {}) {
  return scatter_max(a, {indices}, updates, std::vector<int>{axis}, s);
}
/** Scatter and min updates to given linear indices */
array scatter_min(
    const array& a,
    const std::vector<array>& indices,
    const array& updates,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
inline array scatter_min(
    const array& a,
    const array& indices,
    const array& updates,
    int axis,
    StreamOrDevice s = {}) {
  return scatter_min(a, {indices}, updates, std::vector<int>{axis}, s);
}

/** Square root the elements of an array. */
array sqrt(const array& a, StreamOrDevice s = {});

/** Square root and reciprocal the elements of an array. */
array rsqrt(const array& a, StreamOrDevice s = {});

/** Softmax of an array. */
array softmax(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s = {});

/** Softmax of an array. */
array softmax(const array& a, StreamOrDevice s = {});

/** Softmax of an array. */
inline array softmax(const array& a, int axis, StreamOrDevice s = {}) {
  return softmax(a, std::vector<int>{axis}, s);
}

/** Raise elements of a to the power of b element-wise */
array power(const array& a, const array& b, StreamOrDevice s = {});
inline array operator^(const array& a, const array& b) {
  return power(a, b);
}
template <typename T>
array operator^(T a, const array& b) {
  return power(array(a), b);
}
template <typename T>
array operator^(const array& a, T b) {
  return power(a, array(b));
}

/** Cumulative sum of an array. */
array cumsum(
    const array& a,
    int axis,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

/** Cumulative product of an array. */
array cumprod(
    const array& a,
    int axis,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

/** Cumulative max of an array. */
array cummax(
    const array& a,
    int axis,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

/** Cumulative min of an array. */
array cummin(
    const array& a,
    int axis,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

/** Convolution operations */

/** General convolution with a filter */
array conv_general(
    array input,
    array weight,
    std::vector<int> stride = {},
    std::vector<int> padding_lo = {},
    std::vector<int> padding_hi = {},
    std::vector<int> kernel_dilation = {},
    std::vector<int> input_dilation = {},
    int groups = 1,
    bool flip = false,
    StreamOrDevice s = {});

/** General convolution with a filter */
inline array conv_general(
    const array& input,
    const array& weight,
    std::vector<int> stride = {},
    std::vector<int> padding = {},
    std::vector<int> kernel_dilation = {},
    std::vector<int> input_dilation = {},
    int groups = 1,
    bool flip = false,
    StreamOrDevice s = {}) {
  return conv_general(
      /* const array& input = */ input,
      /* const array& weight = */ weight,
      /* std::vector<int> stride = */ stride,
      /* std::vector<int> padding_lo = */ padding,
      /* std::vector<int> padding_hi = */ padding,
      /* std::vector<int> kernel_dilation = */ kernel_dilation,
      /* std::vector<int> input_dilation = */ input_dilation,
      /* int groups = */ groups,
      /* bool flip = */ flip,
      /* StreamOrDevice s = */ s);
}

/** 1D convolution with a filter */
array conv1d(
    const array& input,
    const array& weight,
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int groups = 1,
    StreamOrDevice s = {});

/** 2D convolution with a filter */
array conv2d(
    const array& input,
    const array& weight,
    const std::pair<int, int>& stride = {1, 1},
    const std::pair<int, int>& padding = {0, 0},
    const std::pair<int, int>& dilation = {1, 1},
    int groups = 1,
    StreamOrDevice s = {});

/** Quantized matmul multiplies x with a quantized matrix w*/
array quantized_matmul(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    bool transpose = true,
    int group_size = 64,
    int bits = 4,
    StreamOrDevice s = {});

/** Quantize a matrix along its last axis */
std::tuple<array, array, array> quantize(
    const array& w,
    int group_size = 64,
    int bits = 4,
    StreamOrDevice s = {});

/** Dequantize a matrix produced by quantize() */
array dequantize(
    const array& w,
    const array& scales,
    const array& biases,
    int group_size = 64,
    int bits = 4,
    StreamOrDevice s = {});

/** TensorDot returns a contraction of a and b over multiple dimensions. */
array tensordot(
    const array& a,
    const array& b,
    const int dims = 2,
    StreamOrDevice s = {});

array tensordot(
    const array& a,
    const array& b,
    const std::pair<std::vector<int>, std::vector<int>>& dims,
    StreamOrDevice s = {});

/** Compute the outer product of two vectors. */
array outer(const array& a, const array& b, StreamOrDevice s = {});

/** Compute the inner product of two vectors. */
array inner(const array& a, const array& b, StreamOrDevice s = {});

/** Compute D = beta * C + alpha * (A @ B) */
array addmm(
    array c,
    array a,
    array b,
    const float& alpha = 1.f,
    const float& beta = 1.f,
    StreamOrDevice s = {});

/** Extract a diagonal or construct a diagonal array */
array diagonal(
    const array& a,
    int offset = 0,
    int axis1 = 0,
    int axis2 = 1,
    StreamOrDevice s = {});

/** Extract diagonal from a 2d array or create a diagonal matrix. */
array diag(const array& a, int k = 0, StreamOrDevice s = {});

/**
 * Implements the identity function but allows injecting dependencies to other
 * arrays. This ensures that these other arrays will have been computed
 * when the outputs of this function are computed.
 */
std::vector<array> depends(
    const std::vector<array>& inputs,
    const std::vector<array>& dependencies);

/** convert an array to an atleast ndim array */
array atleast_1d(const array& a, StreamOrDevice s = {});
std::vector<array> atleast_1d(
    const std::vector<array>& a,
    StreamOrDevice s = {});
array atleast_2d(const array& a, StreamOrDevice s = {});
std::vector<array> atleast_2d(
    const std::vector<array>& a,
    StreamOrDevice s = {});
array atleast_3d(const array& a, StreamOrDevice s = {});
std::vector<array> atleast_3d(
    const std::vector<array>& a,
    StreamOrDevice s = {});

} // namespace mlx::core
