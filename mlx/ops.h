// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx::core {

/**
 * \defgroup ops Core array operations
 * @{
 */

/**
 * A 1D array of numbers starting at `start` (optional),
 * stopping at stop, stepping by `step` (optional). */
MLX_API array arange(
    double start,
    double stop,
    double step,
    Dtype dtype,
    StreamOrDevice s = {});
MLX_API array
arange(double start, double stop, double step, StreamOrDevice s = {});
MLX_API array
arange(double start, double stop, Dtype dtype, StreamOrDevice s = {});
MLX_API array arange(double start, double stop, StreamOrDevice s = {});
MLX_API array arange(double stop, Dtype dtype, StreamOrDevice s = {});
MLX_API array arange(double stop, StreamOrDevice s = {});

MLX_API array arange(int start, int stop, int step, StreamOrDevice s = {});
MLX_API array arange(int start, int stop, StreamOrDevice s = {});
MLX_API array arange(int stop, StreamOrDevice s = {});

/** A 1D array of `num` evenly spaced numbers in the range `[start, stop]` */
MLX_API array linspace(
    double start,
    double stop,
    int num = 50,
    Dtype dtype = float32,
    StreamOrDevice s = {});

/** Convert an array to the given data type. */
MLX_API array astype(array a, Dtype dtype, StreamOrDevice s = {});

/** Create a view of an array with the given shape and strides. */
MLX_API array as_strided(
    array a,
    Shape shape,
    Strides strides,
    size_t offset,
    StreamOrDevice s = {});

/** Copy another array. */
MLX_API array copy(array a, StreamOrDevice s = {});

/** Fill an array of the given shape with the given value(s). */
MLX_API array full(Shape shape, array vals, Dtype dtype, StreamOrDevice s = {});
MLX_API array full(Shape shape, array vals, StreamOrDevice s = {});
template <typename T>
array full(Shape shape, T val, Dtype dtype, StreamOrDevice s = {}) {
  return full(std::move(shape), array(val, dtype), to_stream(s));
}
template <typename T>
array full(Shape shape, T val, StreamOrDevice s = {}) {
  return full(std::move(shape), array(val), to_stream(s));
}

MLX_API array
full_like(const array& a, array vals, Dtype dtype, StreamOrDevice s = {});
MLX_API array full_like(const array& a, array vals, StreamOrDevice s = {});
template <typename T>
array full_like(const array& a, T val, Dtype dtype, StreamOrDevice s = {}) {
  return full_like(a, array(val, dtype), dtype, to_stream(s));
}
template <typename T>
array full_like(const array& a, T val, StreamOrDevice s = {}) {
  return full_like(a, array(val, a.dtype()), to_stream(s));
}

/** Fill an array of the given shape with zeros. */
MLX_API array zeros(const Shape& shape, Dtype dtype, StreamOrDevice s = {});
inline array zeros(const Shape& shape, StreamOrDevice s = {}) {
  return zeros(shape, float32, s);
}
MLX_API array zeros_like(const array& a, StreamOrDevice s = {});

/** Fill an array of the given shape with ones. */
MLX_API array ones(const Shape& shape, Dtype dtype, StreamOrDevice s = {});
inline array ones(const Shape& shape, StreamOrDevice s = {}) {
  return ones(shape, float32, s);
}
MLX_API array ones_like(const array& a, StreamOrDevice s = {});

/** Fill an array of the given shape (n,m) with ones in the specified diagonal
 * k, and zeros everywhere else. */
MLX_API array eye(int n, int m, int k, Dtype dtype, StreamOrDevice s = {});
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
MLX_API array identity(int n, Dtype dtype, StreamOrDevice s = {});
inline array identity(int n, StreamOrDevice s = {}) {
  return identity(n, float32, s);
}

MLX_API array tri(int n, int m, int k, Dtype type, StreamOrDevice s = {});
inline array tri(int n, Dtype type, StreamOrDevice s = {}) {
  return tri(n, n, 0, type, s);
}

MLX_API array tril(array x, int k = 0, StreamOrDevice s = {});
MLX_API array triu(array x, int k = 0, StreamOrDevice s = {});

/** Reshape an array to the given shape. */
MLX_API array reshape(const array& a, Shape shape, StreamOrDevice s = {});

/** Unflatten the axis to the given shape. */
MLX_API array
unflatten(const array& a, int axis, Shape shape, StreamOrDevice s = {});

/** Flatten the dimensions in the range `[start_axis, end_axis]` . */
MLX_API array flatten(
    const array& a,
    int start_axis,
    int end_axis = -1,
    StreamOrDevice s = {});

/** Flatten the array to 1D. */
MLX_API array flatten(const array& a, StreamOrDevice s = {});

/** Multiply the array by the Hadamard matrix of corresponding size. */
MLX_API array hadamard_transform(
    const array& a,
    std::optional<float> scale = std::nullopt,
    StreamOrDevice s = {});

/** Remove singleton dimensions at the given axes. */
MLX_API array
squeeze(const array& a, const std::vector<int>& axes, StreamOrDevice s = {});

/** Remove singleton dimensions at the given axis. */
MLX_API array squeeze(const array& a, int axis, StreamOrDevice s = {});

/** Remove all singleton dimensions. */
MLX_API array squeeze(const array& a, StreamOrDevice s = {});

/** Add a singleton dimension at the given axes. */
MLX_API array expand_dims(
    const array& a,
    const std::vector<int>& axes,
    StreamOrDevice s = {});

/** Add a singleton dimension at the given axis. */
MLX_API array expand_dims(const array& a, int axis, StreamOrDevice s = {});

/** Slice an array. */
MLX_API array slice(
    const array& a,
    Shape start,
    Shape stop,
    Shape strides,
    StreamOrDevice s = {});
inline array slice(
    const array& a,
    std::initializer_list<int> start,
    Shape stop,
    Shape strides,
    StreamOrDevice s = {}) {
  return slice(a, Shape(start), std::move(stop), std::move(strides), s);
}

/** Slice an array with a stride of 1 in each dimension. */
MLX_API array
slice(const array& a, Shape start, Shape stop, StreamOrDevice s = {});

/** Slice an array with dynamic starting indices. */
MLX_API array slice(
    const array& a,
    const array& start,
    std::vector<int> axes,
    Shape slice_size,
    StreamOrDevice s = {});

/** Update a slice from the source array. */
MLX_API array slice_update(
    const array& src,
    const array& update,
    Shape start,
    Shape stop,
    Shape strides,
    StreamOrDevice s = {});

/** Update a slice from the source array with stride 1 in each dimension. */
MLX_API array slice_update(
    const array& src,
    const array& update,
    Shape start,
    Shape stop,
    StreamOrDevice s = {});

/** Update a slice from the source array with dynamic starting indices. */
MLX_API array slice_update(
    const array& src,
    const array& update,
    const array& start,
    std::vector<int> axes,
    StreamOrDevice s = {});

/** Split an array into sub-arrays along a given axis. */
MLX_API std::vector<array>
split(const array& a, int num_splits, int axis, StreamOrDevice s = {});
MLX_API std::vector<array>
split(const array& a, int num_splits, StreamOrDevice s = {});
MLX_API std::vector<array>
split(const array& a, const Shape& indices, int axis, StreamOrDevice s = {});
MLX_API std::vector<array>
split(const array& a, const Shape& indices, StreamOrDevice s = {});

/** A vector of coordinate arrays from coordinate vectors. */
MLX_API std::vector<array> meshgrid(
    const std::vector<array>& arrays,
    bool sparse = false,
    const std::string& indexing = "xy",
    StreamOrDevice s = {});

/**
 * Clip (limit) the values in an array.
 */
MLX_API array clip(
    const array& a,
    const std::optional<array>& a_min = std::nullopt,
    const std::optional<array>& a_max = std::nullopt,
    StreamOrDevice s = {});

/** Concatenate arrays along a given axis. */
MLX_API array
concatenate(std::vector<array> arrays, int axis, StreamOrDevice s = {});
MLX_API array concatenate(std::vector<array> arrays, StreamOrDevice s = {});

/** Stack arrays along a new axis. */
MLX_API array
stack(const std::vector<array>& arrays, int axis, StreamOrDevice s = {});
MLX_API array stack(const std::vector<array>& arrays, StreamOrDevice s = {});

/** Repeat an array along an axis. */
MLX_API array
repeat(const array& arr, int repeats, int axis, StreamOrDevice s = {});
MLX_API array repeat(const array& arr, int repeats, StreamOrDevice s = {});

MLX_API array
tile(const array& arr, std::vector<int> reps, StreamOrDevice s = {});

/** Permutes the dimensions according to the given axes. */
MLX_API array
transpose(const array& a, std::vector<int> axes, StreamOrDevice s = {});
inline array transpose(
    const array& a,
    std::initializer_list<int> axes,
    StreamOrDevice s = {}) {
  return transpose(a, std::vector<int>(axes), s);
}

/** Swap two axes of an array. */
MLX_API array
swapaxes(const array& a, int axis1, int axis2, StreamOrDevice s = {});

/** Move an axis of an array. */
MLX_API array
moveaxis(const array& a, int source, int destination, StreamOrDevice s = {});

/** Pad an array with a constant value */
MLX_API array
pad(const array& a,
    const std::vector<int>& axes,
    const Shape& low_pad_size,
    const Shape& high_pad_size,
    const array& pad_value = array(0),
    const std::string& mode = "constant",
    StreamOrDevice s = {});

/** Pad an array with a constant value along all axes */
MLX_API array
pad(const array& a,
    const std::vector<std::pair<int, int>>& pad_width,
    const array& pad_value = array(0),
    const std::string& mode = "constant",
    StreamOrDevice s = {});
MLX_API array
pad(const array& a,
    const std::pair<int, int>& pad_width,
    const array& pad_value = array(0),
    const std::string& mode = "constant",
    StreamOrDevice s = {});
MLX_API array
pad(const array& a,
    int pad_width,
    const array& pad_value = array(0),
    const std::string& mode = "constant",
    StreamOrDevice s = {});

/** Permutes the dimensions in reverse order. */
MLX_API array transpose(const array& a, StreamOrDevice s = {});

/** Broadcast an array to a given shape. */
MLX_API array
broadcast_to(const array& a, const Shape& shape, StreamOrDevice s = {});

/** Broadcast a vector of arrays against one another. */
MLX_API std::vector<array> broadcast_arrays(
    const std::vector<array>& inputs,
    StreamOrDevice s = {});

/** Returns the bool array with (a == b) element-wise. */
MLX_API array equal(const array& a, const array& b, StreamOrDevice s = {});
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
MLX_API array not_equal(const array& a, const array& b, StreamOrDevice s = {});
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
MLX_API array greater(const array& a, const array& b, StreamOrDevice s = {});
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
MLX_API array
greater_equal(const array& a, const array& b, StreamOrDevice s = {});
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
MLX_API array less(const array& a, const array& b, StreamOrDevice s = {});
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
MLX_API array less_equal(const array& a, const array& b, StreamOrDevice s = {});
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
MLX_API array array_equal(
    const array& a,
    const array& b,
    bool equal_nan,
    StreamOrDevice s = {});
inline array
array_equal(const array& a, const array& b, StreamOrDevice s = {}) {
  return array_equal(a, b, false, s);
}

MLX_API array isnan(const array& a, StreamOrDevice s = {});

MLX_API array isinf(const array& a, StreamOrDevice s = {});

MLX_API array isfinite(const array& a, StreamOrDevice s = {});

MLX_API array isposinf(const array& a, StreamOrDevice s = {});

MLX_API array isneginf(const array& a, StreamOrDevice s = {});

/** Select from x or y depending on condition. */
MLX_API array where(
    const array& condition,
    const array& x,
    const array& y,
    StreamOrDevice s = {});

/** Replace NaN and infinities with finite numbers. */
MLX_API array nan_to_num(
    const array& a,
    float nan = 0.0f,
    const std::optional<float> posinf = std::nullopt,
    const std::optional<float> neginf = std::nullopt,
    StreamOrDevice s = {});

/** True if all elements in the array are true (or non-zero). **/
MLX_API array all(const array& a, bool keepdims, StreamOrDevice s = {});
inline array all(const array& a, StreamOrDevice s = {}) {
  return all(a, false, to_stream(s));
}

/** True if the two arrays are equal within the specified tolerance. */
MLX_API array allclose(
    const array& a,
    const array& b,
    double rtol = 1e-5,
    double atol = 1e-8,
    bool equal_nan = false,
    StreamOrDevice s = {});

/** Returns a boolean array where two arrays are element-wise equal within the
 * specified tolerance. */
MLX_API array isclose(
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
MLX_API array
all(const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/**
 *  Reduces the input along the given axis. An output value is true
 *  if all the corresponding inputs are true.
 **/
MLX_API array
all(const array& a, int axis, bool keepdims = false, StreamOrDevice s = {});

/** True if any elements in the array are true (or non-zero). **/
MLX_API array any(const array& a, bool keepdims, StreamOrDevice s = {});
inline array any(const array& a, StreamOrDevice s = {}) {
  return any(a, false, to_stream(s));
}

/**
 *  Reduces the input along the given axes. An output value is true
 *  if any of the corresponding inputs are true.
 **/
MLX_API array
any(const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/**
 *  Reduces the input along the given axis. An output value is true
 *  if any of the corresponding inputs are true.
 **/
MLX_API array
any(const array& a, int axis, bool keepdims = false, StreamOrDevice s = {});

/** Sums the elements of an array. */
MLX_API array sum(const array& a, bool keepdims, StreamOrDevice s = {});
inline array sum(const array& a, StreamOrDevice s = {}) {
  return sum(a, false, to_stream(s));
}

/** Sums the elements of an array along the given axes. */
MLX_API array
sum(const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/** Sums the elements of an array along the given axis. */
MLX_API array
sum(const array& a, int axis, bool keepdims = false, StreamOrDevice s = {});

/** Computes the mean of the elements of an array. */
MLX_API array mean(const array& a, bool keepdims, StreamOrDevice s = {});
inline array mean(const array& a, StreamOrDevice s = {}) {
  return mean(a, false, to_stream(s));
}

/** Computes the mean of the elements of an array along the given axes */
MLX_API array mean(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/** Computes the mean of the elements of an array along the given axis */
MLX_API array
mean(const array& a, int axis, bool keepdims = false, StreamOrDevice s = {});

/** Computes the median of the elements of an array. */
MLX_API array median(const array& a, bool keepdims, StreamOrDevice s = {});
inline array median(const array& a, StreamOrDevice s = {}) {
  return median(a, false, to_stream(s));
}

/** Computes the median of the elements of an array along the given axes */
MLX_API array median(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/** Computes the median of the elements of an array along the given axis */
MLX_API array
median(const array& a, int axis, bool keepdims = false, StreamOrDevice s = {});

/** Computes the variance of the elements of an array. */
MLX_API array
var(const array& a, bool keepdims, int ddof = 0, StreamOrDevice s = {});
inline array var(const array& a, StreamOrDevice s = {}) {
  return var(a, false, 0, to_stream(s));
}

/** Computes the variance of the elements of an array along the given
 * axes */
MLX_API array
var(const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    int ddof = 0,
    StreamOrDevice s = {});

/** Computes the variance of the elements of an array along the given
 * axis */
MLX_API array
var(const array& a,
    int axis,
    bool keepdims = false,
    int ddof = 0,
    StreamOrDevice s = {});

/** Computes the standard deviation of the elements of an array. */
MLX_API array
std(const array& a, bool keepdims, int ddof = 0, StreamOrDevice s = {});
inline array std(const array& a, StreamOrDevice s = {}) {
  return std(a, false, 0, to_stream(s));
}

/** Computes the standard deviation of the elements of an array along the given
 * axes */
MLX_API array
std(const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    int ddof = 0,
    StreamOrDevice s = {});

/** Computes the standard deviation of the elements of an array along the given
 * axis */
MLX_API array
std(const array& a,
    int axis,
    bool keepdims = false,
    int ddof = 0,
    StreamOrDevice s = {});

/** The product of all elements of the array. */
MLX_API array prod(const array& a, bool keepdims, StreamOrDevice s = {});
inline array prod(const array& a, StreamOrDevice s = {}) {
  return prod(a, false, to_stream(s));
}

/** The product of the elements of an array along the given axes. */
MLX_API array prod(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/** The product of the elements of an array along the given axis. */
MLX_API array
prod(const array& a, int axis, bool keepdims = false, StreamOrDevice s = {});

/** The maximum of all elements of the array. */
MLX_API array max(const array& a, bool keepdims, StreamOrDevice s = {});
inline array max(const array& a, StreamOrDevice s = {}) {
  return max(a, false, to_stream(s));
}

/** The maximum of the elements of an array along the given axes. */
MLX_API array
max(const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/** The maximum of the elements of an array along the given axis. */
MLX_API array
max(const array& a, int axis, bool keepdims = false, StreamOrDevice s = {});

/** The minimum of all elements of the array. */
MLX_API array min(const array& a, bool keepdims, StreamOrDevice s = {});
inline array min(const array& a, StreamOrDevice s = {}) {
  return min(a, false, to_stream(s));
}

/** The minimum of the elements of an array along the given axes. */
MLX_API array
min(const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/** The minimum of the elements of an array along the given axis. */
MLX_API array
min(const array& a, int axis, bool keepdims = false, StreamOrDevice s = {});

/** Returns the Hanning window of size M. */
MLX_API array hanning(int M, StreamOrDevice s = {});

/** Returns the Hamming window of size M. */
MLX_API array hamming(int M, StreamOrDevice s = {});

/** Returns the bartlett window of size M. */
MLX_API array bartlett(int M, StreamOrDevice s = {});

/** Returns the Blackmann window of size M. */
MLX_API array blackman(int M, StreamOrDevice s = {});

/** Returns the index of the minimum value in the array. */
MLX_API array argmin(const array& a, bool keepdims, StreamOrDevice s = {});
inline array argmin(const array& a, StreamOrDevice s = {}) {
  return argmin(a, false, s);
}

/** Returns the indices of the minimum values along a given axis. */
MLX_API array
argmin(const array& a, int axis, bool keepdims = false, StreamOrDevice s = {});

/** Returns the index of the maximum value in the array. */
MLX_API array argmax(const array& a, bool keepdims, StreamOrDevice s = {});
inline array argmax(const array& a, StreamOrDevice s = {}) {
  return argmax(a, false, s);
}

/** Returns the indices of the maximum values along a given axis. */
MLX_API array
argmax(const array& a, int axis, bool keepdims = false, StreamOrDevice s = {});

/** Returns a sorted copy of the flattened array. */
MLX_API array sort(const array& a, StreamOrDevice s = {});

/** Returns a sorted copy of the array along a given axis. */
MLX_API array sort(const array& a, int axis, StreamOrDevice s = {});

/** Returns indices that sort the flattened array. */
MLX_API array argsort(const array& a, StreamOrDevice s = {});

/** Returns indices that sort the array along a given axis. */
MLX_API array argsort(const array& a, int axis, StreamOrDevice s = {});

/**
 * Returns a partitioned copy of the flattened array
 * such that the smaller kth elements are first.
 **/
MLX_API array partition(const array& a, int kth, StreamOrDevice s = {});

/**
 * Returns a partitioned copy of the array along a given axis
 * such that the smaller kth elements are first.
 **/
MLX_API array
partition(const array& a, int kth, int axis, StreamOrDevice s = {});

/**
 * Returns indices that partition the flattened array
 * such that the smaller kth elements are first.
 **/
MLX_API array argpartition(const array& a, int kth, StreamOrDevice s = {});

/**
 * Returns indices that partition the array along a given axis
 * such that the smaller kth elements are first.
 **/
MLX_API array
argpartition(const array& a, int kth, int axis, StreamOrDevice s = {});

/** Returns topk elements of the flattened array. */
MLX_API array topk(const array& a, int k, StreamOrDevice s = {});

/** Returns topk elements of the array along a given axis. */
MLX_API array topk(const array& a, int k, int axis, StreamOrDevice s = {});

/** Cumulative logsumexp of an array. */
MLX_API array logcumsumexp(
    const array& a,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

/** Cumulative logsumexp of an array along the given axis. */
MLX_API array logcumsumexp(
    const array& a,
    int axis,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

/** The logsumexp of all elements of the array. */
MLX_API array logsumexp(const array& a, bool keepdims, StreamOrDevice s = {});
inline array logsumexp(const array& a, StreamOrDevice s = {}) {
  return logsumexp(a, false, to_stream(s));
}

/** The logsumexp of the elements of an array along the given axes. */
MLX_API array logsumexp(
    const array& a,
    const std::vector<int>& axes,
    bool keepdims = false,
    StreamOrDevice s = {});

/** The logsumexp of the elements of an array along the given axis. */
MLX_API array logsumexp(
    const array& a,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {});

/** Absolute value of elements in an array. */
MLX_API array abs(const array& a, StreamOrDevice s = {});

/** Negate an array. */
MLX_API array negative(const array& a, StreamOrDevice s = {});
MLX_API array operator-(const array& a);

/** The sign of the elements in an array. */
MLX_API array sign(const array& a, StreamOrDevice s = {});

/** Logical not of an array */
MLX_API array logical_not(const array& a, StreamOrDevice s = {});

/** Logical and of two arrays */
MLX_API array
logical_and(const array& a, const array& b, StreamOrDevice s = {});
MLX_API array operator&&(const array& a, const array& b);

/** Logical or of two arrays */
MLX_API array logical_or(const array& a, const array& b, StreamOrDevice s = {});
MLX_API array operator||(const array& a, const array& b);

/** The reciprocal (1/x) of the elements in an array. */
MLX_API array reciprocal(const array& a, StreamOrDevice s = {});

/** Add two arrays. */
MLX_API array add(const array& a, const array& b, StreamOrDevice s = {});
MLX_API array operator+(const array& a, const array& b);
template <typename T>
array operator+(T a, const array& b) {
  return add(array(a), b);
}
template <typename T>
array operator+(const array& a, T b) {
  return add(a, array(b));
}

/** Subtract two arrays. */
MLX_API array subtract(const array& a, const array& b, StreamOrDevice s = {});
MLX_API array operator-(const array& a, const array& b);
template <typename T>
array operator-(T a, const array& b) {
  return subtract(array(a), b);
}
template <typename T>
array operator-(const array& a, T b) {
  return subtract(a, array(b));
}

/** Multiply two arrays. */
MLX_API array multiply(const array& a, const array& b, StreamOrDevice s = {});
MLX_API array operator*(const array& a, const array& b);
template <typename T>
array operator*(T a, const array& b) {
  return multiply(array(a), b);
}
template <typename T>
array operator*(const array& a, T b) {
  return multiply(a, array(b));
}

/** Divide two arrays. */
MLX_API array divide(const array& a, const array& b, StreamOrDevice s = {});
MLX_API array operator/(const array& a, const array& b);
MLX_API array operator/(double a, const array& b);
MLX_API array operator/(const array& a, double b);

/** Compute the element-wise quotient and remainder. */
MLX_API std::vector<array>
divmod(const array& a, const array& b, StreamOrDevice s = {});

/** Compute integer division. Equivalent to doing floor(a / x). */
MLX_API array
floor_divide(const array& a, const array& b, StreamOrDevice s = {});

/** Compute the element-wise remainder of division */
MLX_API array remainder(const array& a, const array& b, StreamOrDevice s = {});
MLX_API array operator%(const array& a, const array& b);
template <typename T>
array operator%(T a, const array& b) {
  return remainder(array(a), b);
}
template <typename T>
array operator%(const array& a, T b) {
  return remainder(a, array(b));
}

/** Element-wise maximum between two arrays. */
MLX_API array maximum(const array& a, const array& b, StreamOrDevice s = {});

/** Element-wise minimum between two arrays. */
MLX_API array minimum(const array& a, const array& b, StreamOrDevice s = {});

/** Floor the element of an array. **/
MLX_API array floor(const array& a, StreamOrDevice s = {});

/** Ceil the element of an array. **/
MLX_API array ceil(const array& a, StreamOrDevice s = {});

/** Square the elements of an array. */
MLX_API array square(const array& a, StreamOrDevice s = {});

/** Exponential of the elements of an array. */
MLX_API array exp(const array& a, StreamOrDevice s = {});

/** Sine of the elements of an array */
MLX_API array sin(const array& a, StreamOrDevice s = {});

/** Cosine of the elements of an array */
MLX_API array cos(const array& a, StreamOrDevice s = {});

/** Tangent of the elements of an array */
MLX_API array tan(const array& a, StreamOrDevice s = {});

/** Arc Sine of the elements of an array */
MLX_API array arcsin(const array& a, StreamOrDevice s = {});

/** Arc Cosine of the elements of an array */
MLX_API array arccos(const array& a, StreamOrDevice s = {});

/** Arc Tangent of the elements of an array */
MLX_API array arctan(const array& a, StreamOrDevice s = {});

/** Inverse tangent of the ratio of two arrays */
MLX_API array arctan2(const array& a, const array& b, StreamOrDevice s = {});

/** Hyperbolic Sine of the elements of an array */
MLX_API array sinh(const array& a, StreamOrDevice s = {});

/** Hyperbolic Cosine of the elements of an array */
MLX_API array cosh(const array& a, StreamOrDevice s = {});

/** Hyperbolic Tangent of the elements of an array */
MLX_API array tanh(const array& a, StreamOrDevice s = {});

/** Inverse Hyperbolic Sine of the elements of an array */
MLX_API array arcsinh(const array& a, StreamOrDevice s = {});

/** Inverse Hyperbolic Cosine of the elements of an array */
MLX_API array arccosh(const array& a, StreamOrDevice s = {});

/** Inverse Hyperbolic Tangent of the elements of an array */
MLX_API array arctanh(const array& a, StreamOrDevice s = {});

/** Convert the elements of an array from Radians to Degrees **/
MLX_API array degrees(const array& a, StreamOrDevice s = {});

/** Convert the elements of an array from Degrees to Radians **/
MLX_API array radians(const array& a, StreamOrDevice s = {});

/** Natural logarithm of the elements of an array. */
MLX_API array log(const array& a, StreamOrDevice s = {});

/** Log base 2 of the elements of an array. */
MLX_API array log2(const array& a, StreamOrDevice s = {});

/** Log base 10 of the elements of an array. */
MLX_API array log10(const array& a, StreamOrDevice s = {});

/** Natural logarithm of one plus elements in the array: `log(1 + a)`. */
MLX_API array log1p(const array& a, StreamOrDevice s = {});

/** Log-add-exp of one elements in the array: `log(exp(a) + exp(b))`. */
MLX_API array logaddexp(const array& a, const array& b, StreamOrDevice s = {});

/** Element-wise logistic sigmoid of the array: `1 / (1 + exp(-x)`. */
MLX_API array sigmoid(const array& a, StreamOrDevice s = {});

/** Computes the error function of the elements of an array. */
MLX_API array erf(const array& a, StreamOrDevice s = {});

/** Computes the inverse error function of the elements of an array. */
MLX_API array erfinv(const array& a, StreamOrDevice s = {});

/** Computes the expm1 function of the elements of an array. */
MLX_API array expm1(const array& a, StreamOrDevice s = {});

/** Stop the flow of gradients. */
MLX_API array stop_gradient(const array& a, StreamOrDevice s = {});

/** Round a floating point number */
MLX_API array round(const array& a, int decimals, StreamOrDevice s = {});
inline array round(const array& a, StreamOrDevice s = {}) {
  return round(a, 0, s);
}

/** Matrix-matrix multiplication. */
MLX_API array matmul(const array& a, const array& b, StreamOrDevice s = {});

/** Gather array entries given indices and slices */
MLX_API array gather(
    const array& a,
    const std::vector<array>& indices,
    const std::vector<int>& axes,
    const Shape& slice_sizes,
    StreamOrDevice s = {});
inline array gather(
    const array& a,
    const array& indices,
    int axis,
    const Shape& slice_sizes,
    StreamOrDevice s = {}) {
  return gather(a, {indices}, std::vector<int>{axis}, slice_sizes, s);
}

/**  Compute the Kronecker product of two arrays. */
MLX_API array kron(const array& a, const array& b, StreamOrDevice s = {});

/** Take array slices at the given indices of the specified axis. */
MLX_API array
take(const array& a, const array& indices, int axis, StreamOrDevice s = {});
MLX_API array take(const array& a, int index, int axis, StreamOrDevice s = {});

/** Take array entries at the given indices treating the array as flattened. */
MLX_API array take(const array& a, const array& indices, StreamOrDevice s = {});
MLX_API array take(const array& a, int index, StreamOrDevice s = {});

/** Take array entries given indices along the axis */
MLX_API array take_along_axis(
    const array& a,
    const array& indices,
    int axis,
    StreamOrDevice s = {});

/** Put the values into the array at the given indices along the axis */
MLX_API array put_along_axis(
    const array& a,
    const array& indices,
    const array& values,
    int axis,
    StreamOrDevice s = {});

/** Add the values into the array at the given indices along the axis */
MLX_API array scatter_add_axis(
    const array& a,
    const array& indices,
    const array& values,
    int axis,
    StreamOrDevice s = {});

/** Scatter updates to the given indices.
 *
 * The parameters ``indices`` and ``axes`` determine the locations of ``a``
 * that are updated with the values in ``updates``. Assuming 1-d ``indices``
 * for simplicity, ``indices[i]`` are the indices on axis ``axes[i]`` to which
 * the values in ``updates`` will be applied. Note each array in
 * ``indices`` is assigned to a corresponding axis and hence ``indices.size() ==
 * axes.size()``. If an index/axis pair is not provided then indices along that
 * axis are assumed to be zero.
 *
 * Note the rank of ``updates`` must be equal to the sum of the rank of the
 * broadcasted ``indices`` and the rank of ``a``. In other words, assuming the
 * arrays in ``indices`` have the same shape, ``updates.ndim() ==
 * indices[0].ndim() + a.ndim()``. The leading dimensions of ``updates``
 * correspond to the indices, and the remaining ``a.ndim()`` dimensions are the
 * values that will be applied to the given location in ``a``.
 *
 * For example:
 *
 * @code
 * auto in = zeros({4, 4}, float32);
 * auto indices = array({2});
 * auto updates = reshape(arange(1, 3, float32), {1, 1, 2});
 * std::vector<int> axes{0};
 *
 * auto out = scatter(in, {indices}, updates, axes);
 * @endcode
 *
 * will produce:
 *
 * @code
 * array([[0, 0, 0, 0],
 *        [0, 0, 0, 0],
 *        [1, 2, 0, 0],
 *        [0, 0, 0, 0]], dtype=float32)
 * @endcode
 *
 * This scatters the two-element row vector ``[1, 2]`` starting at the ``(2,
 * 0)`` position of ``a``.
 *
 * Adding another element to ``indices`` will scatter into another location of
 * ``a``. We also have to add an another update for the new index:
 *
 * @code
 * auto in = zeros({4, 4}, float32);
 * auto indices = array({2, 0});
 * auto updates = reshape(arange(1, 5, float32), {2, 1, 2});
 * std::vector<int> axes{0};
 *
 * auto out = scatter(in, {indices}, updates, axes):
 * @endcode
 *
 * will produce:
 *
 * @code
 * array([[3, 4, 0, 0],
 *        [0, 0, 0, 0],
 *        [1, 2, 0, 0],
 *        [0, 0, 0, 0]], dtype=float32)
 * @endcode
 *
 * To control the scatter location on an additional axis, add another index
 * array to ``indices`` and another axis to ``axes``:
 *
 * @code
 * auto in = zeros({4, 4}, float32);
 * auto indices = std::vector{array({2, 0}), array({1, 2})};
 * auto updates = reshape(arange(1, 5, float32), {2, 1, 2});
 * std::vector<int> axes{0, 1};
 *
 * auto out = scatter(in, indices, updates, axes);
 * @endcode
 *
 * will produce:
 *
 * @code
 * array([[0, 0, 3, 4],
 *       [0, 0, 0, 0],
 *       [0, 1, 2, 0],
 *       [0, 0, 0, 0]], dtype=float32)
 * @endcode
 *
 * Items in indices are broadcasted together. This means:
 *
 * @code
 * auto indices = std::vector{array({2, 0}), array({1})};
 * @endcode
 *
 * is equivalent to:
 *
 * @code
 * auto indices = std::vector{array({2, 0}), array({1, 1})};
 * @endcode
 *
 * Note, ``scatter`` does not perform bounds checking on the indices and
 * updates.  Out-of-bounds accesses on ``a`` are undefined and typically result
 * in unintended or invalid memory writes.
 */
MLX_API array scatter(
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
MLX_API array scatter_add(
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
MLX_API array scatter_prod(
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
MLX_API array scatter_max(
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
MLX_API array scatter_min(
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

MLX_API array masked_scatter(
    const array& a,
    const array& mask,
    const array& src,
    StreamOrDevice s = {});

/** Square root the elements of an array. */
MLX_API array sqrt(const array& a, StreamOrDevice s = {});

/** Square root and reciprocal the elements of an array. */
MLX_API array rsqrt(const array& a, StreamOrDevice s = {});

/** Softmax of an array. */
MLX_API array softmax(
    const array& a,
    const std::vector<int>& axes,
    bool precise = false,
    StreamOrDevice s = {});

/** Softmax of an array. */
MLX_API array
softmax(const array& a, bool precise = false, StreamOrDevice s = {});

/** Softmax of an array. */
inline array
softmax(const array& a, int axis, bool precise = false, StreamOrDevice s = {}) {
  return softmax(a, std::vector<int>{axis}, precise, s);
}

/** Raise elements of a to the power of b element-wise */
MLX_API array power(const array& a, const array& b, StreamOrDevice s = {});

/** Cumulative sum of an array. */
MLX_API array cumsum(
    const array& a,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

/** Cumulative sum of an array along the given axis. */
MLX_API array cumsum(
    const array& a,
    int axis,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

/** Cumulative product of an array. */
MLX_API array cumprod(
    const array& a,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

/** Cumulative product of an array along the given axis. */
MLX_API array cumprod(
    const array& a,
    int axis,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

/** Cumulative max of an array. */
MLX_API array cummax(
    const array& a,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

/** Cumulative max of an array along the given axis. */
MLX_API array cummax(
    const array& a,
    int axis,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

/** Cumulative min of an array. */
MLX_API array cummin(
    const array& a,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

/** Cumulative min of an array along the given axis. */
MLX_API array cummin(
    const array& a,
    int axis,
    bool reverse = false,
    bool inclusive = true,
    StreamOrDevice s = {});

/** General convolution with a filter */
MLX_API array conv_general(
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
MLX_API array conv1d(
    const array& input,
    const array& weight,
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int groups = 1,
    StreamOrDevice s = {});

/** 2D convolution with a filter */
MLX_API array conv2d(
    const array& input,
    const array& weight,
    const std::pair<int, int>& stride = {1, 1},
    const std::pair<int, int>& padding = {0, 0},
    const std::pair<int, int>& dilation = {1, 1},
    int groups = 1,
    StreamOrDevice s = {});

/** 3D convolution with a filter */
MLX_API array conv3d(
    const array& input,
    const array& weight,
    const std::tuple<int, int, int>& stride = {1, 1, 1},
    const std::tuple<int, int, int>& padding = {0, 0, 0},
    const std::tuple<int, int, int>& dilation = {1, 1, 1},
    int groups = 1,
    StreamOrDevice s = {});

/** 1D transposed convolution with a filter */
MLX_API array conv_transpose1d(
    const array& input,
    const array& weight,
    int stride = 1,
    int padding = 0,
    int dilation = 1,
    int output_padding = 0,
    int groups = 1,
    StreamOrDevice s = {});

/** 2D transposed convolution with a filter */
MLX_API array conv_transpose2d(
    const array& input,
    const array& weight,
    const std::pair<int, int>& stride = {1, 1},
    const std::pair<int, int>& padding = {0, 0},
    const std::pair<int, int>& dilation = {1, 1},
    const std::pair<int, int>& output_padding = {0, 0},
    int groups = 1,
    StreamOrDevice s = {});

/** 3D transposed convolution with a filter */
MLX_API array conv_transpose3d(
    const array& input,
    const array& weight,
    const std::tuple<int, int, int>& stride = {1, 1, 1},
    const std::tuple<int, int, int>& padding = {0, 0, 0},
    const std::tuple<int, int, int>& dilation = {1, 1, 1},
    const std::tuple<int, int, int>& output_padding = {0, 0, 0},
    int groups = 1,
    StreamOrDevice s = {});

/** Quantized matmul multiplies x with a quantized matrix w*/
MLX_API array quantized_matmul(
    array x,
    array w,
    array scales,
    std::optional<array> biases = std::nullopt,
    bool transpose = true,
    std::optional<int> group_size = std::nullopt,
    std::optional<int> bits = std::nullopt,
    const std::string& mode = "affine",
    StreamOrDevice s = {});

/** Quantize a matrix along its last axis */
MLX_API std::vector<array> quantize(
    const array& w,
    std::optional<int> group_size = std::nullopt,
    std::optional<int> bits = std::nullopt,
    const std::string& mode = "affine",
    const std::optional<array>& global_scale = std::nullopt,
    StreamOrDevice s = {});

/** Dequantize a matrix produced by quantize() */
MLX_API array dequantize(
    const array& w,
    const array& scales,
    const std::optional<array>& biases = std::nullopt,
    std::optional<int> group_size = std::nullopt,
    std::optional<int> bits = std::nullopt,
    const std::string& mode = "affine",
    const std::optional<array>& global_scale = std::nullopt,
    std::optional<Dtype> dtype = std::nullopt,
    StreamOrDevice s = {});

MLX_API array qqmm(
    array x, // input activations
    array w, // maybe quantized weights
    const std::optional<array> w_scales = std::nullopt, // optional scales if w
                                                        // is quantized
    std::optional<int> group_size = std::nullopt,
    std::optional<int> bits = std::nullopt,
    const std::string& mode = "nvfp4",
    const std::optional<array> global_scale_x = std::nullopt,
    const std::optional<array> global_scale_w = std::nullopt,
    StreamOrDevice s = {});

/** Convert an E4M3 float8 to the given floating point dtype. */
MLX_API array from_fp8(array x, Dtype dtype, StreamOrDevice s = {});

/** Convert a floating point matrix to E4M3 float8. */
MLX_API array to_fp8(array x, StreamOrDevice s = {});

/** Compute matrix products with matrix-level gather. */
MLX_API array gather_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases = std::nullopt,
    std::optional<array> lhs_indices = std::nullopt,
    std::optional<array> rhs_indices = std::nullopt,
    bool transpose = true,
    std::optional<int> group_size = std::nullopt,
    std::optional<int> bits = std::nullopt,
    const std::string& mode = "affine",
    bool sorted_indices = false,
    StreamOrDevice s = {});

/** Returns a contraction of a and b over multiple dimensions. */
MLX_API array tensordot(
    const array& a,
    const array& b,
    const int axis = 2,
    StreamOrDevice s = {});

MLX_API array tensordot(
    const array& a,
    const array& b,
    const std::vector<int>& axes_a,
    const std::vector<int>& axes_b,
    StreamOrDevice s = {});

/** Compute the outer product of two vectors. */
MLX_API array outer(const array& a, const array& b, StreamOrDevice s = {});

/** Compute the inner product of two vectors. */
MLX_API array inner(const array& a, const array& b, StreamOrDevice s = {});

/** Compute D = beta * C + alpha * (A @ B) */
MLX_API array addmm(
    array c,
    array a,
    array b,
    const float& alpha = 1.f,
    const float& beta = 1.f,
    StreamOrDevice s = {});

/** Compute matrix product with block masking */
MLX_API array block_masked_mm(
    array a,
    array b,
    int block_size,
    std::optional<array> mask_out = std::nullopt,
    std::optional<array> mask_lhs = std::nullopt,
    std::optional<array> mask_rhs = std::nullopt,
    StreamOrDevice s = {});

/** Compute matrix product with matrix-level gather */
MLX_API array gather_mm(
    array a,
    array b,
    std::optional<array> lhs_indices = std::nullopt,
    std::optional<array> rhs_indices = std::nullopt,
    bool sorted_indices = false,
    StreamOrDevice s = {});

/**
 * Compute a matrix product but segment the inner dimension and write the
 * result separately for each segment.
 */
MLX_API array
segmented_mm(array a, array b, array segments, StreamOrDevice s = {});

/** Extract a diagonal or construct a diagonal array */
MLX_API array diagonal(
    const array& a,
    int offset = 0,
    int axis1 = 0,
    int axis2 = 1,
    StreamOrDevice s = {});

/** Extract diagonal from a 2d array or create a diagonal matrix. */
MLX_API array diag(const array& a, int k = 0, StreamOrDevice s = {});

/** Return the sum along a specified diagonal in the given array. */
MLX_API array trace(
    const array& a,
    int offset,
    int axis1,
    int axis2,
    Dtype dtype,
    StreamOrDevice s = {});
MLX_API array
trace(const array& a, int offset, int axis1, int axis2, StreamOrDevice s = {});
MLX_API array trace(const array& a, StreamOrDevice s = {});

/**
 * Implements the identity function but allows injecting dependencies to other
 * arrays. This ensures that these other arrays will have been computed
 * when the outputs of this function are computed.
 */
MLX_API std::vector<array> depends(
    const std::vector<array>& inputs,
    const std::vector<array>& dependencies);

/** convert an array to an atleast ndim array */
MLX_API array atleast_1d(const array& a, StreamOrDevice s = {});
MLX_API std::vector<array> atleast_1d(
    const std::vector<array>& a,
    StreamOrDevice s = {});
MLX_API array atleast_2d(const array& a, StreamOrDevice s = {});
MLX_API std::vector<array> atleast_2d(
    const std::vector<array>& a,
    StreamOrDevice s = {});
MLX_API array atleast_3d(const array& a, StreamOrDevice s = {});
MLX_API std::vector<array> atleast_3d(
    const std::vector<array>& a,
    StreamOrDevice s = {});

/**
 * Extract the number of elements along some axes as a scalar array. Used to
 * allow shape dependent shapeless compilation (pun intended).
 */
MLX_API array number_of_elements(
    const array& a,
    std::vector<int> axes,
    bool inverted,
    Dtype dtype = int32,
    StreamOrDevice s = {});

MLX_API array conjugate(const array& a, StreamOrDevice s = {});

/** Bitwise and. */
MLX_API array
bitwise_and(const array& a, const array& b, StreamOrDevice s = {});
MLX_API array operator&(const array& a, const array& b);

/** Bitwise inclusive or. */
MLX_API array bitwise_or(const array& a, const array& b, StreamOrDevice s = {});
MLX_API array operator|(const array& a, const array& b);

/** Bitwise exclusive or. */
MLX_API array
bitwise_xor(const array& a, const array& b, StreamOrDevice s = {});
MLX_API array operator^(const array& a, const array& b);

/** Shift bits to the left. */
MLX_API array left_shift(const array& a, const array& b, StreamOrDevice s = {});
MLX_API array operator<<(const array& a, const array& b);

/** Shift bits to the right. */
MLX_API array
right_shift(const array& a, const array& b, StreamOrDevice s = {});
MLX_API array operator>>(const array& a, const array& b);

/** Invert the bits. */
MLX_API array bitwise_invert(const array& a, StreamOrDevice s = {});
MLX_API array operator~(const array& a);

MLX_API array view(const array& a, const Dtype& dtype, StreamOrDevice s = {});

/** Roll elements along an axis and introduce them on the other side */
MLX_API array roll(const array& a, int shift, StreamOrDevice s = {});
MLX_API array roll(const array& a, const Shape& shift, StreamOrDevice s = {});
MLX_API array roll(const array& a, int shift, int axis, StreamOrDevice s = {});
MLX_API array roll(
    const array& a,
    int shift,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
MLX_API array
roll(const array& a, const Shape& shift, int axis, StreamOrDevice s = {});
MLX_API array roll(
    const array& a,
    const Shape& shift,
    const std::vector<int>& axes,
    StreamOrDevice s = {});

/* The real part of a complex array. */
MLX_API array real(const array& a, StreamOrDevice s = {});

/* The imaginary part of a complex array. */
MLX_API array imag(const array& a, StreamOrDevice s = {});

/* Ensure the array's underlying memory is contiguous. */
MLX_API array
contiguous(const array& a, bool allow_col_major = false, StreamOrDevice s = {});

/** @} */

} // namespace mlx::core
