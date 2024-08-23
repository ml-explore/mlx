// Copyright © 2023 Apple Inc.

#pragma once

#include <optional>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

namespace mlx::core::linalg {

/**
 * Compute vector or matrix norms.
 *
 * - If axis and ord are both unspecified, computes the 2-norm of flatten(x).
 * - If axis is not provided but ord is, then x must be either 1D or 2D.
 * - If axis is provided, but ord is not, then the 2-norm (or Frobenius norm
 *   for matrices) is computed along the given axes. At most 2 axes can be
 *   specified.
 * - If both axis and ord are provided, then the corresponding matrix or vector
 *   norm is computed. At most 2 axes can be specified.
 */
array norm(
    const array& a,
    const double ord,
    const std::optional<std::vector<int>>& axis = std::nullopt,
    bool keepdims = false,
    StreamOrDevice s = {});
inline array norm(
    const array& a,
    const double ord,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {}) {
  return norm(a, ord, std::vector<int>{axis}, keepdims, s);
}
array norm(
    const array& a,
    const std::string& ord,
    const std::optional<std::vector<int>>& axis = std::nullopt,
    bool keepdims = false,
    StreamOrDevice s = {});
inline array norm(
    const array& a,
    const std::string& ord,
    int axis,
    bool keepdims = false,
    StreamOrDevice s = {}) {
  return norm(a, ord, std::vector<int>{axis}, keepdims, s);
}
array norm(
    const array& a,
    const std::optional<std::vector<int>>& axis = std::nullopt,
    bool keepdims = false,
    StreamOrDevice s = {});
inline array
norm(const array& a, int axis, bool keepdims = false, StreamOrDevice s = {}) {
  return norm(a, std::vector<int>{axis}, keepdims, s);
}

std::pair<array, array> qr(const array& a, StreamOrDevice s = {});

std::vector<array> svd(const array& a, StreamOrDevice s = {});

array inv(const array& a, StreamOrDevice s = {});

array tri_inv(const array& a, bool upper = false, StreamOrDevice s = {});

array cholesky(const array& a, bool upper = false, StreamOrDevice s = {});

array pinv(const array& a, StreamOrDevice s = {});

array cholesky_inv(const array& a, bool upper = false, StreamOrDevice s = {});

/**
 * Compute the cross product of two vectors or vector fields.
 *
 * - The cross product is defined only for 3-dimensional vectors.
 * - The function supports arrays that are 1-D or higher. For arrays with more than one dimension,
 *   the cross product is calculated along the specified axis.
 * - The function requires both input arrays to be of the same shape and have at least 3 elements along the specified axis.
 */
array cross_product(const array& a, const array& b, int axis = -1, StreamOrDevice s = {});

} // namespace mlx::core::linalg
