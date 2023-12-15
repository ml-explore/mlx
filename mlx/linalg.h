#pragma once

#include <variant>

#include "array.h"
#include "device.h"
#include "stream.h"
#include "string.h"

namespace mlx::core::linalg {

using StreamOrDevice = std::variant<std::monostate, Stream, Device>;

/** TODO:
 * Norm, SVD, QR, LU, Cholesky, Eig, Eigh, Conj, Transpose, Inverse, Solve
 * and more...
 */

/**
 * Compute the Matrix or vector norm.
 * Currently only supports vector norms and Frobenius norm.
 *
 **/
array norm(
    const array& a,
    const std::variant<int, std::string>& ord,
    const std::vector<int>& axes,
    StreamOrDevice s = {});
array norm(
    const array& a,
    const std::variant<int, std::string>& ord,
    StreamOrDevice s = {});
array norm(const array& a, const std::vector<int>& axes, StreamOrDevice s = {});
array norm(const array& a, StreamOrDevice s = {});

/** Compute the SVD decomposition. */
array svd(const array& a, StreamOrDevice s = {});

/** Compute the QR Factorization. */
array qr(const array& a, StreamOrDevice s = {});

/** Compute the inverse of a matrix. */
array inv(const array& a, StreamOrDevice s = {});

/** Compute the determinant of a matrix. */
array det(const array& a, StreamOrDevice s = {});

/** Compute the trace of a matrix. */
array trace(const array& a, StreamOrDevice s = {});

} // namespace mlx::core::linalg