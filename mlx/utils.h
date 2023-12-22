// Copyright © 2023 Apple Inc.

#pragma once

#include <numeric>
#include "array.h"
#include "device.h"
#include "dtype.h"
#include "stream.h"

namespace mlx::core {

/** The type from promoting the arrays' types with one another. */
Dtype result_type(const std::vector<array>& arrays);

std::vector<int> broadcast_shapes(
    const std::vector<int>& s1,
    const std::vector<int>& s2);

bool is_same_shape(const std::vector<array>& arrays);

/**
 * Returns the axis normalized to be in the range [0, ndim).
 * Based on numpy's normalize_axis_index. See
 * https://numpy.org/devdocs/reference/generated/numpy.lib.array_utils.normalize_axis_index.html
 */
int normalize_axis(int axis, int ndim);
std::vector<int> normalize_axes(const std::vector<int>& axes, int ndim);

std::ostream& operator<<(std::ostream& os, const Device& d);
std::ostream& operator<<(std::ostream& os, const Stream& s);
std::ostream& operator<<(std::ostream& os, const Dtype& d);
std::ostream& operator<<(std::ostream& os, const Dtype::Kind& k);
std::ostream& operator<<(std::ostream& os, array a);
std::ostream& operator<<(std::ostream& os, const std::vector<int>& v);
std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v);
inline std::ostream& operator<<(std::ostream& os, const complex64_t& v) {
  return os << v.real() << (v.imag() > 0 ? "+" : "") << v.imag() << "j";
}
inline std::ostream& operator<<(std::ostream& os, const float16_t& v) {
  return os << static_cast<float>(v);
}
inline std::ostream& operator<<(std::ostream& os, const bfloat16_t& v) {
  return os << static_cast<float>(v);
}

using IntOrVec = std::variant<std::monostate, int, std::vector<int>>;
inline std::vector<int> get_reduce_axes(const IntOrVec& v, int dims) {
  std::vector<int> axes;
  if (std::holds_alternative<std::monostate>(v)) {
    axes.resize(dims);
    std::iota(axes.begin(), axes.end(), 0);
  } else if (auto pv = std::get_if<int>(&v); pv) {
    axes.push_back(*pv);
  } else {
    axes = std::get<std::vector<int>>(v);
  }
  return axes;
}
} // namespace mlx::core
