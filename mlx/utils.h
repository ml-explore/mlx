// Copyright © 2023 Apple Inc.

#pragma once

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
} // namespace mlx::core
