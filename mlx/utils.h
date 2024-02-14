// Copyright © 2023 Apple Inc.

#pragma once

#include <variant>

#include "array.h"
#include "device.h"
#include "dtype.h"
#include "stream.h"

namespace mlx::core {

using StreamOrDevice = std::variant<std::monostate, Stream, Device>;
Stream to_stream(StreamOrDevice s);

struct StreamContext {
 public:
  StreamContext(StreamOrDevice s) : _stream(default_stream(default_device())) {
    if (std::holds_alternative<std::monostate>(s)) {
      throw std::runtime_error(
          "[StreamContext] Invalid argument, please specify a stream or device.");
    }
    auto _s = to_stream(s);
    set_default_device(_s.device);
    set_default_stream(_s);
  }

  ~StreamContext() {
    set_default_device(_stream.device);
    set_default_stream(_stream);
  }

 private:
  Stream _stream;
};

struct PrintFormatter {
  inline void print(std::ostream& os, bool val);
  inline void print(std::ostream& os, int16_t val);
  inline void print(std::ostream& os, uint16_t val);
  inline void print(std::ostream& os, int32_t val);
  inline void print(std::ostream& os, uint32_t val);
  inline void print(std::ostream& os, int64_t val);
  inline void print(std::ostream& os, uint64_t val);
  inline void print(std::ostream& os, float16_t val);
  inline void print(std::ostream& os, bfloat16_t val);
  inline void print(std::ostream& os, float val);
  inline void print(std::ostream& os, complex64_t val);

  bool capitalize_bool{false};
};

extern PrintFormatter global_formatter;

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
