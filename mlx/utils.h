// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <exception>
#include <variant>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/dtype.h"
#include "mlx/stream.h"

namespace mlx::core {

using StreamOrDevice = std::variant<std::monostate, Stream, Device>;
Stream to_stream(StreamOrDevice s);
Stream to_stream(StreamOrDevice s, Device default_);

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
  inline void print(std::ostream& os, double val);
  inline void print(std::ostream& os, complex64_t val);

  bool capitalize_bool{false};
};

PrintFormatter& get_global_formatter();

/** Print the exception and then abort. */
void abort_with_exception(const std::exception& error);

/** Holds information about floating-point types. */
struct finfo {
  explicit finfo(Dtype dtype);
  Dtype dtype;
  double min;
  double max;
};

/** Holds information about integral types. */
struct iinfo {
  explicit iinfo(Dtype dtype);
  Dtype dtype;
  int64_t min;
  uint64_t max;
};

/** The type from promoting the arrays' types with one another. */
inline Dtype result_type(const array& a, const array& b) {
  return promote_types(a.dtype(), b.dtype());
}
inline Dtype result_type(const array& a, const array& b, const array& c) {
  return promote_types(result_type(a, b), c.dtype());
}
Dtype result_type(const std::vector<array>& arrays);

Shape broadcast_shapes(const Shape& s1, const Shape& s2);

/**
 * Returns the axis normalized to be in the range [0, ndim).
 */
int normalize_axis_index(
    int axis,
    int ndim,
    const std::string& msg_prefix = "");

std::ostream& operator<<(std::ostream& os, const Device& d);
std::ostream& operator<<(std::ostream& os, const Stream& s);
std::ostream& operator<<(std::ostream& os, const Dtype& d);
std::ostream& operator<<(std::ostream& os, const Dtype::Kind& k);
std::ostream& operator<<(std::ostream& os, array a);
std::ostream& operator<<(std::ostream& os, const std::vector<int>& v);
std::ostream& operator<<(std::ostream& os, const std::vector<int64_t>& v);
inline std::ostream& operator<<(std::ostream& os, const complex64_t& v) {
  return os << v.real() << (v.imag() >= 0 ? "+" : "") << v.imag() << "j";
}
inline std::ostream& operator<<(std::ostream& os, const float16_t& v) {
  return os << static_cast<float>(v);
}
inline std::ostream& operator<<(std::ostream& os, const bfloat16_t& v) {
  return os << static_cast<float>(v);
}

inline bool is_power_of_2(int n) {
  return ((n & (n - 1)) == 0) && n != 0;
}

inline int next_power_of_2(int n) {
  if (is_power_of_2(n)) {
    return n;
  }
  return pow(2, std::ceil(std::log2(n)));
}

namespace env {

int get_var(const char* name, int default_value);

inline int bfs_max_width() {
  static int bfs_max_width_ = get_var("MLX_BFS_MAX_WIDTH", 20);
  return bfs_max_width_;
}

inline int max_ops_per_buffer(int default_value) {
  static int max_ops_per_buffer_ =
      get_var("MLX_MAX_OPS_PER_BUFFER", default_value);
  return max_ops_per_buffer_;
}

inline int max_mb_per_buffer(int default_value) {
  static int max_mb_per_buffer_ =
      get_var("MLX_MAX_MB_PER_BUFFER", default_value);
  return max_mb_per_buffer_;
}

inline bool metal_fast_synch() {
  static bool metal_fast_synch = get_var("MLX_METAL_FAST_SYNCH", 0);
  return metal_fast_synch;
}

} // namespace env

} // namespace mlx::core
