// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <exception>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <variant>

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/config.h"
#include "mlx/device.h"
#include "mlx/dtype.h"
#include "mlx/stream.h"

namespace mlx::core {

using StreamOrDevice = std::variant<
    std::monostate,
    Stream,
    ThreadLocalStream,
    Device,
    Device::DeviceType>;
MLX_API Stream to_stream(StreamOrDevice s);
MLX_API Stream to_stream(StreamOrDevice s, Device default_);

struct StreamContext {
 public:
  StreamContext(StreamOrDevice s)
      : _stream(default_stream(default_device())),
        _owner(std::this_thread::get_id()) {
    if (std::holds_alternative<std::monostate>(s)) {
      throw std::runtime_error(
          "[StreamContext] Invalid argument, please specify a stream or device.");
    }
    auto _s = to_stream(s);
    set_default_device(_s.device);
    set_default_stream(_s);
  }

  // noexcept(false) so the thread check below can throw. A destructor that
  // throws while another exception unwinds this thread calls std::terminate.
  ~StreamContext() noexcept(false) {
    if (std::this_thread::get_id() != _owner) {
      throw std::runtime_error(
          "[StreamContext] Destroyed in a different thread than it was "
          "constructed in.");
    }
    set_default_device(_stream.device);
    set_default_stream(_stream);
  }

 private:
  Stream _stream;
  std::thread::id _owner;
};

struct MLX_API PrintOptions {
  int precision{-1};
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
  PrintOptions format_options;
};

MLX_API void set_printoptions(PrintOptions options);

MLX_API PrintFormatter& get_global_formatter();

/** Return whether current thread is the first one that called this function. */
bool is_main_thread();

/** Holds information about floating-point types. */
struct MLX_API finfo {
  explicit finfo(Dtype dtype);
  Dtype dtype;
  int bits;
  double min;
  double max;
  double eps;
  double smallest_normal;
};

/** Holds information about integral types. */
struct MLX_API iinfo {
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
MLX_API Dtype result_type(const std::vector<array>& arrays);

MLX_API Shape broadcast_shapes(const Shape& s1, const Shape& s2);

template <typename T>
inline ShapeElem safe_cast(T dim, std::string_view op = "") {
  constexpr int64_t lo = std::numeric_limits<ShapeElem>::min();
  constexpr int64_t hi = std::numeric_limits<ShapeElem>::max();
  auto v = static_cast<int64_t>(dim);
  if (v < lo || v > hi) {
    std::ostringstream msg;
    if (!op.empty()) {
      msg << "[" << op << "] ";
    }
    msg << "Shape dimension " << v << " is outside the supported range [" << lo
        << ", " << hi
        << "]. MLX currently uses 32-bit integers for shape dimensions.";
    throw std::overflow_error(msg.str());
  }
  return static_cast<ShapeElem>(v);
}

/**
 * Returns the axis normalized to be in the range [0, ndim).
 */
MLX_API int
normalize_axis_index(int axis, int ndim, const std::string& msg_prefix = "");

MLX_API std::ostream& operator<<(std::ostream& os, const Device& d);
MLX_API std::ostream& operator<<(std::ostream& os, const Stream& s);
MLX_API std::ostream& operator<<(std::ostream& os, const Dtype& d);
MLX_API std::ostream& operator<<(std::ostream& os, const Dtype::Kind& k);
MLX_API std::ostream& operator<<(std::ostream& os, array a);
inline std::ostream& operator<<(std::ostream& os, const complex64_t& v) {
  return os << v.real() << (v.imag() >= 0 ? "+" : "") << v.imag() << "j";
}
inline std::ostream& operator<<(std::ostream& os, const float16_t& v) {
  return os << static_cast<float>(v);
}
inline std::ostream& operator<<(std::ostream& os, const bfloat16_t& v) {
  return os << static_cast<float>(v);
}

template <typename Vec, typename = std::enable_if_t<is_vector_v<Vec>>>
inline std::ostream& operator<<(std::ostream& os, const Vec& v) {
  os << "(";
  for (auto it = v.begin(); it != v.end(); ++it) {
    os << *it;
    if (it != std::prev(v.end())) {
      os << ",";
    }
  }
  os << ")";
  return os;
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

inline int max_ops_per_buffer(int default_value) {
  return config::get("MLX_MAX_OPS_PER_BUFFER", default_value);
}

inline int max_mb_per_buffer(int default_value) {
  return config::get("MLX_MAX_MB_PER_BUFFER", default_value);
}

inline bool enable_tf32() {
  return config::get("MLX_ENABLE_TF32", 1);
}

} // namespace env

} // namespace mlx::core
