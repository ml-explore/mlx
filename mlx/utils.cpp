// Copyright © 2023 Apple Inc.

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

#include "mlx/dtype_utils.h"
#include "mlx/types/limits.h"
#include "mlx/utils.h"

namespace mlx::core {

Stream to_stream(StreamOrDevice s) {
  if (std::holds_alternative<std::monostate>(s)) {
    return default_stream(default_device());
  } else if (std::holds_alternative<Device>(s)) {
    return default_stream(std::get<Device>(s));
  } else {
    return std::get<Stream>(s);
  }
}

Stream to_stream(StreamOrDevice s, Device default_) {
  if (std::holds_alternative<std::monostate>(s)) {
    return default_stream(default_);
  } else if (std::holds_alternative<Device>(s)) {
    return default_stream(std::get<Device>(s));
  } else {
    return std::get<Stream>(s);
  }
}

void PrintFormatter::print(std::ostream& os, bool val) {
  if (capitalize_bool) {
    os << (val ? "True" : "False");
  } else {
    os << val;
  }
}
inline void PrintFormatter::print(std::ostream& os, int16_t val) {
  os << val;
}
inline void PrintFormatter::print(std::ostream& os, uint16_t val) {
  os << val;
}
inline void PrintFormatter::print(std::ostream& os, int32_t val) {
  os << val;
}
inline void PrintFormatter::print(std::ostream& os, uint32_t val) {
  os << val;
}
inline void PrintFormatter::print(std::ostream& os, int64_t val) {
  os << val;
}
inline void PrintFormatter::print(std::ostream& os, uint64_t val) {
  os << val;
}
inline void PrintFormatter::print(std::ostream& os, float16_t val) {
  os << val;
}
inline void PrintFormatter::print(std::ostream& os, bfloat16_t val) {
  os << val;
}
inline void PrintFormatter::print(std::ostream& os, float val) {
  os << val;
}
inline void PrintFormatter::print(std::ostream& os, double val) {
  os << val;
}
inline void PrintFormatter::print(std::ostream& os, complex64_t val) {
  os << val.real();
  if (val.imag() >= 0 || std::isnan(val.imag())) {
    os << "+" << val.imag() << "j";
  } else {
    os << "-" << -val.imag() << "j";
  }
}

PrintFormatter& get_global_formatter() {
  static PrintFormatter formatter;
  return formatter;
}

void abort_with_exception(const std::exception& error) {
  std::ostringstream msg;
  msg << "Terminating due to uncaught exception: " << error.what();
  std::cerr << msg.str() << std::endl;
  std::abort();
}

Dtype result_type(const std::vector<array>& arrays) {
  Dtype t = bool_;
  for (auto& arr : arrays) {
    t = promote_types(t, arr.dtype());
  }
  return t;
}

Shape broadcast_shapes(const Shape& s1, const Shape& s2) {
  // Use the same broadcasting rules as numpy
  // https://numpy.org/doc/1.20/user/theory.broadcasting.html
  // "The size of the trailing axes for both arrays in an operation must
  // either be the same size or one of them must be one."
  int ndim1 = s1.size();
  int ndim2 = s2.size();
  int ndim = std::max(ndim1, ndim2);
  int diff = std::abs(ndim1 - ndim2);
  const auto& big = ndim1 > ndim2 ? s1 : s2;
  const auto& small = ndim1 > ndim2 ? s2 : s1;
  Shape out_shape(ndim);
  for (int i = ndim - 1; i >= diff; --i) {
    auto a = big[i];
    auto b = small[i - diff];
    if (b == a) {
      out_shape[i] = a;
    } else if (a == 1 || b == 1) {
      // 0 if a or b is 0 otherwise max(a, b)
      out_shape[i] = a * b;
    } else {
      std::ostringstream msg;
      msg << "[broadcast_shapes] Shapes " << s1 << " and " << s2
          << " cannot be broadcast.";
      throw std::invalid_argument(msg.str());
    }
  }
  for (int i = diff - 1; i >= 0; --i) {
    out_shape[i] = big[i];
  }
  return out_shape;
}

int normalize_axis_index(
    int axis,
    int ndim,
    const std::string& msg_prefix /* = "" */) {
  if (axis < -ndim || axis >= ndim) {
    std::ostringstream msg;
    msg << msg_prefix << "Axis " << axis << " is out of bounds for array with "
        << ndim << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  return axis < 0 ? axis + ndim : axis;
}

std::ostream& operator<<(std::ostream& os, const Device& d) {
  os << "Device(";
  switch (d.type) {
    case Device::cpu:
      os << "cpu";
      break;
    case Device::gpu:
      os << "gpu";
      break;
  }
  os << ", " << d.index << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Stream& s) {
  os << "Stream(";
  os << s.device;
  os << ", " << s.index << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, int8_t x) {
  os << static_cast<int>(x);
  return os;
}

std::ostream& operator<<(std::ostream& os, uint8_t x) {
  os << static_cast<unsigned int>(x);
  return os;
}

namespace {

template <typename T>
void print_subarray(std::ostream& os, const array& a, size_t index, int dim) {
  int num_print = 3;
  int n = a.shape(dim);
  size_t s = a.strides()[dim];
  bool is_last = dim == a.ndim() - 1;
  auto prefix = is_last ? "" : std::string(7 + dim, ' ');
  auto postfix = is_last ? ", " : ",\n";
  os << "[";
  for (int i = 0; i < n; ++i) {
    os << (i == 0 ? "" : prefix);
    if (i == num_print && n > 2 * num_print) {
      os << "...";
      i = n - num_print - 1;
      index += s * (n - 2 * num_print - 1);
    } else if (is_last) {
      get_global_formatter().print(os, a.data<T>()[index]);
    } else {
      print_subarray<T>(os, a, index, dim + 1);
    }
    os << (i == n - 1 ? "" : postfix);
    index += s;
  }
  os << "]";
}

template <typename T>
void print_array(std::ostream& os, const array& a) {
  os << std::boolalpha;
  os << "array(";
  if (a.ndim() == 0) {
    auto data = a.data<T>();
    get_global_formatter().print(os, data[0]);
  } else {
    print_subarray<T>(os, a, 0, 0);
  }
  os << ", dtype=" << a.dtype() << ")";
  os << std::noboolalpha;
}

} // namespace

std::ostream& operator<<(std::ostream& os, const Dtype& dtype) {
  return os << dtype_to_string(dtype);
}

std::ostream& operator<<(std::ostream& os, const Dtype::Kind& k) {
  switch (k) {
    case Dtype::Kind::b:
      return os << "b";
    case Dtype::Kind::i:
      return os << "i";
    case Dtype::Kind::u:
      return os << "u";
    case Dtype::Kind::f:
      return os << "f";
    case Dtype::Kind::c:
      return os << "c";
    case Dtype::Kind::V:
      return os << "V";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, array a) {
  a.eval();
  dispatch_all_types(a.dtype(), [&](auto type_tag) {
    print_array<MLX_GET_TYPE(type_tag)>(os, a);
  });
  return os;
}

namespace env {

int get_var(const char* name, int default_value) {
  if (const char* buff_str = std::getenv(name)) {
    return atoi(buff_str);
  } else {
    return default_value;
  }
}

} // namespace env

template <typename T>
void set_finfo_limits(double& min, double& max, double& eps) {
  min = numeric_limits<T>::lowest();
  max = numeric_limits<T>::max();
  eps = numeric_limits<T>::epsilon();
}

finfo::finfo(Dtype dtype) : dtype(dtype) {
  if (!issubdtype(dtype, inexact)) {
    std::ostringstream msg;
    msg << "[finfo] dtype " << dtype << " is not inexact.";
    throw std::invalid_argument(msg.str());
  }
  if (dtype == float32) {
    set_finfo_limits<float>(min, max, eps);
  } else if (dtype == float16) {
    set_finfo_limits<float16_t>(min, max, eps);
  } else if (dtype == bfloat16) {
    set_finfo_limits<bfloat16_t>(min, max, eps);
  } else if (dtype == float64) {
    set_finfo_limits<double>(min, max, eps);
  } else if (dtype == complex64) {
    this->dtype = float32;
    set_finfo_limits<float>(min, max, eps);
  }
}

template <typename T>
void set_iinfo_limits(int64_t& min, uint64_t& max) {
  min = std::numeric_limits<T>::min();
  max = std::numeric_limits<T>::max();
}

iinfo::iinfo(Dtype dtype) : dtype(dtype) {
  dispatch_int_types(dtype, "[iinfo]", [&](auto type_tag) {
    set_iinfo_limits<MLX_GET_TYPE(type_tag)>(min, max);
  });
}

} // namespace mlx::core
