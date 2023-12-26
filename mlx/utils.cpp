// Copyright © 2023 Apple Inc.

#include <sstream>
#include <vector>

#include "utils.h"

namespace mlx::core {

Dtype result_type(const std::vector<array>& arrays) {
  std::vector<Dtype> dtypes(1, bool_);
  for (auto& arr : arrays) {
    dtypes.push_back(promote_types(dtypes.back(), arr.dtype()));
  }
  return dtypes.back();
}

std::vector<int> broadcast_shapes(
    const std::vector<int>& s1,
    const std::vector<int>& s2) {
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
  std::vector<int> out_shape(ndim);
  for (int i = ndim - 1; i >= diff; --i) {
    int a = big[i];
    int b = small[i - diff];
    if (b == a) {
      out_shape[i] = a;
    } else if (a == 1 || b == 1) {
      // 0 if a or b is 0 otherwise max(a, b)
      out_shape[i] = a * b;
    } else {
      std::ostringstream msg;
      msg << "Shapes " << s1 << " and " << s2 << " cannot be broadcast.";
      throw std::invalid_argument(msg.str());
    }
  }
  for (int i = diff - 1; i >= 0; --i) {
    out_shape[i] = big[i];
  }
  return out_shape;
}

bool is_same_shape(const std::vector<array>& arrays) {
  if (arrays.empty()) {
    return true;
  }
  return std::all_of(arrays.begin() + 1, arrays.end(), [&](const array& a) {
    return (a.shape() == arrays[0].shape());
  });
}

int normalize_axis(int axis, int ndim) {
  if (ndim <= 0) {
    throw std::invalid_argument("Number of dimensions must be positive.");
  }
  if (axis < -ndim || axis >= ndim) {
    std::ostringstream msg;
    msg << "Axis " << axis << " is out of bounds for array with " << ndim
        << " dimensions.";
    throw std::invalid_argument(msg.str());
  }
  if (axis < 0) {
    axis += ndim;
  }
  return axis;
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
  os << static_cast<uint>(x);
  return os;
}

namespace {

inline size_t elem_to_loc(
    int elem,
    const std::vector<int>& shape,
    const std::vector<size_t>& strides) {
  size_t loc = 0;
  for (int i = shape.size() - 1; i >= 0; --i) {
    auto q_and_r = ldiv(elem, shape[i]);
    loc += q_and_r.rem * strides[i];
    elem = q_and_r.quot;
  }
  return loc;
}

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
      os << a.data<T>()[index];
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
  std::vector<int> indices(a.ndim(), 0);
  os << std::boolalpha;
  os << "array(";
  if (a.ndim() == 0) {
    auto data = a.data<T>();
    os << data[0];
  } else {
    print_subarray<T>(os, a, 0, 0);
  }
  os << ", dtype=" << a.dtype() << ")";
  os << std::noboolalpha;
}

} // namespace

std::ostream& operator<<(std::ostream& os, const Dtype& dtype) {
  switch (dtype) {
    case bool_:
      return os << "bool";
    case uint8:
      return os << "uint8";
    case uint16:
      return os << "uint16";
    case uint32:
      return os << "uint32";
    case uint64:
      return os << "uint64";
    case int8:
      return os << "int8";
    case int16:
      return os << "int16";
    case int32:
      return os << "int32";
    case int64:
      return os << "int64";
    case float16:
      return os << "float16";
    case float32:
      return os << "float32";
    case bfloat16:
      return os << "bfloat16";
    case complex64:
      return os << "complex64";
  }
  return os;
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
  if (!a.is_evaled()) {
    a.eval();
  }
  switch (a.dtype()) {
    case bool_:
      print_array<bool>(os, a);
      break;
    case uint8:
      print_array<uint8_t>(os, a);
      break;
    case uint16:
      print_array<uint16_t>(os, a);
      break;
    case uint32:
      print_array<uint32_t>(os, a);
      break;
    case uint64:
      print_array<uint64_t>(os, a);
      break;
    case int8:
      print_array<int8_t>(os, a);
      break;
    case int16:
      print_array<int16_t>(os, a);
      break;
    case int32:
      print_array<int32_t>(os, a);
      break;
    case int64:
      print_array<int64_t>(os, a);
      break;
    case float16:
      print_array<float16_t>(os, a);
      break;
    case bfloat16:
      print_array<bfloat16_t>(os, a);
      break;
    case float32:
      print_array<float>(os, a);
      break;
    case complex64:
      print_array<complex64_t>(os, a);
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<int>& v) {
  os << "(";
  for (int i = 0; i < v.size(); ++i) {
    os << v[i] << ((i == v.size() - 1) ? "" : ",");
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<size_t>& v) {
  os << "(";
  for (int i = 0; i < v.size(); ++i) {
    os << v[i] << ((i == v.size() - 1) ? "" : ",");
  }
  os << ")";
  return os;
}

} // namespace mlx::core
