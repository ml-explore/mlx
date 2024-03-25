// Copyright © 2024 Apple Inc.

#include <nanobind/stl/complex.h>

#include "python/src/convert.h"

#include "mlx/utils.h"

namespace nanobind {
template <>
struct ndarray_traits<float16_t> {
  static constexpr bool is_complex = false;
  static constexpr bool is_float = true;
  static constexpr bool is_bool = false;
  static constexpr bool is_int = false;
  static constexpr bool is_signed = true;
};

template <>
struct ndarray_traits<bfloat16_t> {
  static constexpr bool is_complex = false;
  static constexpr bool is_float = true;
  static constexpr bool is_bool = false;
  static constexpr bool is_int = false;
  static constexpr bool is_signed = true;
};

static constexpr dlpack::dtype bfloat16{4, 16, 1};
}; // namespace nanobind

template <typename T>
array nd_array_to_mlx_contiguous(
    nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> nd_array,
    const std::vector<int>& shape,
    Dtype dtype) {
  // Make a copy of the numpy buffer
  // Get buffer ptr pass to array constructor
  auto data_ptr = nd_array.data();
  return array(static_cast<const T*>(data_ptr), shape, dtype);
}

array nd_array_to_mlx(
    nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> nd_array,
    std::optional<Dtype> dtype) {
  // Compute the shape and size
  std::vector<int> shape;
  for (int i = 0; i < nd_array.ndim(); i++) {
    shape.push_back(check_shape_dim(nd_array.shape(i)));
  }
  auto type = nd_array.dtype();

  // Copy data and make array
  if (type == nb::dtype<bool>()) {
    return nd_array_to_mlx_contiguous<bool>(
        nd_array, shape, dtype.value_or(bool_));
  } else if (type == nb::dtype<uint8_t>()) {
    return nd_array_to_mlx_contiguous<uint8_t>(
        nd_array, shape, dtype.value_or(uint8));
  } else if (type == nb::dtype<uint16_t>()) {
    return nd_array_to_mlx_contiguous<uint16_t>(
        nd_array, shape, dtype.value_or(uint16));
  } else if (type == nb::dtype<uint32_t>()) {
    return nd_array_to_mlx_contiguous<uint32_t>(
        nd_array, shape, dtype.value_or(uint32));
  } else if (type == nb::dtype<uint64_t>()) {
    return nd_array_to_mlx_contiguous<uint64_t>(
        nd_array, shape, dtype.value_or(uint64));
  } else if (type == nb::dtype<int8_t>()) {
    return nd_array_to_mlx_contiguous<int8_t>(
        nd_array, shape, dtype.value_or(int8));
  } else if (type == nb::dtype<int16_t>()) {
    return nd_array_to_mlx_contiguous<int16_t>(
        nd_array, shape, dtype.value_or(int16));
  } else if (type == nb::dtype<int32_t>()) {
    return nd_array_to_mlx_contiguous<int32_t>(
        nd_array, shape, dtype.value_or(int32));
  } else if (type == nb::dtype<int64_t>()) {
    return nd_array_to_mlx_contiguous<int64_t>(
        nd_array, shape, dtype.value_or(int64));
  } else if (type == nb::dtype<float16_t>()) {
    return nd_array_to_mlx_contiguous<float16_t>(
        nd_array, shape, dtype.value_or(float16));
  } else if (type == nb::dtype<bfloat16_t>()) {
    return nd_array_to_mlx_contiguous<bfloat16_t>(
        nd_array, shape, dtype.value_or(bfloat16));
  } else if (type == nb::dtype<float>()) {
    return nd_array_to_mlx_contiguous<float>(
        nd_array, shape, dtype.value_or(float32));
  } else if (type == nb::dtype<double>()) {
    return nd_array_to_mlx_contiguous<double>(
        nd_array, shape, dtype.value_or(float32));
  } else if (type == nb::dtype<std::complex<float>>()) {
    return nd_array_to_mlx_contiguous<complex64_t>(
        nd_array, shape, dtype.value_or(complex64));
  } else if (type == nb::dtype<std::complex<double>>()) {
    return nd_array_to_mlx_contiguous<complex128_t>(
        nd_array, shape, dtype.value_or(complex64));
  } else {
    throw std::invalid_argument("Cannot convert numpy array to mlx array.");
  }
}

template <typename Lib, typename T>
nb::ndarray<Lib> mlx_to_nd_array(
    array a,
    std::optional<nb::dlpack::dtype> t = {}) {
  // Eval if not already evaled
  if (!a.is_evaled()) {
    nb::gil_scoped_release nogil;
    a.eval();
  }
  std::vector<size_t> shape(a.shape().begin(), a.shape().end());
  std::vector<int64_t> strides(a.strides().begin(), a.strides().end());
  return nb::ndarray<Lib>(
      a.data<T>(),
      a.ndim(),
      shape.data(),
      nb::handle(),
      strides.data(),
      t.value_or(nb::dtype<T>()));
}

template <typename Lib>
nb::ndarray<Lib> mlx_to_nd_array(const array& a) {
  switch (a.dtype()) {
    case bool_:
      return mlx_to_nd_array<Lib, bool>(a);
    case uint8:
      return mlx_to_nd_array<Lib, uint8_t>(a);
    case uint16:
      return mlx_to_nd_array<Lib, uint16_t>(a);
    case uint32:
      return mlx_to_nd_array<Lib, uint32_t>(a);
    case uint64:
      return mlx_to_nd_array<Lib, uint64_t>(a);
    case int8:
      return mlx_to_nd_array<Lib, int8_t>(a);
    case int16:
      return mlx_to_nd_array<Lib, int16_t>(a);
    case int32:
      return mlx_to_nd_array<Lib, int32_t>(a);
    case int64:
      return mlx_to_nd_array<Lib, int64_t>(a);
    case float16:
      return mlx_to_nd_array<Lib, float16_t>(a);
    case bfloat16:
      return mlx_to_nd_array<Lib, bfloat16_t>(a, nb::bfloat16);
    case float32:
      return mlx_to_nd_array<Lib, float>(a);
    case complex64:
      return mlx_to_nd_array<Lib, std::complex<float>>(a);
  }
}

nb::ndarray<nb::numpy> mlx_to_np_array(const array& a) {
  return mlx_to_nd_array<nb::numpy>(a);
}
