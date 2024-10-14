// Copyright © 2024 Apple Inc.

#include <nanobind/stl/complex.h>

#include "python/src/convert.h"
#include "python/src/utils.h"

#include "mlx/utils.h"

enum PyScalarT {
  pybool = 0,
  pyint = 1,
  pyfloat = 2,
  pycomplex = 3,
};

namespace nanobind {
template <>
struct ndarray_traits<float16_t> {
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
  } else if (type == nb::bfloat16) {
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

template <typename T, typename... NDParams>
nb::ndarray<NDParams...> mlx_to_nd_array_impl(
    array a,
    std::optional<nb::dlpack::dtype> t = {}) {
  {
    nb::gil_scoped_release nogil;
    a.eval();
  }
  std::vector<size_t> shape(a.shape().begin(), a.shape().end());
  std::vector<int64_t> strides(a.strides().begin(), a.strides().end());
  return nb::ndarray<NDParams...>(
      a.data<T>(),
      a.ndim(),
      shape.data(),
      /* owner= */ nb::none(),
      strides.data(),
      t.value_or(nb::dtype<T>()));
}

template <typename... NDParams>
nb::ndarray<NDParams...> mlx_to_nd_array(const array& a) {
  switch (a.dtype()) {
    case bool_:
      return mlx_to_nd_array_impl<bool, NDParams...>(a);
    case uint8:
      return mlx_to_nd_array_impl<uint8_t, NDParams...>(a);
    case uint16:
      return mlx_to_nd_array_impl<uint16_t, NDParams...>(a);
    case uint32:
      return mlx_to_nd_array_impl<uint32_t, NDParams...>(a);
    case uint64:
      return mlx_to_nd_array_impl<uint64_t, NDParams...>(a);
    case int8:
      return mlx_to_nd_array_impl<int8_t, NDParams...>(a);
    case int16:
      return mlx_to_nd_array_impl<int16_t, NDParams...>(a);
    case int32:
      return mlx_to_nd_array_impl<int32_t, NDParams...>(a);
    case int64:
      return mlx_to_nd_array_impl<int64_t, NDParams...>(a);
    case float16:
      return mlx_to_nd_array_impl<float16_t, NDParams...>(a);
    case bfloat16:
      throw nb::type_error("bfloat16 arrays cannot be converted to NumPy.");
    case float32:
      return mlx_to_nd_array_impl<float, NDParams...>(a);
    case complex64:
      return mlx_to_nd_array_impl<std::complex<float>, NDParams...>(a);
    default:
      throw nb::type_error("type cannot be converted to NumPy.");
  }
}

nb::ndarray<nb::numpy> mlx_to_np_array(const array& a) {
  return mlx_to_nd_array<nb::numpy>(a);
}

nb::ndarray<> mlx_to_dlpack(const array& a) {
  return mlx_to_nd_array<>(a);
}

nb::object to_scalar(array& a) {
  if (a.size() != 1) {
    throw std::invalid_argument(
        "[convert] Only length-1 arrays can be converted to Python scalars.");
  }
  {
    nb::gil_scoped_release nogil;
    a.eval();
  }
  switch (a.dtype()) {
    case bool_:
      return nb::cast(a.item<bool>());
    case uint8:
      return nb::cast(a.item<uint8_t>());
    case uint16:
      return nb::cast(a.item<uint16_t>());
    case uint32:
      return nb::cast(a.item<uint32_t>());
    case uint64:
      return nb::cast(a.item<uint64_t>());
    case int8:
      return nb::cast(a.item<int8_t>());
    case int16:
      return nb::cast(a.item<int16_t>());
    case int32:
      return nb::cast(a.item<int32_t>());
    case int64:
      return nb::cast(a.item<int64_t>());
    case float16:
      return nb::cast(static_cast<float>(a.item<float16_t>()));
    case float32:
      return nb::cast(a.item<float>());
    case bfloat16:
      return nb::cast(static_cast<float>(a.item<bfloat16_t>()));
    case complex64:
      return nb::cast(a.item<std::complex<float>>());
    default:
      throw nb::type_error("type cannot be converted to Python scalar.");
  }
}

template <typename T, typename U = T>
nb::list to_list(array& a, size_t index, int dim) {
  nb::list pl;
  auto stride = a.strides()[dim];
  for (int i = 0; i < a.shape(dim); ++i) {
    if (dim == a.ndim() - 1) {
      pl.append(static_cast<U>(a.data<T>()[index]));
    } else {
      pl.append(to_list<T, U>(a, index, dim + 1));
    }
    index += stride;
  }
  return pl;
}

nb::object tolist(array& a) {
  if (a.ndim() == 0) {
    return to_scalar(a);
  }
  {
    nb::gil_scoped_release nogil;
    a.eval();
  }
  switch (a.dtype()) {
    case bool_:
      return to_list<bool>(a, 0, 0);
    case uint8:
      return to_list<uint8_t>(a, 0, 0);
    case uint16:
      return to_list<uint16_t>(a, 0, 0);
    case uint32:
      return to_list<uint32_t>(a, 0, 0);
    case uint64:
      return to_list<uint64_t>(a, 0, 0);
    case int8:
      return to_list<int8_t>(a, 0, 0);
    case int16:
      return to_list<int16_t>(a, 0, 0);
    case int32:
      return to_list<int32_t>(a, 0, 0);
    case int64:
      return to_list<int64_t>(a, 0, 0);
    case float16:
      return to_list<float16_t, float>(a, 0, 0);
    case float32:
      return to_list<float>(a, 0, 0);
    case bfloat16:
      return to_list<bfloat16_t, float>(a, 0, 0);
    case complex64:
      return to_list<std::complex<float>>(a, 0, 0);
    default:
      throw nb::type_error("data type cannot be converted to Python list.");
  }
}

template <typename T, typename U>
void fill_vector(T list, std::vector<U>& vals) {
  for (auto l : list) {
    if (nb::isinstance<nb::list>(l)) {
      fill_vector(nb::cast<nb::list>(l), vals);
    } else if (nb::isinstance<nb::tuple>(*list.begin())) {
      fill_vector(nb::cast<nb::tuple>(l), vals);
    } else {
      vals.push_back(nb::cast<U>(l));
    }
  }
}

template <typename T>
PyScalarT validate_shape(
    T list,
    const std::vector<int>& shape,
    int idx,
    bool& all_python_primitive_elements) {
  if (idx >= shape.size()) {
    throw std::invalid_argument("Initialization encountered extra dimension.");
  }
  auto s = shape[idx];
  if (nb::len(list) != s) {
    throw std::invalid_argument(
        "Initialization encountered non-uniform length.");
  }

  if (s == 0) {
    return pyfloat;
  }

  PyScalarT type = pybool;
  for (auto l : list) {
    PyScalarT t;
    if (nb::isinstance<nb::list>(l)) {
      t = validate_shape(
          nb::cast<nb::list>(l), shape, idx + 1, all_python_primitive_elements);
    } else if (nb::isinstance<nb::tuple>(*list.begin())) {
      t = validate_shape(
          nb::cast<nb::tuple>(l),
          shape,
          idx + 1,
          all_python_primitive_elements);
    } else if (nb::isinstance<array>(l)) {
      all_python_primitive_elements = false;
      auto arr = nb::cast<array>(l);
      if (arr.ndim() + idx + 1 == shape.size() &&
          std::equal(
              arr.shape().cbegin(),
              arr.shape().cend(),
              shape.cbegin() + idx + 1)) {
        t = pybool;
      } else {
        throw std::invalid_argument(
            "Initialization encountered non-uniform length.");
      }
    } else {
      if (nb::isinstance<nb::bool_>(l)) {
        t = pybool;
      } else if (nb::isinstance<nb::int_>(l)) {
        t = pyint;
      } else if (nb::isinstance<nb::float_>(l)) {
        t = pyfloat;
      } else if (PyComplex_Check(l.ptr())) {
        t = pycomplex;
      } else {
        std::ostringstream msg;
        msg << "Invalid type  " << nb::type_name(l.type()).c_str()
            << " received in array initialization.";
        throw std::invalid_argument(msg.str());
      }

      if (idx + 1 != shape.size()) {
        throw std::invalid_argument(
            "Initialization encountered non-uniform length.");
      }
    }
    type = std::max(type, t);
  }
  return type;
}

template <typename T>
void get_shape(T list, std::vector<int>& shape) {
  shape.push_back(check_shape_dim(nb::len(list)));
  if (shape.back() > 0) {
    auto l = list.begin();
    if (nb::isinstance<nb::list>(*l)) {
      return get_shape(nb::cast<nb::list>(*l), shape);
    } else if (nb::isinstance<nb::tuple>(*l)) {
      return get_shape(nb::cast<nb::tuple>(*l), shape);
    } else if (nb::isinstance<array>(*l)) {
      auto arr = nb::cast<array>(*l);
      for (int i = 0; i < arr.ndim(); i++) {
        shape.push_back(check_shape_dim(arr.shape(i)));
      }
      return;
    }
  }
}

template <typename T>
array array_from_list_impl(
    T pl,
    const PyScalarT& inferred_type,
    std::optional<Dtype> specified_type,
    const std::vector<int>& shape) {
  // Make the array
  switch (inferred_type) {
    case pybool: {
      std::vector<bool> vals;
      fill_vector(pl, vals);
      return array(vals.begin(), shape, specified_type.value_or(bool_));
    }
    case pyint: {
      auto dtype = specified_type.value_or(int32);
      if (dtype == int64) {
        std::vector<int64_t> vals;
        fill_vector(pl, vals);
        return array(vals.begin(), shape, dtype);
      } else if (dtype == uint64) {
        std::vector<uint64_t> vals;
        fill_vector(pl, vals);
        return array(vals.begin(), shape, dtype);
      } else if (dtype == uint32) {
        std::vector<uint32_t> vals;
        fill_vector(pl, vals);
        return array(vals.begin(), shape, dtype);
      } else if (issubdtype(dtype, inexact)) {
        std::vector<float> vals;
        fill_vector(pl, vals);
        return array(vals.begin(), shape, dtype);
      } else {
        std::vector<int> vals;
        fill_vector(pl, vals);
        return array(vals.begin(), shape, dtype);
      }
    }
    case pyfloat: {
      std::vector<float> vals;
      fill_vector(pl, vals);
      return array(vals.begin(), shape, specified_type.value_or(float32));
    }
    case pycomplex: {
      std::vector<std::complex<float>> vals;
      fill_vector(pl, vals);
      return array(
          reinterpret_cast<complex64_t*>(vals.data()),
          shape,
          specified_type.value_or(complex64));
    }
    default: {
      std::ostringstream msg;
      msg << "Should not happen, inferred: " << inferred_type
          << " on subarray made of only python primitive types.";
      throw std::runtime_error(msg.str());
    }
  }
}

template <typename T>
array array_from_list_impl(T pl, std::optional<Dtype> dtype) {
  // Compute the shape
  std::vector<int> shape;
  get_shape(pl, shape);

  // Validate the shape and type
  bool all_python_primitive_elements = true;
  auto type = validate_shape(pl, shape, 0, all_python_primitive_elements);

  if (all_python_primitive_elements) {
    // `pl` does not contain mlx arrays
    return array_from_list_impl(pl, type, dtype, shape);
  }

  // `pl` contains mlx arrays
  std::vector<array> arrays;
  for (auto l : pl) {
    arrays.push_back(create_array(nb::cast<ArrayInitType>(l), dtype));
  }
  return stack(arrays);
}

array array_from_list(nb::list pl, std::optional<Dtype> dtype) {
  return array_from_list_impl(pl, dtype);
}

array array_from_list(nb::tuple pl, std::optional<Dtype> dtype) {
  return array_from_list_impl(pl, dtype);
}

array create_array(ArrayInitType v, std::optional<Dtype> t) {
  if (auto pv = std::get_if<nb::bool_>(&v); pv) {
    return array(nb::cast<bool>(*pv), t.value_or(bool_));
  } else if (auto pv = std::get_if<nb::int_>(&v); pv) {
    return array(nb::cast<int>(*pv), t.value_or(int32));
  } else if (auto pv = std::get_if<nb::float_>(&v); pv) {
    return array(nb::cast<float>(*pv), t.value_or(float32));
  } else if (auto pv = std::get_if<std::complex<float>>(&v); pv) {
    return array(static_cast<complex64_t>(*pv), t.value_or(complex64));
  } else if (auto pv = std::get_if<nb::list>(&v); pv) {
    return array_from_list(*pv, t);
  } else if (auto pv = std::get_if<nb::tuple>(&v); pv) {
    return array_from_list(*pv, t);
  } else if (auto pv = std::get_if<
                 nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>>(&v);
             pv) {
    return nd_array_to_mlx(*pv, t);
  } else if (auto pv = std::get_if<array>(&v); pv) {
    return astype(*pv, t.value_or((*pv).dtype()));
  } else {
    auto arr = to_array_with_accessor(std::get<nb::object>(v));
    return astype(arr, t.value_or(arr.dtype()));
  }
}
