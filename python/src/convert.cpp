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
struct ndarray_traits<mx::float16_t> {
  static constexpr bool is_complex = false;
  static constexpr bool is_float = true;
  static constexpr bool is_bool = false;
  static constexpr bool is_int = false;
  static constexpr bool is_signed = true;
};

static constexpr dlpack::dtype bfloat16{4, 16, 1};
}; // namespace nanobind

int check_shape_dim(int64_t dim) {
  if (dim > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        "Shape dimension falls outside supported `int` range.");
  }
  return static_cast<int>(dim);
}

template <typename T>
mx::array nd_array_to_mlx_contiguous(
    nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> nd_array,
    const mx::Shape& shape,
    mx::Dtype dtype) {
  // Make a copy of the numpy buffer
  // Get buffer ptr pass to array constructor
  auto data_ptr = nd_array.data();
  return mx::array(static_cast<const T*>(data_ptr), shape, dtype);
}

mx::array nd_array_to_mlx(
    nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> nd_array,
    std::optional<mx::Dtype> dtype) {
  // Compute the shape and size
  mx::Shape shape;
  for (int i = 0; i < nd_array.ndim(); i++) {
    shape.push_back(check_shape_dim(nd_array.shape(i)));
  }
  auto type = nd_array.dtype();

  // Copy data and make array
  if (type == nb::dtype<bool>()) {
    return nd_array_to_mlx_contiguous<bool>(
        nd_array, shape, dtype.value_or(mx::bool_));
  } else if (type == nb::dtype<uint8_t>()) {
    return nd_array_to_mlx_contiguous<uint8_t>(
        nd_array, shape, dtype.value_or(mx::uint8));
  } else if (type == nb::dtype<uint16_t>()) {
    return nd_array_to_mlx_contiguous<uint16_t>(
        nd_array, shape, dtype.value_or(mx::uint16));
  } else if (type == nb::dtype<uint32_t>()) {
    return nd_array_to_mlx_contiguous<uint32_t>(
        nd_array, shape, dtype.value_or(mx::uint32));
  } else if (type == nb::dtype<uint64_t>()) {
    return nd_array_to_mlx_contiguous<uint64_t>(
        nd_array, shape, dtype.value_or(mx::uint64));
  } else if (type == nb::dtype<int8_t>()) {
    return nd_array_to_mlx_contiguous<int8_t>(
        nd_array, shape, dtype.value_or(mx::int8));
  } else if (type == nb::dtype<int16_t>()) {
    return nd_array_to_mlx_contiguous<int16_t>(
        nd_array, shape, dtype.value_or(mx::int16));
  } else if (type == nb::dtype<int32_t>()) {
    return nd_array_to_mlx_contiguous<int32_t>(
        nd_array, shape, dtype.value_or(mx::int32));
  } else if (type == nb::dtype<int64_t>()) {
    return nd_array_to_mlx_contiguous<int64_t>(
        nd_array, shape, dtype.value_or(mx::int64));
  } else if (type == nb::dtype<mx::float16_t>()) {
    return nd_array_to_mlx_contiguous<mx::float16_t>(
        nd_array, shape, dtype.value_or(mx::float16));
  } else if (type == nb::bfloat16) {
    return nd_array_to_mlx_contiguous<mx::bfloat16_t>(
        nd_array, shape, dtype.value_or(mx::bfloat16));
  } else if (type == nb::dtype<float>()) {
    return nd_array_to_mlx_contiguous<float>(
        nd_array, shape, dtype.value_or(mx::float32));
  } else if (type == nb::dtype<double>()) {
    return nd_array_to_mlx_contiguous<double>(
        nd_array, shape, dtype.value_or(mx::float32));
  } else if (type == nb::dtype<std::complex<float>>()) {
    return nd_array_to_mlx_contiguous<mx::complex64_t>(
        nd_array, shape, dtype.value_or(mx::complex64));
  } else if (type == nb::dtype<std::complex<double>>()) {
    return nd_array_to_mlx_contiguous<mx::complex128_t>(
        nd_array, shape, dtype.value_or(mx::complex64));
  } else {
    throw std::invalid_argument("Cannot convert numpy array to mlx array.");
  }
}

template <typename T, typename... NDParams>
nb::ndarray<NDParams...> mlx_to_nd_array_impl(
    mx::array a,
    std::optional<nb::dlpack::dtype> t = {}) {
  {
    nb::gil_scoped_release nogil;
    a.eval();
  }
  std::vector<size_t> shape(a.shape().begin(), a.shape().end());
  return nb::ndarray<NDParams...>(
      a.data<T>(),
      a.ndim(),
      shape.data(),
      /* owner= */ nb::none(),
      a.strides().data(),
      t.value_or(nb::dtype<T>()));
}

template <typename... NDParams>
nb::ndarray<NDParams...> mlx_to_nd_array(const mx::array& a) {
  switch (a.dtype()) {
    case mx::bool_:
      return mlx_to_nd_array_impl<bool, NDParams...>(a);
    case mx::uint8:
      return mlx_to_nd_array_impl<uint8_t, NDParams...>(a);
    case mx::uint16:
      return mlx_to_nd_array_impl<uint16_t, NDParams...>(a);
    case mx::uint32:
      return mlx_to_nd_array_impl<uint32_t, NDParams...>(a);
    case mx::uint64:
      return mlx_to_nd_array_impl<uint64_t, NDParams...>(a);
    case mx::int8:
      return mlx_to_nd_array_impl<int8_t, NDParams...>(a);
    case mx::int16:
      return mlx_to_nd_array_impl<int16_t, NDParams...>(a);
    case mx::int32:
      return mlx_to_nd_array_impl<int32_t, NDParams...>(a);
    case mx::int64:
      return mlx_to_nd_array_impl<int64_t, NDParams...>(a);
    case mx::float16:
      return mlx_to_nd_array_impl<mx::float16_t, NDParams...>(a);
    case mx::bfloat16:
      throw nb::type_error("bfloat16 arrays cannot be converted to NumPy.");
    case mx::float32:
      return mlx_to_nd_array_impl<float, NDParams...>(a);
    case mx::complex64:
      return mlx_to_nd_array_impl<std::complex<float>, NDParams...>(a);
    default:
      throw nb::type_error("type cannot be converted to NumPy.");
  }
}

nb::ndarray<nb::numpy> mlx_to_np_array(const mx::array& a) {
  return mlx_to_nd_array<nb::numpy>(a);
}

nb::ndarray<> mlx_to_dlpack(const mx::array& a) {
  return mlx_to_nd_array<>(a);
}

nb::object to_scalar(mx::array& a) {
  if (a.size() != 1) {
    throw std::invalid_argument(
        "[convert] Only length-1 arrays can be converted to Python scalars.");
  }
  {
    nb::gil_scoped_release nogil;
    a.eval();
  }
  switch (a.dtype()) {
    case mx::bool_:
      return nb::cast(a.item<bool>());
    case mx::uint8:
      return nb::cast(a.item<uint8_t>());
    case mx::uint16:
      return nb::cast(a.item<uint16_t>());
    case mx::uint32:
      return nb::cast(a.item<uint32_t>());
    case mx::uint64:
      return nb::cast(a.item<uint64_t>());
    case mx::int8:
      return nb::cast(a.item<int8_t>());
    case mx::int16:
      return nb::cast(a.item<int16_t>());
    case mx::int32:
      return nb::cast(a.item<int32_t>());
    case mx::int64:
      return nb::cast(a.item<int64_t>());
    case mx::float16:
      return nb::cast(static_cast<float>(a.item<mx::float16_t>()));
    case mx::float32:
      return nb::cast(a.item<float>());
    case mx::bfloat16:
      return nb::cast(static_cast<float>(a.item<mx::bfloat16_t>()));
    case mx::complex64:
      return nb::cast(a.item<std::complex<float>>());
    default:
      throw nb::type_error("type cannot be converted to Python scalar.");
  }
}

template <typename T, typename U = T>
nb::list to_list(mx::array& a, size_t index, int dim) {
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

nb::object tolist(mx::array& a) {
  if (a.ndim() == 0) {
    return to_scalar(a);
  }
  {
    nb::gil_scoped_release nogil;
    a.eval();
  }
  switch (a.dtype()) {
    case mx::bool_:
      return to_list<bool>(a, 0, 0);
    case mx::uint8:
      return to_list<uint8_t>(a, 0, 0);
    case mx::uint16:
      return to_list<uint16_t>(a, 0, 0);
    case mx::uint32:
      return to_list<uint32_t>(a, 0, 0);
    case mx::uint64:
      return to_list<uint64_t>(a, 0, 0);
    case mx::int8:
      return to_list<int8_t>(a, 0, 0);
    case mx::int16:
      return to_list<int16_t>(a, 0, 0);
    case mx::int32:
      return to_list<int32_t>(a, 0, 0);
    case mx::int64:
      return to_list<int64_t>(a, 0, 0);
    case mx::float16:
      return to_list<mx::float16_t, float>(a, 0, 0);
    case mx::float32:
      return to_list<float>(a, 0, 0);
    case mx::bfloat16:
      return to_list<mx::bfloat16_t, float>(a, 0, 0);
    case mx::complex64:
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
    const mx::Shape& shape,
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
    } else if (nb::isinstance<mx::array>(l)) {
      all_python_primitive_elements = false;
      auto arr = nb::cast<mx::array>(l);
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
        msg << "Invalid type " << nb::type_name(l.type()).c_str()
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
void get_shape(T list, mx::Shape& shape) {
  shape.push_back(check_shape_dim(nb::len(list)));
  if (shape.back() > 0) {
    auto l = list.begin();
    if (nb::isinstance<nb::list>(*l)) {
      return get_shape(nb::cast<nb::list>(*l), shape);
    } else if (nb::isinstance<nb::tuple>(*l)) {
      return get_shape(nb::cast<nb::tuple>(*l), shape);
    } else if (nb::isinstance<mx::array>(*l)) {
      auto arr = nb::cast<mx::array>(*l);
      for (int i = 0; i < arr.ndim(); i++) {
        shape.push_back(arr.shape(i));
      }
      return;
    }
  }
}

template <typename T>
mx::array array_from_list_impl(
    T pl,
    const PyScalarT& inferred_type,
    std::optional<mx::Dtype> specified_type,
    const mx::Shape& shape) {
  // Make the array
  switch (inferred_type) {
    case pybool: {
      std::vector<bool> vals;
      fill_vector(pl, vals);
      return mx::array(vals.begin(), shape, specified_type.value_or(mx::bool_));
    }
    case pyint: {
      auto dtype = specified_type.value_or(mx::int32);
      if (dtype == mx::int64) {
        std::vector<int64_t> vals;
        fill_vector(pl, vals);
        return mx::array(vals.begin(), shape, dtype);
      } else if (dtype == mx::uint64) {
        std::vector<uint64_t> vals;
        fill_vector(pl, vals);
        return mx::array(vals.begin(), shape, dtype);
      } else if (dtype == mx::uint32) {
        std::vector<uint32_t> vals;
        fill_vector(pl, vals);
        return mx::array(vals.begin(), shape, dtype);
      } else if (mx::issubdtype(dtype, mx::inexact)) {
        std::vector<float> vals;
        fill_vector(pl, vals);
        return mx::array(vals.begin(), shape, dtype);
      } else {
        std::vector<int> vals;
        fill_vector(pl, vals);
        return mx::array(vals.begin(), shape, dtype);
      }
    }
    case pyfloat: {
      std::vector<float> vals;
      fill_vector(pl, vals);
      return mx::array(
          vals.begin(), shape, specified_type.value_or(mx::float32));
    }
    case pycomplex: {
      std::vector<std::complex<float>> vals;
      fill_vector(pl, vals);
      return mx::array(
          reinterpret_cast<mx::complex64_t*>(vals.data()),
          shape,
          specified_type.value_or(mx::complex64));
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
mx::array array_from_list_impl(T pl, std::optional<mx::Dtype> dtype) {
  // Compute the shape
  mx::Shape shape;
  get_shape(pl, shape);

  // Validate the shape and type
  bool all_python_primitive_elements = true;
  auto type = validate_shape(pl, shape, 0, all_python_primitive_elements);

  if (all_python_primitive_elements) {
    // `pl` does not contain mlx arrays
    return array_from_list_impl(pl, type, dtype, shape);
  }

  // `pl` contains mlx arrays
  std::vector<mx::array> arrays;
  for (auto l : pl) {
    arrays.push_back(create_array(nb::cast<ArrayInitType>(l), dtype));
  }
  return mx::stack(arrays);
}

mx::array array_from_list(nb::list pl, std::optional<mx::Dtype> dtype) {
  return array_from_list_impl(pl, dtype);
}

mx::array array_from_list(nb::tuple pl, std::optional<mx::Dtype> dtype) {
  return array_from_list_impl(pl, dtype);
}

mx::array create_array(ArrayInitType v, std::optional<mx::Dtype> t) {
  if (auto pv = std::get_if<nb::bool_>(&v); pv) {
    return mx::array(nb::cast<bool>(*pv), t.value_or(mx::bool_));
  } else if (auto pv = std::get_if<nb::int_>(&v); pv) {
    auto val = nb::cast<long>(*pv);
    auto default_type = (val > std::numeric_limits<int>::max() ||
                         val < std::numeric_limits<int>::min())
        ? mx::int64
        : mx::int32;
    return mx::array(val, t.value_or(default_type));
  } else if (auto pv = std::get_if<nb::float_>(&v); pv) {
    return mx::array(nb::cast<float>(*pv), t.value_or(mx::float32));
  } else if (auto pv = std::get_if<std::complex<float>>(&v); pv) {
    return mx::array(
        static_cast<mx::complex64_t>(*pv), t.value_or(mx::complex64));
  } else if (auto pv = std::get_if<nb::list>(&v); pv) {
    return array_from_list(*pv, t);
  } else if (auto pv = std::get_if<nb::tuple>(&v); pv) {
    return array_from_list(*pv, t);
  } else if (auto pv = std::get_if<
                 nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>>(&v);
             pv) {
    return nd_array_to_mlx(*pv, t);
  } else if (auto pv = std::get_if<mx::array>(&v); pv) {
    return mx::astype(*pv, t.value_or((*pv).dtype()));
  } else {
    auto arr = to_array_with_accessor(std::get<ArrayLike>(v).obj);
    return mx::astype(arr, t.value_or(arr.dtype()));
  }
}
