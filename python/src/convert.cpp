// Copyright © 2024 Apple Inc.

#include <limits>
#include <memory>
#include <sstream>

#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>

#include "python/src/convert.h"
#include "python/src/utils.h"

#include "mlx/ops.h"
#include "mlx/utils.h"

enum PyScalarT {
  pybool = 0,
  pyint = 1,
  pyfloat = 2,
  pycomplex = 3,
};

int check_shape_dim(int64_t dim) {
  if (dim > std::numeric_limits<int>::max() ||
      dim < std::numeric_limits<int>::min()) {
    std::ostringstream msg;
    msg << "Shape dimension " << dim << " is outside the supported range ["
        << std::numeric_limits<int>::min() << ", "
        << std::numeric_limits<int>::max()
        << "]. MLX currently uses 32-bit integers for shape dimensions.";
    PyErr_SetString(PyExc_OverflowError, msg.str().c_str());
    nb::detail::raise_python_error();
  }
  return static_cast<int>(dim);
}

template <typename... NDParams>
mx::Shape get_shape(const nb::ndarray<NDParams...>& nd_array) {
  mx::Shape shape;
  shape.reserve(nd_array.ndim());
  for (int i = 0; i < nd_array.ndim(); i++) {
    shape.push_back(check_shape_dim(nd_array.shape(i)));
  }
  return shape;
}

template <typename T>
mx::array nd_array_to_mlx_contiguous(
    nb::ndarray<nb::ro, nb::c_contig> nd_array,
    const mx::Shape& shape,
    mx::Dtype dtype) {
  // Make a copy of the numpy buffer
  // Get buffer ptr pass to array constructor
  auto data_ptr = nd_array.data();
  return mx::array(static_cast<const T*>(data_ptr), shape, dtype);
}

template <typename F>
auto dispatch_dlpack_dtype(
    nb::dlpack::dtype type,
    F&& f,
    const char* error_message) {
  if (type == nb::dtype<bool>()) {
    return f.template operator()<bool>(mx::bool_);
  } else if (type == nb::dtype<uint8_t>()) {
    return f.template operator()<uint8_t>(mx::uint8);
  } else if (type == nb::dtype<uint16_t>()) {
    return f.template operator()<uint16_t>(mx::uint16);
  } else if (type == nb::dtype<uint32_t>()) {
    return f.template operator()<uint32_t>(mx::uint32);
  } else if (type == nb::dtype<uint64_t>()) {
    return f.template operator()<uint64_t>(mx::uint64);
  } else if (type == nb::dtype<int8_t>()) {
    return f.template operator()<int8_t>(mx::int8);
  } else if (type == nb::dtype<int16_t>()) {
    return f.template operator()<int16_t>(mx::int16);
  } else if (type == nb::dtype<int32_t>()) {
    return f.template operator()<int32_t>(mx::int32);
  } else if (type == nb::dtype<int64_t>()) {
    return f.template operator()<int64_t>(mx::int64);
  } else if (type == nb::dtype<mx::float16_t>()) {
    return f.template operator()<mx::float16_t>(mx::float16);
  } else if (type == nb::dtype<mx::bfloat16_t>()) {
    return f.template operator()<mx::bfloat16_t>(mx::bfloat16);
  } else if (type == nb::dtype<float>()) {
    return f.template operator()<float>(mx::float32);
  } else if (type == nb::dtype<double>()) {
    return f.template operator()<double>(mx::float32);
  } else if (type == nb::dtype<std::complex<float>>()) {
    return f.template operator()<mx::complex64_t>(mx::complex64);
  } else if (type == nb::dtype<std::complex<double>>()) {
    return f.template operator()<mx::complex128_t>(mx::complex64);
  } else {
    throw std::invalid_argument(error_message);
  }
}

mx::array metal_dlpack_to_mlx(
    nb::ndarray<nb::ro, nb::c_contig> nd_array,
    std::optional<mx::Dtype> dtype);

mx::array host_accessible_array(mx::array a) {
  a.eval();
  a.wait();
  if (a.buffer().is_host_accessible()) {
    return a;
  }
  auto out = mx::copy_to_new_buffer(std::move(a), mx::Device::gpu);
  out.eval();
  out.wait();
  out.detach();
  return out;
}

mx::array nd_array_to_mlx(
    nb::ndarray<nb::ro, nb::c_contig> nd_array,
    std::optional<mx::Dtype> dtype,
    std::optional<nb::dlpack::dtype> nb_dtype) {
  switch (nd_array.device_type()) {
    case nb::device::cpu::value: {
      auto shape = get_shape(nd_array);
      auto type = nb_dtype.value_or(nd_array.dtype());
      return dispatch_dlpack_dtype(
          type,
          [&]<typename T>(mx::Dtype default_dtype) {
            return nd_array_to_mlx_contiguous<T>(
                nd_array, shape, dtype.value_or(default_dtype));
          },
          "Cannot convert numpy array to mlx array.");
    }
    case nb::device::metal::value:
      return metal_dlpack_to_mlx(std::move(nd_array), dtype);
    default:
      throw std::invalid_argument("Unsupported DLPack device.");
  }
}

template <typename T>
mx::array metal_dlpack_to_mlx_contiguous(
    std::shared_ptr<nb::ndarray<nb::ro, nb::c_contig>> owner,
    const mx::Shape& shape,
    mx::Dtype type,
    std::optional<mx::Dtype> dtype) {
  auto itemsize = mx::size_of(type);
  if (owner->itemsize() != itemsize) {
    throw std::invalid_argument(
        "Cannot convert Metal DLPack dtype to mlx dtype.");
  }

  auto byte_offset = owner->data_offset();
  if (byte_offset % itemsize != 0) {
    throw std::invalid_argument(
        "Metal DLPack byte offset is not aligned to dtype size.");
  }

  auto out = mx::array(
      mx::allocator::Buffer(owner->data_handle()),
      shape,
      type,
      [](mx::allocator::Buffer) {});
  auto flags = out.flags();
  out.set_data(
      out.buffer(),
      out.data_size(),
      out.strides(),
      flags,
      [owner = std::move(owner)](mx::allocator::Buffer) {});

  auto offset = static_cast<int64_t>(byte_offset / itemsize);
  if (offset != 0) {
    out.copy_shared_buffer(out, out.strides(), flags, out.data_size(), offset);
  }

  if (dtype) {
    auto result = (*dtype == out.dtype())
        ? mx::copy_to_new_buffer(out, mx::Device::gpu)
        : mx::astype(out, *dtype, mx::Device::gpu);
    result.eval();
    result.wait();
    result.detach();
    return result;
  }
  return out;
}

template <typename T, typename... NDParams>
nb::ndarray<NDParams...> mlx_to_nd_array_impl(
    mx::array a,
    std::optional<nb::dlpack::dtype> t = {}) {
  {
    nb::gil_scoped_release nogil;
    a = host_accessible_array(std::move(a));
  }
  std::vector<size_t> shape(a.shape().begin(), a.shape().end());
  auto owner = nb::cast(a);
  return nb::ndarray<NDParams...>(
      a.data<T>(),
      a.ndim(),
      shape.data(),
      /* owner= */ owner,
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
    case mx::float64:
      return mlx_to_nd_array_impl<double, NDParams...>(a);
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
  auto host = mx::array(a);
  {
    nb::gil_scoped_release nogil;
    host = host_accessible_array(std::move(host));
  }
  switch (host.dtype()) {
    case mx::bool_:
      return nb::cast(host.item<bool>());
    case mx::uint8:
      return nb::cast(host.item<uint8_t>());
    case mx::uint16:
      return nb::cast(host.item<uint16_t>());
    case mx::uint32:
      return nb::cast(host.item<uint32_t>());
    case mx::uint64:
      return nb::cast(host.item<uint64_t>());
    case mx::int8:
      return nb::cast(host.item<int8_t>());
    case mx::int16:
      return nb::cast(host.item<int16_t>());
    case mx::int32:
      return nb::cast(host.item<int32_t>());
    case mx::int64:
      return nb::cast(host.item<int64_t>());
    case mx::float16:
      return nb::cast(static_cast<float>(host.item<mx::float16_t>()));
    case mx::float32:
      return nb::cast(host.item<float>());
    case mx::bfloat16:
      return nb::cast(static_cast<float>(host.item<mx::bfloat16_t>()));
    case mx::complex64:
      return nb::cast(host.item<std::complex<float>>());
    case mx::float64:
      return nb::cast(host.item<double>());
    default:
      throw nb::type_error("type cannot be converted to Python scalar.");
  }
}

mx::array metal_dlpack_to_mlx(
    nb::ndarray<nb::ro, nb::c_contig> nd_array,
    std::optional<mx::Dtype> dtype) {
  auto owner =
      std::make_shared<nb::ndarray<nb::ro, nb::c_contig>>(std::move(nd_array));
  auto shape = get_shape(*owner);

  return dispatch_dlpack_dtype(
      owner->dtype(),
      [&]<typename T>(mx::Dtype type) {
        return metal_dlpack_to_mlx_contiguous<T>(owner, shape, type, dtype);
      },
      "Cannot convert Metal DLPack array to mlx array.");
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
  auto host = mx::array(a);
  {
    nb::gil_scoped_release nogil;
    host = host_accessible_array(std::move(host));
  }
  switch (host.dtype()) {
    case mx::bool_:
      return to_list<bool>(host, 0, 0);
    case mx::uint8:
      return to_list<uint8_t>(host, 0, 0);
    case mx::uint16:
      return to_list<uint16_t>(host, 0, 0);
    case mx::uint32:
      return to_list<uint32_t>(host, 0, 0);
    case mx::uint64:
      return to_list<uint64_t>(host, 0, 0);
    case mx::int8:
      return to_list<int8_t>(host, 0, 0);
    case mx::int16:
      return to_list<int16_t>(host, 0, 0);
    case mx::int32:
      return to_list<int32_t>(host, 0, 0);
    case mx::int64:
      return to_list<int64_t>(host, 0, 0);
    case mx::float16:
      return to_list<mx::float16_t, float>(host, 0, 0);
    case mx::float32:
      return to_list<float>(host, 0, 0);
    case mx::bfloat16:
      return to_list<mx::bfloat16_t, float>(host, 0, 0);
    case mx::float64:
      return to_list<double>(host, 0, 0);
    case mx::complex64:
      return to_list<std::complex<float>>(host, 0, 0);
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
      auto out_type = specified_type.value_or(mx::float32);
      if (out_type == mx::float64) {
        std::vector<double> vals;
        fill_vector(pl, vals);
        return mx::array(vals.begin(), shape, out_type);
      } else {
        std::vector<float> vals;
        fill_vector(pl, vals);
        return mx::array(vals.begin(), shape, out_type);
      }
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
    arrays.push_back(create_array(nb::cast<nb::object>(l), dtype));
  }
  return mx::stack(arrays);
}

mx::array array_from_list(nb::list pl, std::optional<mx::Dtype> dtype) {
  return array_from_list_impl(pl, dtype);
}

mx::array array_from_list(nb::tuple pl, std::optional<mx::Dtype> dtype) {
  return array_from_list_impl(pl, dtype);
}

mx::array create_array(nb::object v, std::optional<mx::Dtype> t) {
  if (nb::isinstance<nb::bool_>(v)) {
    return mx::array(nb::cast<bool>(v), t.value_or(mx::bool_));
  } else if (nb::isinstance<nb::int_>(v)) {
    auto val = nb::cast<int64_t>(v);
    auto default_type = (val > std::numeric_limits<int>::max() ||
                         val < std::numeric_limits<int>::min())
        ? mx::int64
        : mx::int32;
    return mx::array(val, t.value_or(default_type));
  } else if (nb::isinstance<nb::float_>(v)) {
    auto out_type = t.value_or(mx::float32);
    if (out_type == mx::float64) {
      return mx::array(nb::cast<double>(v), out_type);
    } else {
      return mx::array(nb::cast<float>(v), out_type);
    }
  } else if (PyComplex_Check(v.ptr())) {
    return mx::array(
        static_cast<mx::complex64_t>(nb::cast<std::complex<float>>(v)),
        t.value_or(mx::complex64));
  } else if (nb::isinstance<nb::list>(v)) {
    return array_from_list(nb::cast<nb::list>(v), t);
  } else if (nb::isinstance<nb::tuple>(v)) {
    return array_from_list(nb::cast<nb::tuple>(v), t);
  } else if (nb::isinstance<mx::array>(v)) {
    auto arr = nb::cast<mx::array>(v);
    return mx::astype(arr, t.value_or(arr.dtype()));
  } else if (nb::ndarray_check(v)) {
    using ContigArray = nb::ndarray<nb::ro, nb::c_contig>;
    ContigArray nd;
    std::optional<nb::dlpack::dtype> nb_dtype;
    // Nanobind does not recognize bfloat16 numpy array:
    // https://github.com/wjakob/nanobind/discussions/560
    if (nb::hasattr(v, "dtype") && v.attr("dtype").equal(nb::str("bfloat16"))) {
      nd = nb::cast<ContigArray>(v.attr("view")("uint16"));
      nb_dtype = nb::dtype<mx::bfloat16_t>();
    } else {
      nd = nb::cast<ContigArray>(v);
    }
    return nd_array_to_mlx(nd, t, nb_dtype);
  } else {
    auto arr = to_array_with_accessor(v);
    return mx::astype(arr, t.value_or(arr.dtype()));
  }
}
