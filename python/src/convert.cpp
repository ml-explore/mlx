// Copyright © 2024 Apple Inc.

#include <algorithm>
#include <limits>
#include <sstream>
#include <tuple>

#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>

#include "python/src/convert.h"
#include "python/src/utils.h"

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/cuda.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/dtype_utils.h"
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

mx::Shape get_shape(const nb::ndarray<nb::ro>& nd_array) {
  mx::Shape shape;
  shape.reserve(nd_array.ndim());
  for (int i = 0; i < nd_array.ndim(); i++) {
    shape.push_back(check_shape_dim(nd_array.shape(i)));
  }
  return shape;
}

mx::Strides get_strides(const nb::ndarray<nb::ro>& nd_array) {
  mx::Strides strides;
  strides.reserve(nd_array.ndim());
  for (int i = 0; i < nd_array.ndim(); i++) {
    strides.push_back(nd_array.stride(i));
  }
  return strides;
}

size_t strided_storage_size(
    const mx::Shape& shape,
    const mx::Strides& strides) {
  size_t storage_size = 1;
  for (int i = 0; i < shape.size(); i++) {
    if (shape[i] == 0) {
      return 0;
    }
    if (strides[i] < 0) {
      throw std::invalid_argument(
          "Cannot convert DLPack arrays with negative strides to mlx array.");
    }
    storage_size += (shape[i] - 1) * strides[i];
  }
  return storage_size;
}

auto get_strided_layout(
    const nb::ndarray<nb::ro>& nd_array,
    const mx::Shape& shape) {
  auto strides = get_strides(nd_array);
  auto storage_size = strided_storage_size(shape, strides);
  auto [no_bsx_size, is_row_contiguous, is_col_contiguous] = shape.empty()
      ? std::make_tuple(storage_size, true, true)
      : mx::check_contiguity(shape, strides);
  mx::array::Flags flags{
      no_bsx_size == storage_size,
      is_row_contiguous,
      is_col_contiguous,
  };
  return std::make_tuple(storage_size, std::move(strides), flags);
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

mx::Dtype mlx_dtype_from_dlpack(
    nb::dlpack::dtype type,
    const char* error_message) {
  return dispatch_dlpack_dtype(
      type, []<typename T>(mx::Dtype dtype) { return dtype; }, error_message);
}

nb::dlpack::dtype mlx_dtype_to_dl_dtype(mx::Dtype dtype) {
  nb::dlpack::dtype result;
  dispatch_all_types(dtype, [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    result = nb::dtype<T>();
  });
  return result;
}

template <typename SrcT>
mx::array cpu_nd_array_to_mlx(
    nb::ndarray<nb::ro> nd_array,
    const mx::Shape& shape,
    mx::Dtype dst_dtype) {
  auto out = mx::array(shape, dst_dtype, nullptr, {});
  auto [storage_size, strides, flags] = get_strided_layout(nd_array, shape);
  out.set_data(
      mx::allocator::malloc(storage_size * mx::size_of(dst_dtype)),
      storage_size,
      std::move(strides),
      flags);
  if (storage_size > 0) {
    dispatch_all_types(dst_dtype, [&](auto type_tag) {
      using DstT = MLX_GET_TYPE(type_tag);
      auto src = static_cast<const SrcT*>(nd_array.data());
      auto dst = out.data<DstT>();
      std::copy(src, src + storage_size, dst);
    });
  }
  out.set_status(mx::array::Status::available);
  return out;
}

mx::array metal_nd_array_to_mlx(
    nb::ndarray<nb::ro> nd_array,
    mx::Dtype src_dtype,
    mx::Dtype dst_dtype,
    bool copy) {
  if (!mx::metal::is_available()) {
    throw std::invalid_argument("Metal DLPack import is not available.");
  }
  auto shape = get_shape(nd_array);
  if (nd_array.itemsize() != mx::size_of(src_dtype)) {
    throw std::invalid_argument(
        "Cannot convert Metal DLPack dtype to mlx dtype.");
  }
  auto [storage_size, strides, flags] = get_strided_layout(nd_array, shape);
  auto data_handle = nd_array.data_handle();
  mx::array out(shape, src_dtype, nullptr, {});
  out.set_data(
      mx::allocator::Buffer(data_handle),
      storage_size,
      std::move(strides),
      flags,
      nd_array.byte_offset(),
      [owner = std::move(nd_array)](mx::allocator::Buffer) {});
  out.set_status(mx::array::Status::available);

  if (copy) {
    auto result = mx::astype(out, dst_dtype, true, mx::Device::gpu);
    result.eval();
    return result;
  }
  return out;
}

mx::array nd_array_to_mlx(
    nb::ndarray<nb::ro> nd_array,
    std::optional<mx::Dtype> requested_dtype,
    std::optional<nb::dlpack::dtype> src_dlpack_dtype_override,
    std::optional<bool> copy) {
  auto src_dlpack_dtype = src_dlpack_dtype_override.value_or(nd_array.dtype());
  auto src_mlx_dtype =
      mlx_dtype_from_dlpack(src_dlpack_dtype, "Cannot convert array to mlx.");
  auto dst_dtype = requested_dtype.value_or(src_mlx_dtype);
  auto device_type = nd_array.device_type();
  // CPU ndarrays are copied below, and their data_handle() is a host pointer,
  // not a GPU buffer handle that can be queried by the active allocator.
  bool can_reuse_buffer = device_type == nb::device::cpu::value
      ? true
      : mx::allocator::can_reuse_alien_buffer(nd_array.data_handle());
  bool should_copy =
      copy.value_or(false) || dst_dtype != src_mlx_dtype || !can_reuse_buffer;
  if (copy.has_value() && copy.value() == false && dst_dtype != src_mlx_dtype) {
    throw std::invalid_argument(
        "Cannot convert DLPack array to requested dtype without a copy.");
  }
  switch (device_type) {
    case nb::device::cpu::value: {
      if (copy.has_value() && copy.value() == false) {
        throw std::invalid_argument(
            "Cannot import a CPU DLPack array without a copy.");
      }
      auto shape = get_shape(nd_array);
      return dispatch_dlpack_dtype(
          src_dlpack_dtype,
          [&]<typename T>(mx::Dtype src_dtype) {
            return cpu_nd_array_to_mlx<T>(nd_array, shape, dst_dtype);
          },
          "Cannot convert numpy array to mlx array.");
    }
    case nb::device::metal::value: {
      if (copy.has_value() && copy.value() == false && !can_reuse_buffer) {
        throw std::invalid_argument(
            "Cannot import a private Metal DLPack buffer without a copy.");
      }
      return metal_nd_array_to_mlx(
          nd_array, src_mlx_dtype, dst_dtype, should_copy);
    }
    case nb::device::cuda::value:
    case nb::device::cuda_managed::value:
      throw std::invalid_argument("CUDA DLPack import is not supported.");
    default:
      throw std::invalid_argument("Unsupported DLPack device.");
  }
}

template <typename... NDParams>
nb::ndarray<NDParams...> mlx_to_nd_array(
    const mx::array& a,
    std::optional<std::tuple<int, int>> dl_device) {
  auto default_device = mx::metal::is_available()
      ? std::tuple{nb::device::metal::value, 0}
      : std::tuple{nb::device::cpu::value, 0};
  auto [device_type, device_id] = dl_device.value_or(default_device);

  if (device_type == nb::device::cuda::value ||
      device_type == nb::device::cuda_managed::value) {
    throw nb::buffer_error("CUDA DLPack export is not supported.");
  }
  if (device_type != nb::device::cpu::value &&
      device_type != nb::device::metal::value) {
    throw nb::buffer_error(
        "Cannot export mlx array to requested DLPack device.");
  }
  if (device_type == nb::device::metal::value && !mx::metal::is_available()) {
    throw nb::buffer_error("Metal DLPack export is not available.");
  }

  auto arr = a;
  void* data = nullptr;
  uint64_t byte_offset = 0;
  {
    nb::gil_scoped_release nogil;
    arr.eval();
  }
  data = device_type == nb::device::cpu::value ? arr.buffer().raw_ptr()
                                               : arr.buffer().ptr();
  byte_offset = arr.offset();

  std::vector<size_t> shape(arr.shape().begin(), arr.shape().end());
  auto owner = nb::cast(arr);
  return nb::ndarray<NDParams...>(
      data,
      arr.ndim(),
      shape.data(),
      /* owner= */ owner,
      arr.strides().data(),
      mlx_dtype_to_dl_dtype(arr.dtype()),
      device_type,
      device_id,
      '\0',
      byte_offset);
}

nb::ndarray<nb::numpy> mlx_to_np_array(const mx::array& a) {
  if (a.dtype() == mx::bfloat16) {
    throw nb::type_error("bfloat16 arrays cannot be converted to NumPy.");
  }
  return mlx_to_nd_array<nb::numpy>(a, std::tuple{nb::device::cpu::value, 0});
}

nb::ndarray<> mlx_to_dlpack(
    const mx::array& a,
    std::optional<std::tuple<int, int>> dl_device) {
  return mlx_to_nd_array<>(a, dl_device);
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
    case mx::float64:
      return nb::cast(a.item<double>());
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
    case mx::float64:
      return to_list<double>(a, 0, 0);
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

mx::array create_array(
    nb::object v,
    std::optional<mx::Dtype> t,
    std::optional<bool> copy) {
  if (!nb::isinstance<mx::array>(v) && nb::ndarray_check(v)) {
    using ContigArray = nb::ndarray<nb::ro>;
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
    return nd_array_to_mlx(nd, t, nb_dtype, copy);
  }

  if (copy.has_value() && copy.value() == false) {
    throw std::invalid_argument(
        "Unable to avoid copy while creating an array as requested.");
  }

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
    auto dtype = t.value_or(arr.dtype());
    return mx::astype(arr, dtype, copy);
  } else {
    auto arr = to_array_with_accessor(v);
    return mx::astype(arr, t.value_or(arr.dtype()), copy);
  }
}
