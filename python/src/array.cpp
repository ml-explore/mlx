// Copyright © 2023 Apple Inc.

#include <cstdint>
#include <cstring>
#include <sstream>

#include <pybind11/numpy.h>

#include "python/src/indexing.h"
#include "python/src/utils.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

namespace py = pybind11;
using namespace py::literals;

enum PyScalarT {
  pybool = 0,
  pyint = 1,
  pyfloat = 2,
  pycomplex = 3,
};

template <typename T>
py::list to_list(array& a, size_t index, int dim) {
  py::list pl;
  auto stride = a.strides()[dim];
  for (int i = 0; i < a.shape(dim); ++i) {
    if (dim == a.ndim() - 1) {
      pl.append((a.data<T>()[index]));
    } else {
      pl.append(to_list<T>(a, index, dim + 1));
    }
    index += stride;
  }
  return pl;
}

auto to_scalar(array& a) {
  bool retain_graph = a.is_tracer();
  switch (a.dtype()) {
    case bool_:
      return py::cast(a.item<bool>(retain_graph));
    case uint8:
      return py::cast(a.item<uint8_t>(retain_graph));
    case uint16:
      return py::cast(a.item<uint16_t>(retain_graph));
    case uint32:
      return py::cast(a.item<uint32_t>(retain_graph));
    case uint64:
      return py::cast(a.item<uint64_t>(retain_graph));
    case int8:
      return py::cast(a.item<int8_t>(retain_graph));
    case int16:
      return py::cast(a.item<int16_t>(retain_graph));
    case int32:
      return py::cast(a.item<int32_t>(retain_graph));
    case int64:
      return py::cast(a.item<int64_t>(retain_graph));
    case float16:
      return py::cast(static_cast<float>(a.item<float16_t>(retain_graph)));
    case float32:
      return py::cast(a.item<float>(retain_graph));
    case bfloat16:
      return py::cast(static_cast<float>(a.item<float16_t>(retain_graph)));
    case complex64:
      return py::cast(a.item<std::complex<float>>(retain_graph));
  }
}

py::object tolist(array& a) {
  if (a.ndim() == 0) {
    return to_scalar(a);
  }
  a.eval(a.is_tracer());
  py::object pl;
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
      return to_list<float16_t>(a, 0, 0);
    case float32:
      return to_list<float>(a, 0, 0);
    case bfloat16:
      return to_list<float16_t>(a, 0, 0);
    case complex64:
      return to_list<std::complex<float>>(a, 0, 0);
  }
}

template <typename T, typename U>
void fill_vector(T list, std::vector<U>& vals) {
  for (auto l : list) {
    if (py::isinstance<py::list>(l)) {
      fill_vector(l.template cast<py::list>(), vals);
    } else if (py::isinstance<py::tuple>(*list.begin())) {
      fill_vector(l.template cast<py::tuple>(), vals);
    } else {
      vals.push_back(l.template cast<U>());
    }
  }
}

template <typename T>
PyScalarT validate_shape(T list, const std::vector<int>& shape, int idx) {
  if (idx >= shape.size()) {
    throw std::invalid_argument("Initialization encountered extra dimension.");
  }
  auto s = shape[idx];
  if (py::len(list) != s) {
    throw std::invalid_argument(
        "Initialization encountered non-uniform length.");
  }

  if (s == 0) {
    return pyfloat;
  }

  PyScalarT type = pybool;
  for (auto l : list) {
    PyScalarT t;
    if (py::isinstance<py::list>(l)) {
      t = validate_shape(l.template cast<py::list>(), shape, idx + 1);
    } else if (py::isinstance<py::tuple>(*list.begin())) {
      t = validate_shape(l.template cast<py::tuple>(), shape, idx + 1);
    } else if (py::isinstance<py::bool_>(l)) {
      t = pybool;
    } else if (py::isinstance<py::int_>(l)) {
      t = pyint;
    } else if (py::isinstance<py::float_>(l)) {
      t = pyfloat;
    } else if (PyComplex_Check(l.ptr())) {
      t = pycomplex;
    } else {
      std::ostringstream msg;
      msg << "Invalid type in array initialization" << l.get_type() << ".";
      throw std::invalid_argument(msg.str());
    }
    type = std::max(type, t);
  }
  return type;
}

template <typename T>
void get_shape(T list, std::vector<int>& shape) {
  shape.push_back(py::len(list));
  if (shape.back() > 0) {
    auto& l = *list.begin();
    if (py::isinstance<py::list>(l)) {
      return get_shape(l.template cast<py::list>(), shape);
    } else if (py::isinstance<py::tuple>(l)) {
      return get_shape(l.template cast<py::tuple>(), shape);
    }
  }
}

template <typename T>
array array_from_list(T pl, std::optional<Dtype> dtype) {
  // Compute the shape
  std::vector<int> shape;
  get_shape(pl, shape);

  // Validate the shape and type
  auto type = validate_shape(pl, shape, 0);

  size_t size = 1;
  for (auto s : shape) {
    size *= s;
  }

  // Make the array
  switch (type) {
    case pybool: {
      std::vector<bool> vals;
      fill_vector(pl, vals);
      return array(vals.begin(), shape, dtype.value_or(bool_));
    }
    case pyint: {
      std::vector<int> vals;
      fill_vector(pl, vals);
      return array(vals.begin(), shape, dtype.value_or(int32));
    }
    case pyfloat: {
      std::vector<float> vals;
      fill_vector(pl, vals);
      return array(vals.begin(), shape, dtype.value_or(float32));
    }
    case pycomplex: {
      std::vector<std::complex<float>> vals;
      fill_vector(pl, vals);
      return array(
          reinterpret_cast<complex64_t*>(vals.data()),
          shape,
          dtype.value_or(complex64));
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// MLX -> Numpy
///////////////////////////////////////////////////////////////////////////////

size_t elem_to_loc(
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

struct PyArrayPayload {
  array a;
};

template <typename T>
py::array_t<T> mlx_array_to_np_t(const array& src) {
  // Let py::capsule hold onto a copy of the array which holds a shared ptr to
  // the data
  const py::capsule freeWhenDone(new PyArrayPayload({src}), [](void* payload) {
    delete reinterpret_cast<PyArrayPayload*>(payload);
  });
  // Collect strides
  std::vector<size_t> strides{src.strides().begin(), src.strides().end()};
  for (int i = 0; i < src.ndim(); i++) {
    strides[i] *= src.itemsize();
  }
  // Pack the capsule with the array
  py::array_t<T> out(src.shape(), strides, src.data<T>(), freeWhenDone);
  // Mark array as read-only
  py::detail::array_proxy(out.ptr())->flags &=
      ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
  // Return array
  return py::array_t(src.shape(), strides, src.data<T>(), out);
}

template <typename T>
py::array mlx_array_to_np_t(const array& src, const py::dtype& dt) {
  // Let py::capsule hold onto a copy of the array which holds a shared ptr to
  // the data
  const py::capsule freeWhenDone(new PyArrayPayload({src}), [](void* payload) {
    delete reinterpret_cast<PyArrayPayload*>(payload);
  });
  // Collect strides
  std::vector<size_t> strides{src.strides().begin(), src.strides().end()};
  for (int i = 0; i < src.ndim(); i++) {
    strides[i] *= src.itemsize();
  }
  // Pack the capsule with the array
  py::array out(dt, src.shape(), strides, src.data<T>(), freeWhenDone);
  // Mark array as read-only
  py::detail::array_proxy(out.ptr())->flags &=
      ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
  // Return array
  return py::array(dt, src.shape(), strides, src.data<T>(), out);
}

py::array mlx_array_to_np(const array& src) {
  // Eval if not already evaled
  if (!src.is_evaled()) {
    eval({src}, src.is_tracer());
  }

  switch (src.dtype()) {
    case bool_:
      return mlx_array_to_np_t<bool>(src);
    case uint8:
      return mlx_array_to_np_t<uint8_t>(src);
    case uint16:
      return mlx_array_to_np_t<uint16_t>(src);
    case uint32:
      return mlx_array_to_np_t<uint32_t>(src);
    case uint64:
      return mlx_array_to_np_t<uint64_t>(src);
    case int8:
      return mlx_array_to_np_t<int8_t>(src);
    case int16:
      return mlx_array_to_np_t<int16_t>(src);
    case int32:
      return mlx_array_to_np_t<int32_t>(src);
    case int64:
      return mlx_array_to_np_t<int64_t>(src);
    case float16:
      return mlx_array_to_np_t<float16_t>(src, py::dtype("float16"));
    case float32:
      return mlx_array_to_np_t<float>(src);
    case bfloat16: {
      auto a = astype(src, float32);
      eval({a}, src.is_tracer());
      return mlx_array_to_np_t<float>(a);
    }
    case complex64:
      return mlx_array_to_np_t<complex64_t>(src, py::dtype("complex64"));
  }
}

///////////////////////////////////////////////////////////////////////////////
// Numpy -> MLX
///////////////////////////////////////////////////////////////////////////////

template <typename T>
array np_array_to_mlx_contiguous(
    py::array_t<T, py::array::c_style | py::array::forcecast> np_array,
    const std::vector<int>& shape,
    Dtype dtype) {
  // Make a copy of the numpy buffer
  // Get buffer ptr pass to array constructor
  py::buffer_info buf = np_array.request();
  const T* data_ptr = static_cast<T*>(buf.ptr);
  return array(data_ptr, shape, dtype);

  // Note: Leaving the following memoryless copy from np to mx commented
  // out for the time being since it is unsafe given that the incoming
  // numpy array may change the underlying data

  // // Share underlying numpy buffer
  // // Copy to increase ref count
  // auto deleter = [np_array](void*) {};
  // void* data_ptr = np_array.mutable_data();
  // // Use buffer from numpy
  // return array(data_ptr, deleter, shape, dtype);
}

template <>
array np_array_to_mlx_contiguous(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast>
        np_array,
    const std::vector<int>& shape,
    Dtype dtype) {
  // Get buffer ptr pass to array constructor
  py::buffer_info buf = np_array.request();
  auto data_ptr = static_cast<std::complex<float>*>(buf.ptr);
  return array(reinterpret_cast<complex64_t*>(data_ptr), shape, dtype);
}

array np_array_to_mlx(py::array np_array, std::optional<Dtype> dtype) {
  // Compute the shape and size
  std::vector<int> shape;
  for (int i = 0; i < np_array.ndim(); i++) {
    shape.push_back(np_array.shape(i));
  }

  // Get dtype
  auto type = np_array.dtype();

  // Copy data and make array
  if (type.is(py::dtype::of<int>())) {
    return np_array_to_mlx_contiguous<int32_t>(
        np_array, shape, dtype.value_or(int32));
  } else if (type.is(py::dtype::of<uint32_t>())) {
    return np_array_to_mlx_contiguous<uint32_t>(
        np_array, shape, dtype.value_or(uint32));
  } else if (type.is(py::dtype::of<bool>())) {
    return np_array_to_mlx_contiguous<bool>(
        np_array, shape, dtype.value_or(bool_));
  } else if (type.is(py::dtype::of<double>())) {
    return np_array_to_mlx_contiguous<double>(
        np_array, shape, dtype.value_or(float32));
  } else if (type.is(py::dtype::of<float>())) {
    return np_array_to_mlx_contiguous<float>(
        np_array, shape, dtype.value_or(float32));
  } else if (type.is(py::dtype("float16"))) {
    return np_array_to_mlx_contiguous<float>(
        np_array, shape, dtype.value_or(float16));
  } else if (type.is(py::dtype::of<uint8_t>())) {
    return np_array_to_mlx_contiguous<uint8_t>(
        np_array, shape, dtype.value_or(uint8));
  } else if (type.is(py::dtype::of<uint16_t>())) {
    return np_array_to_mlx_contiguous<uint16_t>(
        np_array, shape, dtype.value_or(uint16));
  } else if (type.is(py::dtype::of<uint64_t>())) {
    return np_array_to_mlx_contiguous<uint64_t>(
        np_array, shape, dtype.value_or(uint64));
  } else if (type.is(py::dtype::of<int8_t>())) {
    return np_array_to_mlx_contiguous<int8_t>(
        np_array, shape, dtype.value_or(int8));
  } else if (type.is(py::dtype::of<int16_t>())) {
    return np_array_to_mlx_contiguous<int16_t>(
        np_array, shape, dtype.value_or(int16));
  } else if (type.is(py::dtype::of<int64_t>())) {
    return np_array_to_mlx_contiguous<int64_t>(
        np_array, shape, dtype.value_or(int64));
  } else if (type.is(py::dtype::of<std::complex<float>>())) {
    return np_array_to_mlx_contiguous<std::complex<float>>(
        np_array, shape, dtype.value_or(complex64));
  } else if (type.is(py::dtype::of<std::complex<double>>())) {
    return np_array_to_mlx_contiguous<std::complex<float>>(
        np_array, shape, dtype.value_or(complex64));
  } else {
    std::ostringstream msg;
    msg << "Cannot convert numpy array of type " << type << " to mlx array.";
    throw std::invalid_argument(msg.str());
  }
}

///////////////////////////////////////////////////////////////////////////////
// Module
///////////////////////////////////////////////////////////////////////////////

void init_array(py::module_& m) {
  // Types
  py::class_<Dtype>(
      m,
      "Dtype",
      R"pbdoc(
      An object to hold the type of a :class:`array`.

      See the :ref:`list of types <data_types>` for more details
      on available data types.
      )pbdoc")
      .def_readonly(
          "size", &Dtype::size, R"pbdoc(Size of the type in bytes.)pbdoc")
      .def(
          "__repr__",
          [](const Dtype& t) {
            std::ostringstream os;
            os << "mlx.core.";
            os << t;
            return os.str();
          })
      .def("__eq__", [](const Dtype& t1, const Dtype& t2) { return t1 == t2; })
      .def("__hash__", [](const Dtype& t) {
        return static_cast<int64_t>(t.val);
      });
  m.attr("bool_") = py::cast(bool_);
  m.attr("uint8") = py::cast(uint8);
  m.attr("uint16") = py::cast(uint16);
  m.attr("uint32") = py::cast(uint32);
  m.attr("uint64") = py::cast(uint64);
  m.attr("int8") = py::cast(int8);
  m.attr("int16") = py::cast(int16);
  m.attr("int32") = py::cast(int32);
  m.attr("int64") = py::cast(int64);
  m.attr("float16") = py::cast(float16);
  m.attr("float32") = py::cast(float32);
  m.attr("bfloat16") = py::cast(bfloat16);
  m.attr("complex64") = py::cast(complex64);

  auto array_class = py::class_<array>(
      m, "array", R"pbdoc(An N-dimensional array object.)pbdoc");

  {
    py::options options;
    options.disable_function_signatures();

    array_class.def(
        py::init([](std::variant<
                        py::bool_,
                        py::int_,
                        py::float_,
                        std::complex<float>,
                        py::list,
                        py::tuple,
                        py::array,
                        py::buffer,
                        py::object> v,
                    std::optional<Dtype> t) {
          if (auto pv = std::get_if<py::bool_>(&v); pv) {
            return array(py::cast<bool>(*pv), t.value_or(bool_));
          } else if (auto pv = std::get_if<py::int_>(&v); pv) {
            return array(py::cast<int>(*pv), t.value_or(int32));
          } else if (auto pv = std::get_if<py::float_>(&v); pv) {
            return array(py::cast<float>(*pv), t.value_or(float32));
          } else if (auto pv = std::get_if<std::complex<float>>(&v); pv) {
            return array(static_cast<complex64_t>(*pv), t.value_or(complex64));
          } else if (auto pv = std::get_if<py::list>(&v); pv) {
            return array_from_list(*pv, t);
          } else if (auto pv = std::get_if<py::tuple>(&v); pv) {
            return array_from_list(*pv, t);
          } else if (auto pv = std::get_if<py::array>(&v); pv) {
            return np_array_to_mlx(*pv, t);
          } else if (auto pv = std::get_if<py::buffer>(&v); pv) {
            return np_array_to_mlx(*pv, t);
          } else {
            auto arr = to_array_with_accessor(std::get<py::object>(v));
            return astype(arr, t.value_or(arr.dtype()));
          }
        }),
        "val"_a,
        "dtype"_a = std::nullopt,
        R"pbdoc(
            __init__(self: array, val: Union[scalar, list, tuple, numpy.ndarray, array], dtype: Optional[Dtype] = None)
          )pbdoc");
  }

  array_class
      .def_property_readonly(
          "size", &array::size, R"pbdoc(Number of elments in the array.)pbdoc")
      .def_property_readonly(
          "ndim", &array::ndim, R"pbdoc(The array's dimension.)pbdoc")
      // TODO, this makes a deep copy of the shape
      // implement alternatives to use reference
      // https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html
      .def_property_readonly(
          "shape",
          [](const array& a) { return a.shape(); },
          R"pbdoc(
          The shape of the array as a Python list.

          Returns:
            list(int): A list containing the sizes of each dimension.
        )pbdoc")
      .def_property_readonly(
          "dtype",
          &array::dtype,
          R"pbdoc(
            The array's :class:`Dtype`.
          )pbdoc")
      .def(
          "item",
          &to_scalar,
          R"pbdoc(
            Access the value of a scalar array.

            Returns:
                Standard Python scalar.
          )pbdoc")
      .def(
          "tolist",
          &tolist,
          R"pbdoc(
            Convert the array to a Python :class:`list`.

            Returns:
                list: The Python list.

                If the array is a scalar then a standard Python scalar is returned.

                If the array has more than one dimension then the result is a nested
                list of lists.

                The value type of the list correpsonding to the last dimension is either
                ``bool``, ``int`` or ``float`` depending on the ``dtype`` of the array.
          )pbdoc")
      .def("__array__", &mlx_array_to_np)
      .def(
          "astype",
          &astype,
          "dtype"_a,
          "stream"_a = none,
          R"pbdoc(
            Cast the array to a specified type.

            Args:
                dtype (Dtype): Type to which the array is cast.
                stream (Stream): Stream (or device) for the operation.

            Returns:
                array: The array with type ``dtype``.
          )pbdoc")
      .def("__getitem__", mlx_get_item)
      .def("__setitem__", mlx_set_item)
      .def(
          "__len__",
          [](const array& a) {
            if (a.ndim() == 0) {
              throw py::type_error("len() 0-dimensional array.");
            }
            return a.shape(0);
          })
      .def(
          "__iter__",
          [](const array& a) { return py::make_iterator(a); },
          py::keep_alive<0, 1>())
      .def(
          "__add__",
          [](const array& a, const ScalarOrArray v) {
            return add(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__radd__",
          [](const array& a, const ScalarOrArray v) {
            return add(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__sub__",
          [](const array& a, const ScalarOrArray v) {
            return subtract(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__rsub__",
          [](const array& a, const ScalarOrArray v) {
            return subtract(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__mul__",
          [](const array& a, const ScalarOrArray v) {
            return multiply(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__rmul__",
          [](const array& a, const ScalarOrArray v) {
            return multiply(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__truediv__",
          [](const array& a, const ScalarOrArray v) {
            return divide(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__div__",
          [](const array& a, const ScalarOrArray v) {
            return divide(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__floordiv__",
          [](const array& a, const ScalarOrArray v) {
            auto b = to_array(v, a.dtype());
            return floor_divide(a, b);
          },
          "other"_a)
      .def(
          "__rtruediv__",
          [](const array& a, const ScalarOrArray v) {
            return divide(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__rfloordiv__",
          [](const array& a, const ScalarOrArray v) {
            auto b = to_array(v, a.dtype());
            return floor_divide(b, a);
          },
          "other"_a)
      .def(
          "__rdiv__",
          [](const array& a, const ScalarOrArray v) {
            return divide(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__mod__",
          [](const array& a, const ScalarOrArray v) {
            return remainder(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__rmod__",
          [](const array& a, const ScalarOrArray v) {
            return remainder(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__eq__",
          [](const array& a, const ScalarOrArray v) {
            return equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__lt__",
          [](const array& a, const ScalarOrArray v) {
            return less(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__le__",
          [](const array& a, const ScalarOrArray v) {
            return less_equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__gt__",
          [](const array& a, const ScalarOrArray v) {
            return greater(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__ge__",
          [](const array& a, const ScalarOrArray v) {
            return greater_equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__ne__",
          [](const array& a, const ScalarOrArray v) {
            return not_equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def("__neg__", [](const array& a) { return -a; })
      .def("__bool__", [](array& a) { return py::bool_(to_scalar(a)); })
      .def(
          "__repr__",
          [](array& a) {
            if (!a.is_evaled()) {
              a.eval(a.is_tracer());
            }
            std::ostringstream os;
            os << a;
            return os.str();
          })
      .def(
          "__matmul__", [](array& a, array& other) { return matmul(a, other); })
      .def(
          "__pow__",
          [](const array& a, const ScalarOrArray v) {
            return power(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "flatten",
          [](const array& a,
             int start_axis,
             int end_axis,
             const StreamOrDevice& s) {
            return flatten(a, start_axis, end_axis);
          },
          "start_axis"_a = 0,
          "end_axis"_a = -1,
          py::kw_only(),
          "stream"_a = none,
          R"pbdoc(
            See :func:`flatten`.
          )pbdoc")
      .def(
          "reshape",
          [](const array& a, py::args shape, StreamOrDevice s) {
            if (shape.size() == 1) {
              py::object arg = shape[0];
              if (!py::isinstance<py::int_>(arg)) {
                return reshape(a, py::cast<std::vector<int>>(arg), s);
              }
            }
            return reshape(a, py::cast<std::vector<int>>(shape), s);
          },
          py::kw_only(),
          "stream"_a = none,
          R"pbdoc(
            Equivalent to :func:`reshape` but the shape can be passed either as a
            tuple or as separate arguments.

            See :func:`reshape` for full documentation.
          )pbdoc")
      .def(
          "squeeze",
          [](const array& a, const IntOrVec& v, const StreamOrDevice& s) {
            if (std::holds_alternative<std::monostate>(v)) {
              return squeeze(a, s);
            } else if (auto pv = std::get_if<int>(&v); pv) {
              return squeeze(a, *pv, s);
            } else {
              return squeeze(a, std::get<std::vector<int>>(v), s);
            }
          },
          "axis"_a = none,
          py::kw_only(),
          "stream"_a = none,
          R"pbdoc(
            See :func:`squeeze`.
          )pbdoc")
      .def(
          "abs",
          &mlx::core::abs,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`abs`.")
      .def(
          "square",
          &square,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`square`.")
      .def(
          "sqrt",
          &mlx::core::sqrt,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`sqrt`.")
      .def(
          "rsqrt",
          &rsqrt,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`rsqrt`.")
      .def(
          "reciprocal",
          &reciprocal,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`reciprocal`.")
      .def(
          "exp",
          &mlx::core::exp,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`exp`.")
      .def(
          "log",
          &mlx::core::log,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`log`.")
      .def(
          "log2",
          &mlx::core::log2,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`log2`.")
      .def(
          "log10",
          &mlx::core::log10,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`log10`.")
      .def(
          "sin",
          &mlx::core::sin,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`sin`.")
      .def(
          "cos",
          &mlx::core::cos,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`cos`.")
      .def(
          "log1p",
          &mlx::core::log1p,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`log1p`.")
      .def(
          "all",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return all(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = none,
          "keepdims"_a = false,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`all`.")
      .def(
          "any",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return any(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = none,
          "keepdims"_a = false,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`any`.")
      .def(
          "moveaxis",
          &moveaxis,
          "source"_a,
          "destination"_a,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`moveaxis`.")
      .def(
          "swapaxes",
          &swapaxes,
          "axis1"_a,
          "axis2"_a,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`moveaxis`.")
      .def(
          "transpose",
          [](const array& a, py::args axes, StreamOrDevice s) {
            if (axes.size() > 0) {
              if (axes.size() == 1) {
                py::object arg = axes[0];
                if (!py::isinstance<py::int_>(arg)) {
                  return transpose(a, py::cast<std::vector<int>>(arg), s);
                }
              }
              return transpose(a, py::cast<std::vector<int>>(axes), s);
            } else {
              return transpose(a, s);
            }
          },
          py::kw_only(),
          "stream"_a = none,
          R"pbdoc(
            Equivalent to :func:`transpose` but the axes can be passed either as
            a tuple or as separate arguments.

            See :func:`transpose` for full documentation.
          )pbdoc")
      .def_property_readonly(
          "T",
          [](const array& a) { return transpose(a); },
          "Equivalent to calling ``self.transpose()`` with no arguments.")
      .def(
          "sum",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return sum(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = none,
          "keepdims"_a = false,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`sum`.")
      .def(
          "prod",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return prod(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = none,
          "keepdims"_a = false,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`prod`.")
      .def(
          "min",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return min(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = none,
          "keepdims"_a = false,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`min`.")
      .def(
          "max",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return max(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = none,
          "keepdims"_a = false,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`max`.")
      .def(
          "logsumexp",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return logsumexp(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = none,
          "keepdims"_a = false,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`logsumexp`.")
      .def(
          "mean",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return mean(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = none,
          "keepdims"_a = false,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`mean`.")
      .def(
          "var",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             int ddof,
             StreamOrDevice s) {
            return var(a, get_reduce_axes(axis, a.ndim()), keepdims, ddof, s);
          },
          "axis"_a = none,
          "keepdims"_a = false,
          "ddof"_a = 0,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`var`.")
      .def(
          "split",
          [](const array& a,
             const std::variant<int, std::vector<int>>& indices_or_sections,
             int axis,
             StreamOrDevice s) {
            if (auto pv = std::get_if<int>(&indices_or_sections); pv) {
              return split(a, *pv, axis, s);
            } else {
              return split(
                  a, std::get<std::vector<int>>(indices_or_sections), axis, s);
            }
          },
          "indices_or_sections"_a,
          "axis"_a = 0,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`split`.")
      .def(
          "argmin",
          [](const array& a,
             std::optional<int> axis,
             bool keepdims,
             StreamOrDevice s) {
            if (axis) {
              return argmin(a, *axis, keepdims, s);
            } else {
              return argmin(a, keepdims, s);
            }
          },
          "axis"_a = std::nullopt,
          "keepdims"_a = false,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`argmin`.")
      .def(
          "argmax",
          [](const array& a,
             std::optional<int> axis,
             bool keepdims,
             StreamOrDevice s) {
            if (axis) {
              return argmax(a, *axis, keepdims, s);
            } else {
              return argmax(a, keepdims, s);
            }
          },
          "axis"_a = none,
          "keepdims"_a = false,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`argmax`.")
      .def(
          "cumsum",
          [](const array& a,
             std::optional<int> axis,
             bool reverse,
             bool inclusive,
             StreamOrDevice s) {
            if (axis) {
              return cumsum(a, *axis, reverse, inclusive, s);
            } else {
              // TODO: Implement that in the C++ API as well. See concatenate
              // above.
              return cumsum(reshape(a, {-1}, s), 0, reverse, inclusive, s);
            }
          },
          "axis"_a = none,
          py::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = none,
          "See :func:`cumsum`.")
      .def(
          "cumprod",
          [](const array& a,
             std::optional<int> axis,
             bool reverse,
             bool inclusive,
             StreamOrDevice s) {
            if (axis) {
              return cumprod(a, *axis, reverse, inclusive, s);
            } else {
              // TODO: Implement that in the C++ API as well. See concatenate
              // above.
              return cumprod(reshape(a, {-1}, s), 0, reverse, inclusive, s);
            }
          },
          "axis"_a = none,
          py::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = none,
          "See :func:`cumprod`.")
      .def(
          "cummax",
          [](const array& a,
             std::optional<int> axis,
             bool reverse,
             bool inclusive,
             StreamOrDevice s) {
            if (axis) {
              return cummax(a, *axis, reverse, inclusive, s);
            } else {
              // TODO: Implement that in the C++ API as well. See concatenate
              // above.
              return cummax(reshape(a, {-1}, s), 0, reverse, inclusive, s);
            }
          },
          "axis"_a = none,
          py::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = none,
          "See :func:`cummax`.")
      .def(
          "cummin",
          [](const array& a,
             std::optional<int> axis,
             bool reverse,
             bool inclusive,
             StreamOrDevice s) {
            if (axis) {
              return cummin(a, *axis, reverse, inclusive, s);
            } else {
              // TODO: Implement that in the C++ API as well. See concatenate
              // above.
              return cummin(reshape(a, {-1}, s), 0, reverse, inclusive, s);
            }
          },
          "axis"_a = none,
          py::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = none,
          "See :func:`cummin`.")
      .def(
          "round",
          [](const array& a, int decimals, StreamOrDevice s) {
            return round(a, decimals, s);
          },
          py::pos_only(),
          "decimals"_a = 0,
          py::kw_only(),
          "stream"_a = none,
          "See :func:`round`.");
}
