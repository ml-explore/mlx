// Copyright © 2023-2024 Apple Inc.
#include <cstdint>
#include <cstring>
#include <sstream>

#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "python/src/buffer.h"
#include "python/src/convert.h"
#include "python/src/indexing.h"
#include "python/src/utils.h"

#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

enum PyScalarT {
  pybool = 0,
  pyint = 1,
  pyfloat = 2,
  pycomplex = 3,
};

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

auto to_scalar(array& a) {
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
  }
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
    } else if (nb::isinstance<nb::bool_>(l)) {
      t = pybool;
    } else if (nb::isinstance<nb::int_>(l)) {
      t = pyint;
    } else if (nb::isinstance<nb::float_>(l)) {
      t = pyfloat;
    } else if (PyComplex_Check(l.ptr())) {
      t = pycomplex;
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
      std::ostringstream msg;
      msg << "Invalid type  " << nb::type_name(l.type()).c_str()
          << " received in array initialization.";
      throw std::invalid_argument(msg.str());
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

using ArrayInitType = std::variant<
    nb::bool_,
    nb::int_,
    nb::float_,
    // Must be above ndarray
    array,
    // Must be above complex
    nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>,
    std::complex<float>,
    nb::list,
    nb::tuple,
    nb::object>;

// Forward declaration
array create_array(ArrayInitType v, std::optional<Dtype> t);

template <typename T>
array array_from_list(
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
array array_from_list(T pl, std::optional<Dtype> dtype) {
  // Compute the shape
  std::vector<int> shape;
  get_shape(pl, shape);

  // Validate the shape and type
  bool all_python_primitive_elements = true;
  auto type = validate_shape(pl, shape, 0, all_python_primitive_elements);

  if (all_python_primitive_elements) {
    // `pl` does not contain mlx arrays
    return array_from_list(pl, type, dtype, shape);
  }

  // `pl` contains mlx arrays
  std::vector<array> arrays;
  for (auto l : pl) {
    arrays.push_back(create_array(nb::cast<ArrayInitType>(l), dtype));
  }
  return stack(arrays);
}

///////////////////////////////////////////////////////////////////////////////
// Module
///////////////////////////////////////////////////////////////////////////////

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

class ArrayAt {
 public:
  ArrayAt(array x) : x_(std::move(x)) {}
  ArrayAt& set_indices(nb::object indices) {
    indices_ = indices;
    return *this;
  }
  array add(const ScalarOrArray& v) {
    return mlx_add_item(x_, indices_, v);
  }
  array subtract(const ScalarOrArray& v) {
    return mlx_subtract_item(x_, indices_, v);
  }
  array multiply(const ScalarOrArray& v) {
    return mlx_multiply_item(x_, indices_, v);
  }
  array divide(const ScalarOrArray& v) {
    return mlx_divide_item(x_, indices_, v);
  }
  array maximum(const ScalarOrArray& v) {
    return mlx_maximum_item(x_, indices_, v);
  }
  array minimum(const ScalarOrArray& v) {
    return mlx_minimum_item(x_, indices_, v);
  }

 private:
  array x_;
  nb::object indices_;
};

class ArrayPythonIterator {
 public:
  ArrayPythonIterator(array x) : idx_(0), x_(std::move(x)) {
    if (x_.shape(0) > 0 && x_.shape(0) < 10) {
      splits_ = split(x_, x_.shape(0));
    }
  }

  array next() {
    if (idx_ >= x_.shape(0)) {
      throw nb::stop_iteration();
    }

    if (idx_ >= 0 && idx_ < splits_.size()) {
      return squeeze(splits_[idx_++], 0);
    }

    return *(x_.begin() + idx_++);
  }

 private:
  int idx_;
  array x_;
  std::vector<array> splits_;
};

void init_array(nb::module_& m) {
  // Set Python print formatting options
  mlx::core::global_formatter.capitalize_bool = true;

  // Types
  nb::class_<Dtype>(
      m,
      "Dtype",
      R"pbdoc(
      An object to hold the type of a :class:`array`.

      See the :ref:`list of types <data_types>` for more details
      on available data types.
      )pbdoc")
      .def_ro("size", &Dtype::size, R"pbdoc(Size of the type in bytes.)pbdoc")
      .def(
          "__repr__",
          [](const Dtype& t) {
            std::ostringstream os;
            os << "mlx.core.";
            os << t;
            return os.str();
          })
      .def(
          "__eq__",
          [](const Dtype& t, const nb::object& other) {
            return nb::isinstance<Dtype>(other) && t == nb::cast<Dtype>(other);
          })
      .def("__hash__", [](const Dtype& t) {
        return static_cast<int64_t>(t.val);
      });
  m.attr("bool_") = nb::cast(bool_);
  m.attr("uint8") = nb::cast(uint8);
  m.attr("uint16") = nb::cast(uint16);
  m.attr("uint32") = nb::cast(uint32);
  m.attr("uint64") = nb::cast(uint64);
  m.attr("int8") = nb::cast(int8);
  m.attr("int16") = nb::cast(int16);
  m.attr("int32") = nb::cast(int32);
  m.attr("int64") = nb::cast(int64);
  m.attr("float16") = nb::cast(float16);
  m.attr("float32") = nb::cast(float32);
  m.attr("bfloat16") = nb::cast(bfloat16);
  m.attr("complex64") = nb::cast(complex64);
  nb::class_<Dtype::Category>(
      m,
      "DtypeCategory",
      R"pbdoc(
      Type to hold categories of :class:`dtypes <Dtype>`.

      * :attr:`~mlx.core.generic`

        * :ref:`bool_ <data_types>`
        * :attr:`~mlx.core.number`

          * :attr:`~mlx.core.integer`

            * :attr:`~mlx.core.unsignedinteger`

              * :ref:`uint8 <data_types>`
              * :ref:`uint16 <data_types>`
              * :ref:`uint32 <data_types>`
              * :ref:`uint64 <data_types>`

            * :attr:`~mlx.core.signedinteger`

              * :ref:`int8 <data_types>`
              * :ref:`int32 <data_types>`
              * :ref:`int64 <data_types>`

          * :attr:`~mlx.core.inexact`

            * :attr:`~mlx.core.floating`

              * :ref:`float16 <data_types>`
              * :ref:`bfloat16 <data_types>`
              * :ref:`float32 <data_types>`

            * :attr:`~mlx.core.complexfloating`

              * :ref:`complex128 <data_types>`

      See also :func:`~mlx.core.issubdtype`.
      )pbdoc");
  m.attr("complexfloating") = nb::cast(complexfloating);
  m.attr("floating") = nb::cast(floating);
  m.attr("inexact") = nb::cast(inexact);
  m.attr("signedinteger") = nb::cast(signedinteger);
  m.attr("unsignedinteger") = nb::cast(unsignedinteger);
  m.attr("integer") = nb::cast(integer);
  m.attr("number") = nb::cast(number);
  m.attr("generic") = nb::cast(generic);

  nb::class_<ArrayAt>(
      m,
      "_ArrayAt",
      R"pbdoc(
      A helper object to apply updates at specific indices.
      )pbdoc")
      .def(
          nb::init<const array&>(),
          "x"_a,
          nb::sig("def __init__(self, x: array)"))
      .def("__getitem__", &ArrayAt::set_indices, "indices"_a.none())
      .def("add", &ArrayAt::add, "value"_a)
      .def("subtract", &ArrayAt::subtract, "value"_a)
      .def("multiply", &ArrayAt::multiply, "value"_a)
      .def("divide", &ArrayAt::divide, "value"_a)
      .def("maximum", &ArrayAt::maximum, "value"_a)
      .def("minimum", &ArrayAt::minimum, "value"_a);

  nb::class_<ArrayPythonIterator>(
      m,
      "_ArrayIterator",
      R"pbdoc(
      A helper object to iterate over the 1st dimension of an array.
      )pbdoc")
      .def(
          nb::init<const array&>(),
          "x"_a,
          nb::sig("def __init__(self, x: array)"))
      .def("__next__", &ArrayPythonIterator::next)
      .def("__iter__", [](const ArrayPythonIterator& it) { return it; });

  // Install buffer protocol functions
  PyType_Slot array_slots[] = {
      {Py_bf_getbuffer, (void*)getbuffer},
      {Py_bf_releasebuffer, (void*)releasebuffer},
      {0, nullptr}};

  nb::class_<array>(
      m,
      "array",
      R"pbdoc(An N-dimensional array object.)pbdoc",
      nb::type_slots(array_slots),
      nb::is_weak_referenceable())
      .def(
          "__init__",
          [](array* aptr, ArrayInitType v, std::optional<Dtype> t) {
            new (aptr) array(create_array(v, t));
          },
          "val"_a,
          "dtype"_a = nb::none(),
          nb::sig(
              "def __init__(self: array, val: Union[scalar, list, tuple, numpy.ndarray, array], dtype: Optional[Dtype] = None)"))
      .def_prop_ro(
          "size", &array::size, R"pbdoc(Number of elements in the array.)pbdoc")
      .def_prop_ro("ndim", &array::ndim, R"pbdoc(The array's dimension.)pbdoc")
      .def_prop_ro(
          "itemsize",
          &array::itemsize,
          R"pbdoc(The size of the array's datatype in bytes.)pbdoc")
      .def_prop_ro(
          "nbytes",
          &array::nbytes,
          R"pbdoc(The number of bytes in the array.)pbdoc")
      .def_prop_ro(
          "shape",
          [](const array& a) { return nb::tuple(nb::cast(a.shape())); },
          R"pbdoc(
          The shape of the array as a Python tuple.

          Returns:
            tuple(int): A tuple containing the sizes of each dimension.
        )pbdoc")
      .def_prop_ro(
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

                The value type of the list corresponding to the last dimension is either
                ``bool``, ``int`` or ``float`` depending on the ``dtype`` of the array.
          )pbdoc")
      .def(
          "astype",
          &astype,
          "dtype"_a,
          "stream"_a = nb::none(),
          R"pbdoc(
            Cast the array to a specified type.

            Args:
                dtype (Dtype): Type to which the array is cast.
                stream (Stream): Stream (or device) for the operation.

            Returns:
                array: The array with type ``dtype``.
          )pbdoc")
      .def("__getitem__", mlx_get_item, nb::arg().none())
      .def("__setitem__", mlx_set_item, nb::arg().none(), nb::arg())
      .def_prop_ro(
          "at",
          [](const array& a) { return ArrayAt(a); },
          R"pbdoc(
            Used to apply updates at the given indices.

            .. note::

               Regular in-place updates map to assignment. For instance ``x[idx] += y``
               maps to ``x[idx] = x[idx] + y``. As a result, assigning to the
               same index ignores all but one update. Using ``x.at[idx].add(y)``
               will correctly apply all updates to all indices.

            .. list-table::
               :header-rows: 1

               * - array.at syntax
                 - In-place syntax
               * - ``x = x.at[idx].add(y)``
                 - ``x[idx] += y``
               * - ``x = x.at[idx].subtract(y)``
                 - ``x[idx] -= y``
               * - ``x = x.at[idx].multiply(y)``
                 - ``x[idx] *= y``
               * - ``x = x.at[idx].divide(y)``
                 - ``x[idx] /= y``
               * - ``x = x.at[idx].maximum(y)``
                 - ``x[idx] = mx.maximum(x[idx], y)``
               * - ``x = x.at[idx].minimum(y)``
                 - ``x[idx] = mx.minimum(x[idx], y)``

            Example:
                >>> a = mx.array([0, 0])
                >>> idx = mx.array([0, 1, 0, 1])
                >>> a[idx] += 1
                >>> a
                array([1, 1], dtype=int32)
                >>>
                >>> a = mx.array([0, 0])
                >>> a.at[idx].add(1)
                array([2, 2], dtype=int32)
          )pbdoc")
      .def(
          "__len__",
          [](const array& a) {
            if (a.ndim() == 0) {
              throw nb::type_error("len() 0-dimensional array.");
            }
            return a.shape(0);
          })
      .def("__iter__", [](const array& a) { return ArrayPythonIterator(a); })
      .def(
          "__getstate__",
          [](const array& a) {
            if (a.dtype() == bfloat16) {
            }
            return mlx_to_np_array(a);
          })
      .def(
          "__setstate__",
          [](array& arr,
             const nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>& state) {
            new (&arr) array(nd_array_to_mlx(state, std::nullopt));
          })
      .def("__copy__", [](const array& self) { return array(self); })
      .def(
          "__deepcopy__",
          [](const array& self, nb::dict) { return array(self); },
          "memo"_a)
      .def(
          "__add__",
          [](const array& a, const ScalarOrArray v) {
            auto b = to_array(v, a.dtype());
            return add(a, b);
          },
          "other"_a)
      .def(
          "__iadd__",
          [](array& a, const ScalarOrArray v) -> array& {
            a.overwrite_descriptor(add(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
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
          "__isub__",
          [](array& a, const ScalarOrArray v) -> array& {
            a.overwrite_descriptor(subtract(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
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
          "__imul__",
          [](array& a, const ScalarOrArray v) -> array& {
            a.overwrite_descriptor(multiply(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
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
          "__itruediv__",
          [](array& a, const ScalarOrArray v) -> array& {
            if (!issubdtype(a.dtype(), inexact)) {
              throw std::invalid_argument(
                  "In place division cannot cast to non-floating point type.");
            }
            a.overwrite_descriptor(divide(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rtruediv__",
          [](const array& a, const ScalarOrArray v) {
            return divide(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__div__",
          [](const array& a, const ScalarOrArray v) {
            return divide(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__rdiv__",
          [](const array& a, const ScalarOrArray v) {
            return divide(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__floordiv__",
          [](const array& a, const ScalarOrArray v) {
            return floor_divide(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__ifloordiv__",
          [](array& a, const ScalarOrArray v) -> array& {
            a.overwrite_descriptor(floor_divide(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rfloordiv__",
          [](const array& a, const ScalarOrArray v) {
            auto b = to_array(v, a.dtype());
            return floor_divide(b, a);
          },
          "other"_a)
      .def(
          "__mod__",
          [](const array& a, const ScalarOrArray v) {
            return remainder(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__imod__",
          [](array& a, const ScalarOrArray v) -> array& {
            a.overwrite_descriptor(remainder(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
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
      .def("__bool__", [](array& a) { return nb::bool_(to_scalar(a)); })
      .def(
          "__repr__",
          [](array& a) {
            if (!a.is_evaled()) {
              nb::gil_scoped_release nogil;
              a.eval();
            }
            std::ostringstream os;
            os << a;
            return os.str();
          })
      .def(
          "__matmul__",
          [](const array& a, array& other) { return matmul(a, other); },
          "other"_a)
      .def(
          "__imatmul__",
          [](array& a, array& other) -> array& {
            a.overwrite_descriptor(matmul(a, other));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__pow__",
          [](const array& a, const ScalarOrArray v) {
            return power(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__rpow__",
          [](const array& a, const ScalarOrArray v) {
            return power(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__ipow__",
          [](array& a, const ScalarOrArray v) -> array& {
            a.overwrite_descriptor(power(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__invert__",
          [](const array& a) {
            if (issubdtype(a.dtype(), inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with or bitwise inversion.");
            }
            if (a.dtype() != bool_) {
              throw std::invalid_argument(
                  "Bitwise inversion not yet supported for integer types.");
            }
            return logical_not(a);
          })
      .def(
          "__and__",
          [](const array& a, const ScalarOrArray v) {
            auto b = to_array(v, a.dtype());
            if (issubdtype(a.dtype(), inexact) ||
                issubdtype(b.dtype(), inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise and.");
            }
            if (a.dtype() != bool_ && b.dtype() != bool_) {
              throw std::invalid_argument(
                  "Bitwise and not yet supported for integer types.");
            }
            return logical_and(a, b);
          },
          "other"_a)
      .def(
          "__iand__",
          [](array& a, const ScalarOrArray v) -> array& {
            auto b = to_array(v, a.dtype());
            if (issubdtype(a.dtype(), inexact) ||
                issubdtype(b.dtype(), inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise and.");
            }
            if (a.dtype() != bool_ && b.dtype() != bool_) {
              throw std::invalid_argument(
                  "Bitwise and not yet supported for integer types.");
            }
            a.overwrite_descriptor(logical_and(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__or__",
          [](const array& a, const ScalarOrArray v) {
            auto b = to_array(v, a.dtype());
            if (issubdtype(a.dtype(), inexact) ||
                issubdtype(b.dtype(), inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with or bitwise or.");
            }
            if (a.dtype() != bool_ && b.dtype() != bool_) {
              throw std::invalid_argument(
                  "Bitwise or not yet supported for integer types.");
            }
            return logical_or(a, b);
          },
          "other"_a)
      .def(
          "__ior__",
          [](array& a, const ScalarOrArray v) -> array& {
            auto b = to_array(v, a.dtype());
            if (issubdtype(a.dtype(), inexact) ||
                issubdtype(b.dtype(), inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with or bitwise or.");
            }
            if (a.dtype() != bool_ && b.dtype() != bool_) {
              throw std::invalid_argument(
                  "Bitwise or not yet supported for integer types.");
            }
            a.overwrite_descriptor(logical_or(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
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
          nb::kw_only(),
          "stream"_a = nb::none(),
          R"pbdoc(
            See :func:`flatten`.
          )pbdoc")
      .def(
          "reshape",
          [](const array& a, nb::args shape_, StreamOrDevice s) {
            std::vector<int> shape;
            if (!nb::isinstance<int>(shape_[0])) {
              shape = nb::cast<std::vector<int>>(shape_[0]);
            } else {
              shape = nb::cast<std::vector<int>>(shape_);
            }
            return reshape(a, shape, s);
          },
          "shape"_a,
          "stream"_a = nb::none(),
          R"pbdoc(
            Equivalent to :func:`reshape` but the shape can be passed either as a
            :obj:`tuple` or as separate arguments.

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
          "axis"_a = nb::none(),
          nb::kw_only(),
          "stream"_a = nb::none(),
          R"pbdoc(
            See :func:`squeeze`.
          )pbdoc")
      .def(
          "abs",
          &mlx::core::abs,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`abs`.")
      .def(
          "__abs__", [](const array& a) { return abs(a); }, "See :func:`abs`.")
      .def(
          "square",
          &square,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`square`.")
      .def(
          "sqrt",
          &mlx::core::sqrt,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`sqrt`.")
      .def(
          "rsqrt",
          &rsqrt,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`rsqrt`.")
      .def(
          "reciprocal",
          &reciprocal,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`reciprocal`.")
      .def(
          "exp",
          &mlx::core::exp,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`exp`.")
      .def(
          "log",
          &mlx::core::log,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`log`.")
      .def(
          "log2",
          &mlx::core::log2,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`log2`.")
      .def(
          "log10",
          &mlx::core::log10,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`log10`.")
      .def(
          "sin",
          &mlx::core::sin,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`sin`.")
      .def(
          "cos",
          &mlx::core::cos,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`cos`.")
      .def(
          "log1p",
          &mlx::core::log1p,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`log1p`.")
      .def(
          "all",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return all(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`all`.")
      .def(
          "any",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return any(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`any`.")
      .def(
          "moveaxis",
          &moveaxis,
          "source"_a,
          "destination"_a,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`moveaxis`.")
      .def(
          "swapaxes",
          &swapaxes,
          "axis1"_a,
          "axis2"_a,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`swapaxes`.")
      .def(
          "transpose",
          [](const array& a, nb::args axes_, StreamOrDevice s) {
            if (axes_.size() == 0) {
              return transpose(a, s);
            }
            std::vector<int> axes;
            if (!nb::isinstance<int>(axes_[0])) {
              axes = nb::cast<std::vector<int>>(axes_[0]);
            } else {
              axes = nb::cast<std::vector<int>>(axes_);
            }
            return transpose(a, axes, s);
          },
          "axes"_a,
          "stream"_a = nb::none(),
          R"pbdoc(
            Equivalent to :func:`transpose` but the axes can be passed either as
            a tuple or as separate arguments.

            See :func:`transpose` for full documentation.
          )pbdoc")
      .def_prop_ro(
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
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`sum`.")
      .def(
          "prod",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return prod(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`prod`.")
      .def(
          "min",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return min(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`min`.")
      .def(
          "max",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return max(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`max`.")
      .def(
          "logsumexp",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return logsumexp(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`logsumexp`.")
      .def(
          "mean",
          [](const array& a,
             const IntOrVec& axis,
             bool keepdims,
             StreamOrDevice s) {
            return mean(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
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
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          "ddof"_a = 0,
          nb::kw_only(),
          "stream"_a = nb::none(),
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
          nb::kw_only(),
          "stream"_a = nb::none(),
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
          nb::kw_only(),
          "stream"_a = nb::none(),
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
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
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
          "axis"_a = nb::none(),
          nb::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = nb::none(),
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
          "axis"_a = nb::none(),
          nb::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = nb::none(),
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
          "axis"_a = nb::none(),
          nb::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = nb::none(),
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
          "axis"_a = nb::none(),
          nb::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = nb::none(),
          "See :func:`cummin`.")
      .def(
          "round",
          [](const array& a, int decimals, StreamOrDevice s) {
            return round(a, decimals, s);
          },
          "decimals"_a = 0,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`round`.")
      .def(
          "diagonal",
          [](const array& a,
             int offset,
             int axis1,
             int axis2,
             StreamOrDevice s) { return diagonal(a, offset, axis1, axis2, s); },
          "offset"_a = 0,
          "axis1"_a = 0,
          "axis2"_a = 1,
          "stream"_a = nb::none(),
          "See :func:`diagonal`.")
      .def(
          "diag",
          [](const array& a, int k, StreamOrDevice s) { return diag(a, k, s); },
          "k"_a = 0,
          nb::kw_only(),
          "stream"_a = nb::none(),
          R"pbdoc(
            Extract a diagonal or construct a diagonal matrix.
        )pbdoc");
}
