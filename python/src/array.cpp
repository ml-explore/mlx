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
#include <nanobind/typing.h>

#include "mlx/backend/metal/metal.h"
#include "python/src/buffer.h"
#include "python/src/convert.h"
#include "python/src/indexing.h"
#include "python/src/small_vector.h"
#include "python/src/utils.h"

#include "mlx/mlx.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

class ArrayAt {
 public:
  ArrayAt(mx::array x) : x_(std::move(x)) {}
  ArrayAt& set_indices(nb::object indices) {
    initialized_ = true;
    indices_ = indices;
    return *this;
  }
  void check_initialized() {
    if (!initialized_) {
      throw std::invalid_argument(
          "Must give indices to array.at (e.g. `x.at[0].add(4)`).");
    }
  }

  mx::array add(const ScalarOrArray& v) {
    check_initialized();
    return mlx_add_item(x_, indices_, v);
  }
  mx::array subtract(const ScalarOrArray& v) {
    check_initialized();
    return mlx_subtract_item(x_, indices_, v);
  }
  mx::array multiply(const ScalarOrArray& v) {
    check_initialized();
    return mlx_multiply_item(x_, indices_, v);
  }
  mx::array divide(const ScalarOrArray& v) {
    check_initialized();
    return mlx_divide_item(x_, indices_, v);
  }
  mx::array maximum(const ScalarOrArray& v) {
    check_initialized();
    return mlx_maximum_item(x_, indices_, v);
  }
  mx::array minimum(const ScalarOrArray& v) {
    check_initialized();
    return mlx_minimum_item(x_, indices_, v);
  }

 private:
  mx::array x_;
  bool initialized_{false};
  nb::object indices_;
};

class ArrayPythonIterator {
 public:
  ArrayPythonIterator(mx::array x) : idx_(0), x_(std::move(x)) {
    if (x_.shape(0) > 0 && x_.shape(0) < 10) {
      splits_ = mx::split(x_, x_.shape(0));
    }
  }

  mx::array next() {
    if (idx_ >= x_.shape(0)) {
      throw nb::stop_iteration();
    }

    if (idx_ >= 0 && idx_ < splits_.size()) {
      return mx::squeeze(splits_[idx_++], 0);
    }

    return *(x_.begin() + idx_++);
  }

 private:
  int idx_;
  mx::array x_;
  std::vector<mx::array> splits_;
};

void init_array(nb::module_& m) {
  // Set Python print formatting options
  mx::get_global_formatter().capitalize_bool = true;

  // Types
  nb::class_<mx::Dtype>(
      m,
      "Dtype",
      R"pbdoc(
      An object to hold the type of a :class:`array`.

      See the :ref:`list of types <data_types>` for more details
      on available data types.
      )pbdoc")
      .def_prop_ro(
          "size", &mx::Dtype::size, R"pbdoc(Size of the type in bytes.)pbdoc")
      .def(
          "__repr__",
          [](const mx::Dtype& t) {
            std::ostringstream os;
            os << "mlx.core.";
            os << t;
            return os.str();
          })
      .def(
          "__eq__",
          [](const mx::Dtype& t, const nb::object& other) {
            return nb::isinstance<mx::Dtype>(other) &&
                t == nb::cast<mx::Dtype>(other);
          })
      .def("__hash__", [](const mx::Dtype& t) {
        return static_cast<int64_t>(t.val());
      });

  m.attr("bool_") = nb::cast(mx::bool_);
  m.attr("uint8") = nb::cast(mx::uint8);
  m.attr("uint16") = nb::cast(mx::uint16);
  m.attr("uint32") = nb::cast(mx::uint32);
  m.attr("uint64") = nb::cast(mx::uint64);
  m.attr("int8") = nb::cast(mx::int8);
  m.attr("int16") = nb::cast(mx::int16);
  m.attr("int32") = nb::cast(mx::int32);
  m.attr("int64") = nb::cast(mx::int64);
  m.attr("float16") = nb::cast(mx::float16);
  m.attr("float32") = nb::cast(mx::float32);
  m.attr("float64") = nb::cast(mx::float64);
  m.attr("bfloat16") = nb::cast(mx::bfloat16);
  m.attr("complex64") = nb::cast(mx::complex64);
  nb::enum_<mx::Dtype::Category>(
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
              * :ref:`float64 <data_types>`

            * :attr:`~mlx.core.complexfloating`

              * :ref:`complex64 <data_types>`

      See also :func:`~mlx.core.issubdtype`.
      )pbdoc")
      .value("complexfloating", mx::complexfloating)
      .value("floating", mx::floating)
      .value("inexact", mx::inexact)
      .value("signedinteger", mx::signedinteger)
      .value("unsignedinteger", mx::unsignedinteger)
      .value("integer", mx::integer)
      .value("number", mx::number)
      .value("generic", mx::generic)
      .export_values();

  nb::class_<mx::finfo>(
      m,
      "finfo",
      R"pbdoc(
      Get information on floating-point types.
      )pbdoc")
      .def(nb::init<mx::Dtype>())
      .def_ro(
          "min",
          &mx::finfo::min,
          R"pbdoc(The smallest representable number.)pbdoc")
      .def_ro(
          "max",
          &mx::finfo::max,
          R"pbdoc(The largest representable number.)pbdoc")
      .def_ro(
          "eps",
          &mx::finfo::eps,
          R"pbdoc(
            The difference between 1.0 and the next smallest
            representable number larger than 1.0.
          )pbdoc")
      .def_ro("dtype", &mx::finfo::dtype, R"pbdoc(The :obj:`Dtype`.)pbdoc")
      .def("__repr__", [](const mx::finfo& f) {
        std::ostringstream os;
        os << "finfo("
           << "min=" << f.min << ", max=" << f.max << ", dtype=" << f.dtype
           << ")";
        return os.str();
      });

  nb::class_<mx::iinfo>(
      m,
      "iinfo",
      R"pbdoc(
      Get information on integer types.
      )pbdoc")
      .def(nb::init<mx::Dtype>())
      .def_ro(
          "min",
          &mx::iinfo::min,
          R"pbdoc(The smallest representable number.)pbdoc")
      .def_ro(
          "max",
          &mx::iinfo::max,
          R"pbdoc(The largest representable number.)pbdoc")
      .def_ro("dtype", &mx::iinfo::dtype, R"pbdoc(The :obj:`Dtype`.)pbdoc")
      .def("__repr__", [](const mx::iinfo& i) {
        std::ostringstream os;
        os << "iinfo("
           << "min=" << i.min << ", max=" << i.max << ", dtype=" << i.dtype
           << ")";
        return os.str();
      });

  nb::class_<ArrayAt>(
      m,
      "ArrayAt",
      R"pbdoc(
      A helper object to apply updates at specific indices.
      )pbdoc")
      .def("__getitem__", &ArrayAt::set_indices, "indices"_a.none())
      .def("add", &ArrayAt::add, "value"_a)
      .def("subtract", &ArrayAt::subtract, "value"_a)
      .def("multiply", &ArrayAt::multiply, "value"_a)
      .def("divide", &ArrayAt::divide, "value"_a)
      .def("maximum", &ArrayAt::maximum, "value"_a)
      .def("minimum", &ArrayAt::minimum, "value"_a);

  nb::class_<ArrayLike>(
      m,
      "ArrayLike",
      R"pbdoc(
        Any Python object which has an ``__mlx__array__`` method that
        returns an :obj:`array`.
      )pbdoc")
      .def(nb::init_implicit<nb::object>());

  nb::class_<ArrayPythonIterator>(
      m,
      "ArrayIterator",
      R"pbdoc(
      A helper object to iterate over the 1st dimension of an array.
      )pbdoc")
      .def("__next__", &ArrayPythonIterator::next)
      .def("__iter__", [](const ArrayPythonIterator& it) { return it; });

  // Install buffer protocol functions
  PyType_Slot array_slots[] = {
      {Py_bf_getbuffer, (void*)getbuffer},
      {Py_bf_releasebuffer, (void*)releasebuffer},
      {0, nullptr}};

  nb::class_<mx::array>(
      m,
      "array",
      R"pbdoc(An N-dimensional array object.)pbdoc",
      nb::type_slots(array_slots),
      nb::is_weak_referenceable())
      .def(
          "__init__",
          [](mx::array* aptr, ArrayInitType v, std::optional<mx::Dtype> t) {
            new (aptr) mx::array(create_array(v, t));
          },
          "val"_a,
          "dtype"_a = nb::none(),
          nb::sig(
              "def __init__(self: array, val: Union[scalar, list, tuple, numpy.ndarray, array], dtype: Optional[Dtype] = None)"))
      .def_prop_ro(
          "size",
          &mx::array::size,
          R"pbdoc(Number of elements in the array.)pbdoc")
      .def_prop_ro(
          "ndim", &mx::array::ndim, R"pbdoc(The array's dimension.)pbdoc")
      .def_prop_ro(
          "itemsize",
          &mx::array::itemsize,
          R"pbdoc(The size of the array's datatype in bytes.)pbdoc")
      .def_prop_ro(
          "nbytes",
          &mx::array::nbytes,
          R"pbdoc(The number of bytes in the array.)pbdoc")
      .def_prop_ro(
          "shape",
          [](const mx::array& a) { return nb::cast(a.shape()); },
          nb::sig("def shape(self) -> tuple[int, ...]"),
          R"pbdoc(
          The shape of the array as a Python tuple.

          Returns:
            tuple(int): A tuple containing the sizes of each dimension.
        )pbdoc")
      .def_prop_ro(
          "dtype",
          &mx::array::dtype,
          R"pbdoc(
            The array's :class:`Dtype`.
          )pbdoc")
      .def_prop_ro(
          "real",
          [](const mx::array& a) { return mx::real(a); },
          R"pbdoc(
            The real part of a complex array.
          )pbdoc")
      .def_prop_ro(
          "imag",
          [](const mx::array& a) { return mx::imag(a); },
          R"pbdoc(
            The imaginary part of a complex array.
          )pbdoc")
      .def(
          "item",
          &to_scalar,
          nb::sig("def item(self) -> scalar"),
          R"pbdoc(
            Access the value of a scalar array.

            Returns:
                Standard Python scalar.
          )pbdoc")
      .def(
          "tolist",
          &tolist,
          nb::sig("def tolist(self) -> list_or_scalar"),
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
          &mx::astype,
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
      .def(
          "__array_namespace__",
          [](const mx::array& a,
             const std::optional<std::string>& api_version) {
            if (api_version) {
              throw std::invalid_argument(
                  "Explicitly specifying api_version is not yet implemented.");
            }
            return nb::module_::import_("mlx.core");
          },
          "api_version"_a = nb::none(),
          R"pbdoc(
            Returns an object that has all the array API functions on it.

            See the `Python array API <https://data-apis.org/array-api/latest/index.html>`_
            for more information.

            Args:
                api_version (str, optional): String representing the version
                  of the array API spec to return. Default: ``None``.

            Returns:
                out (Any): An object representing the array API namespace.
          )pbdoc")
      .def("__getitem__", mlx_get_item, nb::arg().none())
      .def("__setitem__", mlx_set_item, nb::arg().none(), nb::arg())
      .def_prop_ro(
          "at",
          [](const mx::array& a) { return ArrayAt(a); },
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
          [](const mx::array& a) {
            if (a.ndim() == 0) {
              throw nb::type_error("len() 0-dimensional array.");
            }
            return a.shape(0);
          })
      .def(
          "__iter__", [](const mx::array& a) { return ArrayPythonIterator(a); })
      .def("__getstate__", &mlx_to_np_array)
      .def(
          "__setstate__",
          [](mx::array& arr,
             const nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>& state) {
            new (&arr) mx::array(nd_array_to_mlx(state, std::nullopt));
          })
      .def("__dlpack__", [](const mx::array& a) { return mlx_to_dlpack(a); })
      .def(
          "__dlpack_device__",
          [](const mx::array& a) {
            // See
            // https://github.com/dmlc/dlpack/blob/5c210da409e7f1e51ddf445134a4376fdbd70d7d/include/dlpack/dlpack.h#L74
            if (mx::metal::is_available()) {
              return nb::make_tuple(8, 0);
            } else if (mx::cu::is_available()) {
              return nb::make_tuple(13, 0);
            } else {
              // CPU device
              return nb::make_tuple(1, 0);
            }
          })
      .def("__copy__", [](const mx::array& self) { return mx::array(self); })
      .def(
          "__deepcopy__",
          [](const mx::array& self, nb::dict) { return mx::array(self); },
          "memo"_a)
      .def(
          "__add__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("addition", v);
            }
            auto b = to_array(v, a.dtype());
            return mx::add(a, b);
          },
          "other"_a)
      .def(
          "__iadd__",
          [](mx::array& a, const ScalarOrArray v) -> mx::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace addition", v);
            }
            a.overwrite_descriptor(mx::add(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__radd__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("addition", v);
            }
            return mx::add(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__sub__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("subtraction", v);
            }
            return mx::subtract(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__isub__",
          [](mx::array& a, const ScalarOrArray v) -> mx::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace subtraction", v);
            }
            a.overwrite_descriptor(mx::subtract(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rsub__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("subtraction", v);
            }
            return mx::subtract(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__mul__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("multiplication", v);
            }
            return mx::multiply(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__imul__",
          [](mx::array& a, const ScalarOrArray v) -> mx::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace multiplication", v);
            }
            a.overwrite_descriptor(mx::multiply(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rmul__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("multiplication", v);
            }
            return mx::multiply(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__truediv__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("division", v);
            }
            return mx::divide(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__itruediv__",
          [](mx::array& a, const ScalarOrArray v) -> mx::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace division", v);
            }
            if (!mx::issubdtype(a.dtype(), mx::inexact)) {
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
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("division", v);
            }
            return mx::divide(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__div__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("division", v);
            }
            return mx::divide(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__rdiv__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("division", v);
            }
            return mx::divide(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__floordiv__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("floor division", v);
            }
            return mx::floor_divide(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__ifloordiv__",
          [](mx::array& a, const ScalarOrArray v) -> mx::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace floor division", v);
            }
            a.overwrite_descriptor(mx::floor_divide(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rfloordiv__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("floor division", v);
            }
            auto b = to_array(v, a.dtype());
            return mx::floor_divide(b, a);
          },
          "other"_a)
      .def(
          "__mod__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("modulus", v);
            }
            return mx::remainder(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__imod__",
          [](mx::array& a, const ScalarOrArray v) -> mx::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace modulus", v);
            }
            a.overwrite_descriptor(mx::remainder(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rmod__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("modulus", v);
            }
            return mx::remainder(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__eq__",
          [](const mx::array& a,
             const ScalarOrArray& v) -> std::variant<mx::array, bool> {
            if (!is_comparable_with_array(v)) {
              return false;
            }
            return mx::equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__lt__",
          [](const mx::array& a, const ScalarOrArray v) -> mx::array {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("less than", v);
            }
            return mx::less(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__le__",
          [](const mx::array& a, const ScalarOrArray v) -> mx::array {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("less than or equal", v);
            }
            return mx::less_equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__gt__",
          [](const mx::array& a, const ScalarOrArray v) -> mx::array {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("greater than", v);
            }
            return mx::greater(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__ge__",
          [](const mx::array& a, const ScalarOrArray v) -> mx::array {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("greater than or equal", v);
            }
            return mx::greater_equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__ne__",
          [](const mx::array& a,
             const ScalarOrArray v) -> std::variant<mx::array, bool> {
            if (!is_comparable_with_array(v)) {
              return true;
            }
            return mx::not_equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def("__neg__", [](const mx::array& a) { return -a; })
      .def("__bool__", [](mx::array& a) { return nb::bool_(to_scalar(a)); })
      .def(
          "__repr__",
          [](mx::array& a) {
            nb::gil_scoped_release nogil;
            std::ostringstream os;
            os << a;
            return os.str();
          })
      .def(
          "__matmul__",
          [](const mx::array& a, mx::array& other) {
            return mx::matmul(a, other);
          },
          "other"_a)
      .def(
          "__imatmul__",
          [](mx::array& a, mx::array& other) -> mx::array& {
            a.overwrite_descriptor(mx::matmul(a, other));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__pow__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("power", v);
            }
            return mx::power(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__rpow__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("power", v);
            }
            return mx::power(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__ipow__",
          [](mx::array& a, const ScalarOrArray v) -> mx::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace power", v);
            }
            a.overwrite_descriptor(mx::power(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__invert__",
          [](const mx::array& a) {
            if (mx::issubdtype(a.dtype(), mx::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise inversion.");
            }
            if (a.dtype() == mx::bool_) {
              return mx::logical_not(a);
            }
            return mx::bitwise_invert(a);
          })
      .def(
          "__and__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("bitwise and", v);
            }
            auto b = to_array(v, a.dtype());
            if (mx::issubdtype(a.dtype(), mx::inexact) ||
                mx::issubdtype(b.dtype(), mx::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise and.");
            }
            return mx::bitwise_and(a, b);
          },
          "other"_a)
      .def(
          "__iand__",
          [](mx::array& a, const ScalarOrArray v) -> mx::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace bitwise and", v);
            }
            auto b = to_array(v, a.dtype());
            if (mx::issubdtype(a.dtype(), mx::inexact) ||
                mx::issubdtype(b.dtype(), mx::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise and.");
            }
            a.overwrite_descriptor(mx::bitwise_and(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__or__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("bitwise or", v);
            }
            auto b = to_array(v, a.dtype());
            if (mx::issubdtype(a.dtype(), mx::inexact) ||
                mx::issubdtype(b.dtype(), mx::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise or.");
            }
            return mx::bitwise_or(a, b);
          },
          "other"_a)
      .def(
          "__ior__",
          [](mx::array& a, const ScalarOrArray v) -> mx::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace bitwise or", v);
            }
            auto b = to_array(v, a.dtype());
            if (mx::issubdtype(a.dtype(), mx::inexact) ||
                mx::issubdtype(b.dtype(), mx::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise or.");
            }
            a.overwrite_descriptor(mx::bitwise_or(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__lshift__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("left shift", v);
            }
            auto b = to_array(v, a.dtype());
            if (mx::issubdtype(a.dtype(), mx::inexact) ||
                mx::issubdtype(b.dtype(), mx::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with left shift.");
            }
            return mx::left_shift(a, b);
          },
          "other"_a)
      .def(
          "__ilshift__",
          [](mx::array& a, const ScalarOrArray v) -> mx::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace left shift", v);
            }
            auto b = to_array(v, a.dtype());
            if (mx::issubdtype(a.dtype(), mx::inexact) ||
                mx::issubdtype(b.dtype(), mx::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with left shift.");
            }
            a.overwrite_descriptor(mx::left_shift(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rshift__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("right shift", v);
            }
            auto b = to_array(v, a.dtype());
            if (mx::issubdtype(a.dtype(), mx::inexact) ||
                mx::issubdtype(b.dtype(), mx::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with right shift.");
            }
            return mx::right_shift(a, b);
          },
          "other"_a)
      .def(
          "__irshift__",
          [](mx::array& a, const ScalarOrArray v) -> mx::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace right shift", v);
            }
            auto b = to_array(v, a.dtype());
            if (mx::issubdtype(a.dtype(), mx::inexact) ||
                mx::issubdtype(b.dtype(), mx::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with right shift.");
            }
            a.overwrite_descriptor(mx::right_shift(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__xor__",
          [](const mx::array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("bitwise xor", v);
            }
            auto b = to_array(v, a.dtype());
            if (mx::issubdtype(a.dtype(), mx::inexact) ||
                mx::issubdtype(b.dtype(), mx::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise xor.");
            }
            return mx::bitwise_xor(a, b);
          },
          "other"_a)
      .def(
          "__ixor__",
          [](mx::array& a, const ScalarOrArray v) -> mx::array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace bitwise xor", v);
            }
            auto b = to_array(v, a.dtype());
            if (mx::issubdtype(a.dtype(), mx::inexact) ||
                mx::issubdtype(b.dtype(), mx::inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed bitwise xor.");
            }
            a.overwrite_descriptor(mx::bitwise_xor(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def("__int__", [](mx::array& a) { return nb::int_(to_scalar(a)); })
      .def("__float__", [](mx::array& a) { return nb::float_(to_scalar(a)); })
      .def(
          "__format__",
          [](mx::array& a, nb::object format_spec) {
            if (nb::len(nb::str(format_spec)) > 0 && a.ndim() > 0) {
              throw nb::type_error(
                  "unsupported format string passed to mx.array.__format__");
            } else if (a.ndim() == 0) {
              auto obj = to_scalar(a);
              return nb::cast<std::string>(
                  nb::handle(PyObject_Format(obj.ptr(), format_spec.ptr())));
            } else {
              nb::gil_scoped_release nogil;
              std::ostringstream os;
              os << a;
              return os.str();
            }
          })
      .def(
          "flatten",
          [](const mx::array& a,
             int start_axis,
             int end_axis,
             const mx::StreamOrDevice& s) {
            return mx::flatten(a, start_axis, end_axis, s);
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
          [](const mx::array& a, nb::args shape_, mx::StreamOrDevice s) {
            mx::Shape shape;
            if (!nb::isinstance<int>(shape_[0])) {
              shape = nb::cast<mx::Shape>(shape_[0]);
            } else {
              shape = nb::cast<mx::Shape>(shape_);
            }
            return mx::reshape(a, std::move(shape), s);
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
          [](const mx::array& a,
             const IntOrVec& v,
             const mx::StreamOrDevice& s) {
            if (std::holds_alternative<std::monostate>(v)) {
              return mx::squeeze(a, s);
            } else if (auto pv = std::get_if<int>(&v); pv) {
              return mx::squeeze(a, *pv, s);
            } else {
              return mx::squeeze(a, std::get<std::vector<int>>(v), s);
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
          &mx::abs,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`abs`.")
      .def(
          "__abs__",
          [](const mx::array& a) { return mx::abs(a); },
          "See :func:`abs`.")
      .def(
          "square",
          &mx::square,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`square`.")
      .def(
          "sqrt",
          &mx::sqrt,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`sqrt`.")
      .def(
          "rsqrt",
          &mx::rsqrt,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`rsqrt`.")
      .def(
          "reciprocal",
          &mx::reciprocal,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`reciprocal`.")
      .def(
          "exp",
          &mx::exp,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`exp`.")
      .def(
          "log",
          &mx::log,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`log`.")
      .def(
          "log2",
          &mx::log2,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`log2`.")
      .def(
          "log10",
          &mx::log10,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`log10`.")
      .def(
          "sin",
          &mx::sin,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`sin`.")
      .def(
          "cos",
          &mx::cos,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`cos`.")
      .def(
          "log1p",
          &mx::log1p,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`log1p`.")
      .def(
          "all",
          [](const mx::array& a,
             const IntOrVec& axis,
             bool keepdims,
             mx::StreamOrDevice s) {
            return mx::all(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`all`.")
      .def(
          "any",
          [](const mx::array& a,
             const IntOrVec& axis,
             bool keepdims,
             mx::StreamOrDevice s) {
            return mx::any(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`any`.")
      .def(
          "moveaxis",
          &mx::moveaxis,
          "source"_a,
          "destination"_a,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`moveaxis`.")
      .def(
          "swapaxes",
          &mx::swapaxes,
          "axis1"_a,
          "axis2"_a,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`swapaxes`.")
      .def(
          "transpose",
          [](const mx::array& a, nb::args axes_, mx::StreamOrDevice s) {
            if (axes_.size() == 0) {
              return mx::transpose(a, s);
            }
            std::vector<int> axes;
            if (!nb::isinstance<int>(axes_[0])) {
              axes = nb::cast<std::vector<int>>(axes_[0]);
            } else {
              axes = nb::cast<std::vector<int>>(axes_);
            }
            return mx::transpose(a, axes, s);
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
          [](const mx::array& a) { return mx::transpose(a); },
          "Equivalent to calling ``self.transpose()`` with no arguments.")
      .def(
          "sum",
          [](const mx::array& a,
             const IntOrVec& axis,
             bool keepdims,
             mx::StreamOrDevice s) {
            return mx::sum(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`sum`.")
      .def(
          "prod",
          [](const mx::array& a,
             const IntOrVec& axis,
             bool keepdims,
             mx::StreamOrDevice s) {
            return mx::prod(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`prod`.")
      .def(
          "min",
          [](const mx::array& a,
             const IntOrVec& axis,
             bool keepdims,
             mx::StreamOrDevice s) {
            return mx::min(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`min`.")
      .def(
          "max",
          [](const mx::array& a,
             const IntOrVec& axis,
             bool keepdims,
             mx::StreamOrDevice s) {
            return mx::max(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`max`.")
      .def(
          "logcumsumexp",
          [](const mx::array& a,
             std::optional<int> axis,
             bool reverse,
             bool inclusive,
             mx::StreamOrDevice s) {
            if (axis) {
              return mx::logcumsumexp(a, *axis, reverse, inclusive, s);
            } else {
              // TODO: Implement that in the C++ API as well. See concatenate
              // above.
              return mx::logcumsumexp(
                  mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
            }
          },
          "axis"_a = nb::none(),
          nb::kw_only(),
          "reverse"_a = false,
          "inclusive"_a = true,
          "stream"_a = nb::none(),
          "See :func:`logcumsumexp`.")
      .def(
          "logsumexp",
          [](const mx::array& a,
             const IntOrVec& axis,
             bool keepdims,
             mx::StreamOrDevice s) {
            return mx::logsumexp(
                a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`logsumexp`.")
      .def(
          "mean",
          [](const mx::array& a,
             const IntOrVec& axis,
             bool keepdims,
             mx::StreamOrDevice s) {
            return mx::mean(a, get_reduce_axes(axis, a.ndim()), keepdims, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`mean`.")
      .def(
          "std",
          [](const mx::array& a,
             const IntOrVec& axis,
             bool keepdims,
             int ddof,
             mx::StreamOrDevice s) {
            return mx::std(
                a, get_reduce_axes(axis, a.ndim()), keepdims, ddof, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          "ddof"_a = 0,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`std`.")
      .def(
          "var",
          [](const mx::array& a,
             const IntOrVec& axis,
             bool keepdims,
             int ddof,
             mx::StreamOrDevice s) {
            return mx::var(
                a, get_reduce_axes(axis, a.ndim()), keepdims, ddof, s);
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          "ddof"_a = 0,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`var`.")
      .def(
          "split",
          [](const mx::array& a,
             const std::variant<int, mx::Shape>& indices_or_sections,
             int axis,
             mx::StreamOrDevice s) {
            if (auto pv = std::get_if<int>(&indices_or_sections); pv) {
              return mx::split(a, *pv, axis, s);
            } else {
              return mx::split(
                  a, std::get<mx::Shape>(indices_or_sections), axis, s);
            }
          },
          "indices_or_sections"_a,
          "axis"_a = 0,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`split`.")
      .def(
          "argmin",
          [](const mx::array& a,
             std::optional<int> axis,
             bool keepdims,
             mx::StreamOrDevice s) {
            if (axis) {
              return mx::argmin(a, *axis, keepdims, s);
            } else {
              return mx::argmin(a, keepdims, s);
            }
          },
          "axis"_a = std::nullopt,
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`argmin`.")
      .def(
          "argmax",
          [](const mx::array& a,
             std::optional<int> axis,
             bool keepdims,
             mx::StreamOrDevice s) {
            if (axis) {
              return mx::argmax(a, *axis, keepdims, s);
            } else {
              return mx::argmax(a, keepdims, s);
            }
          },
          "axis"_a = nb::none(),
          "keepdims"_a = false,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`argmax`.")
      .def(
          "cumsum",
          [](const mx::array& a,
             std::optional<int> axis,
             bool reverse,
             bool inclusive,
             mx::StreamOrDevice s) {
            if (axis) {
              return mx::cumsum(a, *axis, reverse, inclusive, s);
            } else {
              // TODO: Implement that in the C++ API as well. See concatenate
              // above.
              return mx::cumsum(reshape(a, {-1}, s), 0, reverse, inclusive, s);
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
          [](const mx::array& a,
             std::optional<int> axis,
             bool reverse,
             bool inclusive,
             mx::StreamOrDevice s) {
            if (axis) {
              return mx::cumprod(a, *axis, reverse, inclusive, s);
            } else {
              // TODO: Implement that in the C++ API as well. See concatenate
              // above.
              return mx::cumprod(
                  mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
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
          [](const mx::array& a,
             std::optional<int> axis,
             bool reverse,
             bool inclusive,
             mx::StreamOrDevice s) {
            if (axis) {
              return mx::cummax(a, *axis, reverse, inclusive, s);
            } else {
              // TODO: Implement that in the C++ API as well. See concatenate
              // above.
              return mx::cummax(
                  mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
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
          [](const mx::array& a,
             std::optional<int> axis,
             bool reverse,
             bool inclusive,
             mx::StreamOrDevice s) {
            if (axis) {
              return mx::cummin(a, *axis, reverse, inclusive, s);
            } else {
              // TODO: Implement that in the C++ API as well. See concatenate
              // above.
              return mx::cummin(
                  mx::reshape(a, {-1}, s), 0, reverse, inclusive, s);
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
          [](const mx::array& a, int decimals, mx::StreamOrDevice s) {
            return mx::round(a, decimals, s);
          },
          "decimals"_a = 0,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`round`.")
      .def(
          "diagonal",
          [](const mx::array& a,
             int offset,
             int axis1,
             int axis2,
             mx::StreamOrDevice s) {
            return mx::diagonal(a, offset, axis1, axis2, s);
          },
          "offset"_a = 0,
          "axis1"_a = 0,
          "axis2"_a = 1,
          "stream"_a = nb::none(),
          "See :func:`diagonal`.")
      .def(
          "diag",
          [](const mx::array& a, int k, mx::StreamOrDevice s) {
            return mx::diag(a, k, s);
          },
          "k"_a = 0,
          nb::kw_only(),
          "stream"_a = nb::none(),
          R"pbdoc(
            Extract a diagonal or construct a diagonal matrix.
        )pbdoc")
      .def(
          "conj",
          [](const mx::array& a, mx::StreamOrDevice s) {
            return mx::conjugate(to_array(a), s);
          },
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`conj`.")
      .def(
          "view",
          [](const ScalarOrArray& a,
             const mx::Dtype& dtype,
             mx::StreamOrDevice s) { return mx::view(to_array(a), dtype, s); },
          "dtype"_a,
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`view`.");
}
