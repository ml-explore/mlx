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

#include "mlx/backend/metal/metal.h"
#include "python/src/buffer.h"
#include "python/src/convert.h"
#include "python/src/indexing.h"
#include "python/src/utils.h"

#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include "mlx/utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

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
      .def("__getstate__", &mlx_to_np_array)
      .def(
          "__setstate__",
          [](array& arr,
             const nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>& state) {
            new (&arr) array(nd_array_to_mlx(state, std::nullopt));
          })
      .def("__dlpack__", [](const array& a) { return mlx_to_dlpack(a); })
      .def(
          "__dlpack_device__",
          [](const array& a) {
            if (metal::is_available()) {
              // Metal device is available
              constexpr int kDLMetal = 8;
              return kDLMetal;
            } else {
              // CPU device
              constexpr int kDLCPU = 1;
              return kDLCPU;
            }
          })
      .def("__copy__", [](const array& self) { return array(self); })
      .def(
          "__deepcopy__",
          [](const array& self, nb::dict) { return array(self); },
          "memo"_a)
      .def(
          "__add__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("addition", v);
            }
            auto b = to_array(v, a.dtype());
            return add(a, b);
          },
          "other"_a)
      .def(
          "__iadd__",
          [](array& a, const ScalarOrArray v) -> array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace addition", v);
            }
            a.overwrite_descriptor(add(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__radd__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("addition", v);
            }
            return add(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__sub__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("subtraction", v);
            }
            return subtract(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__isub__",
          [](array& a, const ScalarOrArray v) -> array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace subtraction", v);
            }
            a.overwrite_descriptor(subtract(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rsub__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("subtraction", v);
            }
            return subtract(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__mul__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("multiplication", v);
            }
            return multiply(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__imul__",
          [](array& a, const ScalarOrArray v) -> array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace multiplication", v);
            }
            a.overwrite_descriptor(multiply(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rmul__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("multiplication", v);
            }
            return multiply(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__truediv__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("division", v);
            }
            return divide(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__itruediv__",
          [](array& a, const ScalarOrArray v) -> array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace division", v);
            }
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
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("division", v);
            }
            return divide(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__div__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("division", v);
            }
            return divide(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__rdiv__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("division", v);
            }
            return divide(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__floordiv__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("floor division", v);
            }
            return floor_divide(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__ifloordiv__",
          [](array& a, const ScalarOrArray v) -> array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace floor division", v);
            }
            a.overwrite_descriptor(floor_divide(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rfloordiv__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("floor division", v);
            }
            auto b = to_array(v, a.dtype());
            return floor_divide(b, a);
          },
          "other"_a)
      .def(
          "__mod__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("modulus", v);
            }
            return remainder(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__imod__",
          [](array& a, const ScalarOrArray v) -> array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace modulus", v);
            }
            a.overwrite_descriptor(remainder(a, to_array(v, a.dtype())));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rmod__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("modulus", v);
            }
            return remainder(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__eq__",
          [](const array& a,
             const ScalarOrArray& v) -> std::variant<array, bool> {
            if (!is_comparable_with_array(v)) {
              return false;
            }
            return equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__lt__",
          [](const array& a, const ScalarOrArray v) -> array {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("less than", v);
            }
            return less(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__le__",
          [](const array& a, const ScalarOrArray v) -> array {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("less than or equal", v);
            }
            return less_equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__gt__",
          [](const array& a, const ScalarOrArray v) -> array {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("greater than", v);
            }
            return greater(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__ge__",
          [](const array& a, const ScalarOrArray v) -> array {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("greater than or equal", v);
            }
            return greater_equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__ne__",
          [](const array& a,
             const ScalarOrArray v) -> std::variant<array, bool> {
            if (!is_comparable_with_array(v)) {
              return true;
            }
            return not_equal(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def("__neg__", [](const array& a) { return -a; })
      .def("__bool__", [](array& a) { return nb::bool_(to_scalar(a)); })
      .def(
          "__repr__",
          [](array& a) {
            nb::gil_scoped_release nogil;
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
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("power", v);
            }
            return power(a, to_array(v, a.dtype()));
          },
          "other"_a)
      .def(
          "__rpow__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("power", v);
            }
            return power(to_array(v, a.dtype()), a);
          },
          "other"_a)
      .def(
          "__ipow__",
          [](array& a, const ScalarOrArray v) -> array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace power", v);
            }
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
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("bitwise and", v);
            }
            auto b = to_array(v, a.dtype());
            if (issubdtype(a.dtype(), inexact) ||
                issubdtype(b.dtype(), inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise and.");
            }
            return bitwise_and(a, b);
          },
          "other"_a)
      .def(
          "__iand__",
          [](array& a, const ScalarOrArray v) -> array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace bitwise and", v);
            }
            auto b = to_array(v, a.dtype());
            if (issubdtype(a.dtype(), inexact) ||
                issubdtype(b.dtype(), inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with bitwise and.");
            }
            a.overwrite_descriptor(bitwise_and(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__or__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("bitwise or", v);
            }
            auto b = to_array(v, a.dtype());
            if (issubdtype(a.dtype(), inexact) ||
                issubdtype(b.dtype(), inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with or bitwise or.");
            }
            return bitwise_or(a, b);
          },
          "other"_a)
      .def(
          "__ior__",
          [](array& a, const ScalarOrArray v) -> array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace bitwise or", v);
            }
            auto b = to_array(v, a.dtype());
            if (issubdtype(a.dtype(), inexact) ||
                issubdtype(b.dtype(), inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with or bitwise or.");
            }
            a.overwrite_descriptor(bitwise_or(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__lshift__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("left shift", v);
            }
            auto b = to_array(v, a.dtype());
            if (issubdtype(a.dtype(), inexact) ||
                issubdtype(b.dtype(), inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with left shift.");
            }
            return left_shift(a, b);
          },
          "other"_a)
      .def(
          "__ilshift__",
          [](array& a, const ScalarOrArray v) -> array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace left shift", v);
            }
            auto b = to_array(v, a.dtype());
            if (issubdtype(a.dtype(), inexact) ||
                issubdtype(b.dtype(), inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with or left shift.");
            }
            a.overwrite_descriptor(left_shift(a, b));
            return a;
          },
          "other"_a,
          nb::rv_policy::none)
      .def(
          "__rshift__",
          [](const array& a, const ScalarOrArray v) {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("right shift", v);
            }
            auto b = to_array(v, a.dtype());
            if (issubdtype(a.dtype(), inexact) ||
                issubdtype(b.dtype(), inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with right shift.");
            }
            return right_shift(a, b);
          },
          "other"_a)
      .def(
          "__irshift__",
          [](array& a, const ScalarOrArray v) -> array& {
            if (!is_comparable_with_array(v)) {
              throw_invalid_operation("inplace right shift", v);
            }
            auto b = to_array(v, a.dtype());
            if (issubdtype(a.dtype(), inexact) ||
                issubdtype(b.dtype(), inexact)) {
              throw std::invalid_argument(
                  "Floating point types not allowed with or right shift.");
            }
            a.overwrite_descriptor(right_shift(a, b));
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
            return flatten(a, start_axis, end_axis, s);
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
        )pbdoc")
      .def(
          "conj",
          [](const array& a, StreamOrDevice s) {
            return mlx::core::conjugate(to_array(a), s);
          },
          nb::kw_only(),
          "stream"_a = nb::none(),
          "See :func:`conj`.");
}
