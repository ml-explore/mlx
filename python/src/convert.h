// Copyright © 2024 Apple Inc.
#pragma once

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "mlx/array.h"
#include "mlx/ops.h"

namespace mx = mlx::core;
namespace nb = nanobind;

namespace nanobind {

template <>
struct ndarray_traits<mx::float16_t> {
  static constexpr bool is_complex = false;
  static constexpr bool is_float = true;
  static constexpr bool is_bool = false;
  static constexpr bool is_int = false;
  static constexpr bool is_signed = true;
};

template <>
struct ndarray_traits<mx::bfloat16_t> {
  static constexpr bool is_complex = false;
  static constexpr bool is_float = true;
  static constexpr bool is_bool = false;
  static constexpr bool is_int = false;
  static constexpr bool is_signed = true;
};

namespace detail {

template <>
struct dtype_traits<mx::float16_t> {
  static constexpr dlpack::dtype value{
      /* code */ uint8_t(nb::dlpack::dtype_code::Float),
      /* bits */ 16,
      /* lanes */ 1};
  static constexpr const char* name = "float16";
};

template <>
struct dtype_traits<mx::bfloat16_t> {
  static constexpr dlpack::dtype value{
      /* code */ uint8_t(nb::dlpack::dtype_code::Bfloat),
      /* bits */ 16,
      /* lanes */ 1};
  static constexpr const char* name = "bfloat16";
};

} // namespace detail

} // namespace nanobind

struct ArrayLike {
  ArrayLike(nb::object obj) : obj(obj) {};
  nb::object obj;
};

mx::array nd_array_to_mlx(
    nb::ndarray<nb::ro, nb::c_contig> nd_array,
    std::optional<mx::Dtype> mx_dtype,
    std::optional<nb::dlpack::dtype> nb_dtype = std::nullopt);

nb::ndarray<nb::numpy> mlx_to_np_array(const mx::array& a);
nb::ndarray<> mlx_to_dlpack(const mx::array& a);

nb::object to_scalar(mx::array& a);

nb::object tolist(mx::array& a);

mx::array create_array(nb::object v, std::optional<mx::Dtype> t);
mx::array array_from_list(nb::list pl, std::optional<mx::Dtype> dtype);
mx::array array_from_list(nb::tuple pl, std::optional<mx::Dtype> dtype);

// Narrow a Python-side shape dimension (int64) to a C++ mx::ShapeElem (int32),
// raising a clear error if the value would overflow.
int check_shape_dim(int64_t dim);
