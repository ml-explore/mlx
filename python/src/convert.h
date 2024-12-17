// Copyright © 2024 Apple Inc.
#pragma once

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "mlx/array.h"
#include "mlx/ops.h"

namespace mx = mlx::core;
namespace nb = nanobind;

struct ArrayLike {
  ArrayLike(nb::object obj) : obj(obj) {};
  nb::object obj;
};

using ArrayInitType = std::variant<
    nb::bool_,
    nb::int_,
    nb::float_,
    // Must be above ndarray
    mx::array,
    // Must be above complex
    nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu>,
    std::complex<float>,
    nb::list,
    nb::tuple,
    ArrayLike>;

mx::array nd_array_to_mlx(
    nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> nd_array,
    std::optional<mx::Dtype> dtype);

nb::ndarray<nb::numpy> mlx_to_np_array(const mx::array& a);
nb::ndarray<> mlx_to_dlpack(const mx::array& a);

nb::object to_scalar(mx::array& a);

nb::object tolist(mx::array& a);

mx::array create_array(ArrayInitType v, std::optional<mx::Dtype> t);
mx::array array_from_list(nb::list pl, std::optional<mx::Dtype> dtype);
mx::array array_from_list(nb::tuple pl, std::optional<mx::Dtype> dtype);
