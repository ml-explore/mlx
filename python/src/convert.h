// Copyright © 2024 Apple Inc.
#pragma once

#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "mlx/array.h"
#include "mlx/ops.h"

namespace nb = nanobind;
using namespace mlx::core;

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

array nd_array_to_mlx(
    nb::ndarray<nb::ro, nb::c_contig, nb::device::cpu> nd_array,
    std::optional<Dtype> dtype);

nb::ndarray<nb::numpy> mlx_to_np_array(const array& a);
nb::ndarray<> mlx_to_dlpack(const array& a);

nb::object to_scalar(array& a);

nb::object tolist(array& a);

array create_array(ArrayInitType v, std::optional<Dtype> t);
array array_from_list(nb::list pl, std::optional<Dtype> dtype);
array array_from_list(nb::tuple pl, std::optional<Dtype> dtype);
