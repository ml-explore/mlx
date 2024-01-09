// Copyright © 2023 Apple Inc.

#pragma once

#include <pybind11/pybind11.h>

#include "mlx/array.h"
#include "python/src/utils.h"

namespace py = pybind11;
using namespace mlx::core;

array mlx_get_item(const array& src, const py::object& obj);
void mlx_set_item(array& src, const py::object& obj, const ScalarOrArray& v);
array mlx_add_item(
    const array& src,
    const py::object& obj,
    const ScalarOrArray& v);
array mlx_subtract_item(
    const array& src,
    const py::object& obj,
    const ScalarOrArray& v);
array mlx_multiply_item(
    const array& src,
    const py::object& obj,
    const ScalarOrArray& v);
array mlx_divide_item(
    const array& src,
    const py::object& obj,
    const ScalarOrArray& v);
array mlx_maximum_item(
    const array& src,
    const py::object& obj,
    const ScalarOrArray& v);
array mlx_minimum_item(
    const array& src,
    const py::object& obj,
    const ScalarOrArray& v);
