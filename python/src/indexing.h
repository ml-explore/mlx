// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <nanobind/nanobind.h>

#include "mlx/array.h"
#include "python/src/utils.h"

namespace mx = mlx::core;
namespace nb = nanobind;

mx::array mlx_get_item(const mx::array& src, const nb::object& obj);
void mlx_set_item(
    mx::array& src,
    const nb::object& obj,
    const ScalarOrArray& v);
mx::array mlx_add_item(
    const mx::array& src,
    const nb::object& obj,
    const ScalarOrArray& v);
mx::array mlx_subtract_item(
    const mx::array& src,
    const nb::object& obj,
    const ScalarOrArray& v);
mx::array mlx_multiply_item(
    const mx::array& src,
    const nb::object& obj,
    const ScalarOrArray& v);
mx::array mlx_divide_item(
    const mx::array& src,
    const nb::object& obj,
    const ScalarOrArray& v);
mx::array mlx_maximum_item(
    const mx::array& src,
    const nb::object& obj,
    const ScalarOrArray& v);
mx::array mlx_minimum_item(
    const mx::array& src,
    const nb::object& obj,
    const ScalarOrArray& v);
