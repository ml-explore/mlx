// Copyright © 2025 Apple Inc.

#pragma once

#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>

namespace nb = nanobind;
using namespace nb::literals;

nb::callable mlx_func(nb::object func, std::vector<PyObject*> deps);

template <typename F, typename... Deps>
nb::callable mlx_func(F func, Deps&&... deps) {
  return mlx_func(
      nb::cpp_function(std::move(func)), std::vector<PyObject*>{deps.ptr()...});
}

template <typename... Deps>
nb::callable mlx_func(nb::object func, Deps&&... deps) {
  return mlx_func(std::move(func), std::vector<PyObject*>{deps.ptr()...});
}
