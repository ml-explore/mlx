// Copyright © 2025 Apple Inc.

#pragma once

#include <nanobind/nanobind.h>

#include <memory>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

struct PyFunction {
  virtual ~PyFunction() = default;
  virtual nb::object operator()(nb::args& args, nb::kwargs& kwargs) = 0;
};

nb::callable mlx_func(
    std::unique_ptr<PyFunction> func,
    const nb::callable& orig_func,
    std::vector<PyObject*> deps);

template <typename F, typename... Deps>
nb::callable mlx_func(F func, const nb::callable& orig_func, Deps&&... deps) {
  struct Callback : PyFunction {
    F func;

    explicit Callback(F func) : func(std::move(func)) {}

    nb::object operator()(nb::args& args, nb::kwargs& kwargs) override {
      return nb::cast(func(args, kwargs));
    }
  };
  return mlx_func(
      std::make_unique<Callback>(std::move(func)), orig_func, {deps.ptr()...});
}
