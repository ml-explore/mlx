// Copyright © 2025 Apple Inc.

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <nanobind/nanobind.h>

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
  std::unique_ptr<PyFunction> callback =
      std::make_unique<Callback>(std::move(func));
  return mlx_func(
      std::move(callback), orig_func, std::vector<PyObject*>{deps.ptr()...});
}
