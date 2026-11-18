// Copyright © 2026 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>

#include <optional>

#include "mlx/config.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

class PyScopedUpdate {
 public:
  PyScopedUpdate(mx::config::ConfigDict configs)
      : configs_(std::move(configs)) {}

  PyScopedUpdate& enter() {
    scoped_update_.emplace(std::move(configs_));
    return *this;
  }

  void exit(nb::args) {
    scoped_update_.reset();
  }

 private:
  mx::config::ConfigDict configs_;
  std::optional<mx::config::ScopedUpdate> scoped_update_;
};

void init_config(nb::module_& parent_module) {
  auto m = parent_module.def_submodule(
      "config", "mlx.core.config: configuration options");

  nb::class_<PyScopedUpdate>(m, "ScopedUpdate")
      .def("__enter__", &PyScopedUpdate::enter)
      .def("__exit__", &PyScopedUpdate::exit)
      .freeze();

  m.def(
      "update",
      &mx::config::update,
      "name"_a,
      "value"_a,
      nb::sig("def update(name: str, value: int) -> None"),
      R"pbdoc(
        Set the value of a process-wide configuration.

        See :ref:`environment_variables` for available configurations.

        Args:
            name (str): Name of configuration.
            value (int): Value of configuration.
      )pbdoc");
  m.def(
      "get",
      &mx::config::get,
      "name"_a,
      "default_value"_a,
      nb::sig("def get(name: str, default_value: int) -> int"),
      R"pbdoc(
        Read the value of a configuration. The lookup order is:

        1. The value set by :func:`mlx.core.config.scoped_update`.
        2. The value set by :func:`mlx.core.config.update`.
        3. The environment variable.
        4. The ``default_value``.

        See :ref:`environment_variables` for available configurations.

        Args:
            name (str): Name of configuration.
            default_value (int): Fallback value to return when not set.
      )pbdoc");
  m.def(
      "scoped_update",
      [](nb::kwargs kwargs) {
        return PyScopedUpdate(nb::cast<mx::config::ConfigDict>(kwargs));
      },
      nb::sig("def scoped_update(**configs: int) -> ScopedUpdate"),
      R"pbdoc(
        Create a context manager to set configurations for current thread.

        See :ref:`environment_variables` for available configurations.

        Args:
            **configs (dict(str, int)): Configuration names and values as keyword arguments.
      )pbdoc");
}
