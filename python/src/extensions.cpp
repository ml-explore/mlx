// Copyright © 2023-2024 Apple Inc.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlx/extensions.h"
#include "mlx/ops.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mlx::core;

void init_extensions(py::module_& parent_module) {
  py::options options;
  options.disable_function_signatures();

  auto m = parent_module.def_submodule("ext", "mlx.core.ext: fast operations");

  m.def(
      "rms_norm",
      [](const array& x,
         const array& weight,
         float eps,
         bool precise,
         StreamOrDevice s) {
        return ext::rms_norm(x, weight, eps, precise, s);
      },
      "x"_a,
      "weight"_a,
      "eps"_a = 1e-5,
      "precise"_a = false,
      py::kw_only(),
      "stream"_a = std::nullopt,
      R"pbdoc(
        RMS norm.

        Args:
            x (array): Input array.
            weight (array): Weight array.
            eps (float, optional): Constant for numerical stability. Default is ``1e-5``.
            precise (bool, optional): Perform the normalization in ``float32``. Default: ``False``.

        Returns:
            array: The output array.
      )pbdoc");
}
