// Copyright © 2023 Apple Inc.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "axpby/axpby.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mlx::core;

PYBIND11_MODULE(mlx_sample_extensions, m) {
  m.doc() = "Sample C++ and metal extensions for MLX";

  m.def(
      "axpby",
      &axpby,
      "x"_a,
      "y"_a,
      py::pos_only(),
      "alpha"_a,
      "beta"_a,
      py::kw_only(),
      "stream"_a = py::none(),
      R"pbdoc(
        Scale and sum two vectors element-wise
        ``z = alpha * x + beta * y``
        
        Follows numpy style broadcasting between ``x`` and ``y``
        Inputs are upcasted to floats if needed

        Args:
            x (array): Input array.
            y (array): Input array.
            alpha (float): Scaling factor for ``x``.
            beta (float): Scaling factor for ``y``.

        Returns:
            array: ``alpha * x + beta * y``
      )pbdoc");
}