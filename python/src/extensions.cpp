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
      "rope",
      [](const array& x,
         int dims,
         float base,
         float scale,
         bool traditional,
         int offset,
         StreamOrDevice s /* = {} */) {
        return ext::rope(x, dims, base, scale, traditional, offset, s);
      },
      "x"_a,
      "dims"_a,
      "base"_a,
      "scale"_a,
      "traditional"_a,
      "offset"_a,
      py::kw_only(),
      "stream"_a = std::nullopt,
      R"pbdoc(
        RoPE.

        Args:
            x (array): Input array.

        Returns:
            array: The output array.
      )pbdoc");
}
