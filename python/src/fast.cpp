// Copyright © 2023-2024 Apple Inc.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "python/src/utils.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mlx::core;

void init_extensions(py::module_& parent_module) {
  py::options options;
  options.disable_function_signatures();

  auto m =
      parent_module.def_submodule("fast", "mlx.core.fast: fast operations");

  m.def(
      "rope",
      [](const array& a,
         int dims,
         bool traditional,
         float base,
         float scale,
         int offset,
         const StreamOrDevice& s /* = {} */) {
        return fast::rope(a, dims, traditional, base, scale, offset, s);
      },
      "a"_a,
      "dims"_a,
      py::kw_only(),
      "traditional"_a,
      "base"_a,
      "scale"_a,
      "offset"_a,
      "stream"_a = none,
      R"pbdoc(
        rope(a: array, dims: int, *, traditinoal: bool, base: float, scale: float, offset: int, stream: Union[None, Stream, Device] = None) -> array

        Apply rotary positional encoding to the input.

        Args:
            a (array): Input array.
            dims (int): The feature dimensions to be rotated. If the input feature
                is larger than dims then the rest is left unchanged.
            traditional (bool): If set to ``True`` choose the traditional
                implementation which rotates consecutive dimensions.
            base (float): The base used to compute angular frequency for
                each dimension in the positional encodings.
            scale (float): The scale used to scale the positions.
            offset (int): The position offset to start at.

        Returns:
            array: The output array.
      )pbdoc");
}
