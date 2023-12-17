
// Copyright © 2023 Apple Inc.

#include <numeric>
#include <ostream>
#include <variant>

#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlx/linalg.h"
#include "mlx/ops.h"
#include "mlx/utils.h"

#include "python/src/load.h"
#include "python/src/utils.h"

namespace py = pybind11;
using namespace py::literals;

using namespace mlx::core;
using namespace mlx::core::linalg;

void init_linalg(py::module_& parent_module) {
  auto m =
      parent_module.def_submodule("linalg", "mlx.core.linalg: Linear Algebra.");

  m.def(
      "vector_norm",
      [](const array& a,
         const std::variant<double, std::string>& ord,
         const std::variant<std::monostate, int, std::vector<int>>& axis,
         bool keepdims,
         StreamOrDevice s) {
        std::vector<int> axes = std::visit(
            overloaded{
                [](std::monostate s) { return std::vector<int>(); },
                [](int axis) { return std::vector<int>({axis}); },
                [](const std::vector<int> axes) { return axes; }},
            axis);

        if (axes.empty())
          return vector_norm(a, ord, keepdims, s);
        else
          return vector_norm(a, ord, axes, keepdims, s);
      },
      "a"_a,
      "ord"_a = 2.0,
      "axis"_a = none,
      "keepdims"_a = false,
      "stream"_a = none,
      R"pbdoc(
          Computes a vector norm.

          - If :attr:`axis`\ `= None`, :attr:`a` will be flattened before the norm is computed.
          - If :attr:`axis` is an `int` or a `tuple`, the norm will be computed over these dimensions
            and the other dimensions will be treated as batch dimensions.


          :attr:`ord` defines the vector norm that is computed. The following norms are supported:

          ======================   ===============================
          :attr:`ord`              vector norm
          ======================   ===============================
          `2` (default)            `2`-norm (see below)
          `inf`                    `max(abs(x))`
          `-inf`                   `min(abs(x))`
          `0`                      `sum(x != 0)`
          other `int` or `float`   `sum(abs(x)^{ord})^{(1 / ord)}`
          ======================   ===============================

          where `inf` refers to `float('inf')`, NumPy's `inf` object, or any equivalent object.

          Args:
              a (Tensor): tensor, flattened by default, but this behavior can be
                  controlled using :attr:`dim`.
              ord (int, float, inf, -inf, 'fro', 'nuc', optional): order of norm. Default: `2`
              axis (int, Tuple[int], optional): dimensions over which to compute
                  the norm. See above for the behavior when :attr:`dim`\ `= None`.
                  Default: `None`
              keepdims (bool, optional): If set to `True`, the reduced dimensions are retained
                  in the result as dimensions with size one. Default: `False`

          Returns:
              A real-valued tensor, even when :attr:`a` is complex.
        )pbdoc");
}
