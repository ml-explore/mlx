
// Copyright © 2023 Apple Inc.

#include <limits>
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
      "norm",
      [](const array& a, const bool keepdims, const StreamOrDevice stream) {
        return norm(a, {}, keepdims, stream);
      },
      "a"_a,
      "keepdims"_a = false,
      "stream"_a = none,
      R"pbdoc()pbdoc");

  m.def(
      "norm",
      [](const array& a,
         const int axis,
         const bool keepdims,
         const StreamOrDevice stream) {
        return norm(a, {axis}, keepdims, stream);
      },
      "a"_a,
      "axis"_a,
      "keepdims"_a = false,
      "stream"_a = none,
      R"pbdoc()pbdoc");
  m.def(
      "norm",
      [](const array& a,
         const std::vector<int>& axis,
         const bool keepdims,
         const StreamOrDevice stream) {
        return norm(a, axis, keepdims, stream);
      },
      "a"_a,
      "axis"_a,
      "keepdims"_a = false,
      "stream"_a = none,
      R"pbdoc()pbdoc");
  m.def(
      "norm",
      [](const array& a,
         const double ord,
         const bool keepdims,
         const StreamOrDevice stream) {
        if (std::isinf((float)ord) || std::isinf(ord))
          if (ord > 0)
            return norm(a, "inf", {}, keepdims, stream);
          else
            return norm(a, "-inf", {}, keepdims, stream);

        return norm(a, ord, {}, keepdims, stream);
      },
      "a"_a,
      "ord"_a,
      "keepdims"_a = false,
      "stream"_a = none,
      R"pbdoc()pbdoc");
  m.def(
      "norm",
      [](const array& a,
         const double ord,
         const int axis,
         const bool keepdims,
         const StreamOrDevice stream) {
        if (std::isinf((float)ord) || std::isinf(ord))
          if (ord > 0)
            return norm(a, "inf", {axis}, keepdims, stream);
          else
            return norm(a, "-inf", {axis}, keepdims, stream);

        return norm(a, ord, {axis}, keepdims, stream);
      },
      "a"_a,
      "ord"_a,
      "axis"_a,
      "keepdims"_a = false,
      "stream"_a = none,
      R"pbdoc()pbdoc");
  m.def(
      "norm",
      [](const array& a,
         const double ord,
         const std::vector<int>& axis,
         const bool keepdims,
         const StreamOrDevice stream) {
        if (std::isinf((float)ord) || std::isinf(ord))
          if (ord > 0)
            return norm(a, "inf", axis, keepdims, stream);
          else
            return norm(a, "-inf", axis, keepdims, stream);

        return norm(a, ord, axis, keepdims, stream);
      },
      "a"_a,
      "ord"_a,
      "axis"_a,
      "keepdims"_a = false,
      "stream"_a = none,
      R"pbdoc()pbdoc");
  m.def(
      "norm",
      [](const array& a,
         const std::string& ord,
         const bool keepdims,
         const StreamOrDevice stream) {
        return norm(a, ord, {}, keepdims, stream);
      },
      "a"_a,
      "ord"_a,
      "keepdims"_a = false,
      "stream"_a = none,
      R"pbdoc()pbdoc");
  m.def(
      "norm",
      [](const array& a,
         const std::string& ord,
         const int axis,
         const bool keepdims,
         const StreamOrDevice stream) {
        return norm(a, ord, {axis}, keepdims, stream);
      },
      "a"_a,
      "ord"_a,
      "axis"_a,
      "keepdims"_a = false,
      "stream"_a = none,
      R"pbdoc()pbdoc");
  m.def(
      "norm",
      [](const array& a,
         const std::string& ord,
         const std::vector<int>& axis,
         const bool keepdims,
         const StreamOrDevice stream) {
        return norm(a, ord, axis, keepdims, stream);
      },
      "a"_a,
      "ord"_a,
      "axis"_a,
      "keepdims"_a = false,
      "stream"_a = none,
      R"pbdoc()pbdoc");
}
