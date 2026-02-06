// Copyright © 2025 Apple Inc.

#include <nanobind/nanobind.h>

#include "mlx/backend/rocm/rocm.h"

namespace mx = mlx::core;
namespace nb = nanobind;

void init_rocm(nb::module_& m) {
  nb::module_ rocm = m.def_submodule("rocm", "mlx.rocm");

  rocm.def(
      "is_available",
      &mx::rocm::is_available,
      R"pbdoc(
      Check if the ROCm back-end is available.
      )pbdoc");
}
