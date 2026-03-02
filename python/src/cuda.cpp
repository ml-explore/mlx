// Copyright © 2023-2025 Apple Inc.

#include <nanobind/nanobind.h>

#include "mlx/backend/cuda/cuda.h"

namespace mx = mlx::core;
namespace nb = nanobind;

void init_cuda(nb::module_& m) {
  nb::module_ cuda = m.def_submodule("cuda", "mlx.cuda");

  cuda.def(
      "is_available",
      &mx::cu::is_available,
      R"pbdoc(
      Check if the CUDA back-end is available.
      )pbdoc");
}
