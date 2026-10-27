// Copyright © 2023-2025 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

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
  cuda.def(
      "get_arch_list",
      &mx::cu::get_arch_list,
      R"pbdoc(
      Get the CUDA architectures MLX was compiled for.

      Architectures use NVIDIA's ``sm_*`` (binary) and ``compute_*`` (PTX)
      notation, for example ``["sm_80", "sm_120a", "compute_120"]``.

      Returns:
          list(str): The compiled architectures, or an empty list if CUDA
          backend is not enabled.
      )pbdoc");
}
