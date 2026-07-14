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

  // Deterministic bump arena for future train HIP-graph capture. Does not
  // enable graphs. Opt-in from Python; default training leaves this unused.
  rocm.def(
      "train_arena_begin",
      &mx::rocm::train_arena_begin,
      nb::arg("capacity_bytes"),
      R"pbdoc(
      Allocate/arm a fixed HBM bump region for stable device addresses.
      Returns False on failure. Call train_arena_reset() before each step
      when capturing train graphs (not required for eager train).
      )pbdoc");
  rocm.def("train_arena_reset", &mx::rocm::train_arena_reset);
  rocm.def("train_arena_end", &mx::rocm::train_arena_end);
  rocm.def("train_arena_active", &mx::rocm::train_arena_active);
  rocm.def("train_arena_high_water", &mx::rocm::train_arena_high_water);
  rocm.def("train_arena_overflowed", &mx::rocm::train_arena_overflowed);
}
