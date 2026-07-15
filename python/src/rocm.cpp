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

  rocm.def(
      "moe_swiglu_sorted",
      &mx::rocm::moe_swiglu_sorted,
      nb::arg("x"),
      nb::arg("w_gate"),
      nb::arg("w_up"),
      nb::arg("w_down"),
      nb::arg("expert_ids"),
      nb::arg("stream") = nb::none(),
      R"pbdoc(
        Fused sorted-MoE SwiGLU (one host sync for gate+up+silu+down).

        x: [T, D] bf16
        w_gate, w_up: [E, D, I] bf16 (after swapaxes of Linear [E,I,D])
        w_down: [E, I, D] bf16 (after swapaxes of Linear [E,D,I])
        expert_ids: [T] uint32, sorted by expert
        Returns y: [T, D] bf16
      )pbdoc");
}
