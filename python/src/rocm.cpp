// Copyright © 2025 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/backend/rocm/rocm.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

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
      [](const mx::array& x,
         const mx::array& w_gate,
         const mx::array& w_up,
         const mx::array& w_down,
         const mx::array& expert_ids,
         mx::StreamOrDevice s) {
        // Default empty → gpu (factory requires GPU stream).
        if (std::holds_alternative<std::monostate>(s)) {
          s = mx::Device(mx::Device::gpu);
        }
        return mx::rocm::moe_swiglu_sorted(
            x, w_gate, w_up, w_down, expert_ids, s);
      },
      "x"_a,
      "w_gate"_a,
      "w_up"_a,
      "w_down"_a,
      "expert_ids"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"pbdoc(
        Fused sorted-MoE SwiGLU (compile-safe Primitive; one host sync).

        x: [T, D] bf16
        w_gate, w_up: [E, D, I] bf16 (after swapaxes of Linear [E,I,D])
        w_down: [E, I, D] bf16 (after swapaxes of Linear [E,D,I])
        expert_ids: [T] uint32, sorted by expert
        Returns y: [T, D] bf16
      )pbdoc");

  rocm.def(
      "moe_swiglu_sorted_vjp",
      [](const mx::array& x,
         const mx::array& w_gate,
         const mx::array& w_up,
         const mx::array& w_down,
         const mx::array& expert_ids,
         const mx::array& dy,
         mx::StreamOrDevice s) {
        if (std::holds_alternative<std::monostate>(s)) {
          s = mx::Device(mx::Device::gpu);
        }
        return mx::rocm::moe_swiglu_sorted_vjp(
            x, w_gate, w_up, w_down, expert_ids, dy, s);
      },
      "x"_a,
      "w_gate"_a,
      "w_up"_a,
      "w_down"_a,
      "expert_ids"_a,
      "dy"_a,
      nb::kw_only(),
      "stream"_a = nb::none(),
      R"pbdoc(
        Fused sorted-MoE SwiGLU VJP (one host sync for recompute+grads).

        Same weight layouts as moe_swiglu_sorted. Returns
        (dx, dw_gate, dw_up, dw_down) all bf16.
      )pbdoc");
}
