// Copyright © 2023-2024 Apple Inc.
#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/backend/metal/metal.h"
#include "mlx/device.h"
#include "mlx/memory.h"
#include "python/src/small_vector.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

bool DEPRECATE(const char* old_fn, const char* new_fn) {
  std::cerr << old_fn << " is deprecated and will be removed in a future "
            << "version. Use " << new_fn << " instead." << std::endl;
  return true;
}

#define DEPRECATE(oldfn, newfn) static bool dep = DEPRECATE(oldfn, newfn)

void init_metal(nb::module_& m) {
  nb::module_ metal = m.def_submodule("metal", "mlx.metal");
  metal.def(
      "is_available",
      &mx::metal::is_available,
      R"pbdoc(
      Check if the Metal back-end is available.
      )pbdoc");
  metal.def(
      "set_batch_invariant_limit",
      &mx::metal::set_batch_invariant_limit,
      "limit"_a,
      R"pbdoc(
      Select canonical reduction plans for supported short-block Metal
      inference kernels up to ``limit``. This covers dense matmul,
      transposed-weight quantized matmul, and vector SDPA queries up to length
      eight; it does not make every MLX operation batch invariant. Set to zero
      to disable.

      This is useful when token-by-token and short-block inference must
      produce identical results, for example during greedy speculative
      decoding. The setting is process-wide and should remain stable while
      concurrent work is executing. Increasing the limit can reduce
      performance.
      )pbdoc");
  metal.def(
      "get_batch_invariant_limit",
      &mx::metal::get_batch_invariant_limit,
      R"pbdoc(
      Return the configured short-block limit for supported canonical Metal
      inference kernels, or zero when disabled.
      )pbdoc");
  metal.def("get_active_memory", []() {
    DEPRECATE("mx.metal.get_active_memory", "mx.get_active_memory");
    return mx::get_active_memory();
  });
  metal.def("get_peak_memory", []() {
    DEPRECATE("mx.metal.get_peak_memory", "mx.get_peak_memory");
    return mx::get_peak_memory();
  });
  metal.def("reset_peak_memory", []() {
    DEPRECATE("mx.metal.reset_peak_memory", "mx.reset_peak_memory");
    mx::reset_peak_memory();
  });
  metal.def("get_cache_memory", []() {
    DEPRECATE("mx.metal.get_cache_memory", "mx.get_cache_memory");
    return mx::get_cache_memory();
  });
  metal.def(
      "set_memory_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_memory_limit", "mx.set_memory_limit");
        return mx::set_memory_limit(limit);
      },
      "limit"_a);
  metal.def(
      "set_cache_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_cache_limit", "mx.set_cache_limit");
        return mx::set_cache_limit(limit);
      },
      "limit"_a);
  metal.def(
      "set_wired_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_wired_limit", "mx.set_wired_limit");
        return mx::set_wired_limit(limit);
      },
      "limit"_a);
  metal.def("clear_cache", []() {
    DEPRECATE("mx.metal.clear_cache", "mx.clear_cache");
    mx::clear_cache();
  });
  metal.def(
      "start_capture",
      &mx::metal::start_capture,
      "path"_a,
      R"pbdoc(
      Start a Metal capture.

      Args:
        path (str): The path to save the capture which should have
          the extension ``.gputrace``.
      )pbdoc");
  metal.def(
      "stop_capture",
      &mx::metal::stop_capture,
      R"pbdoc(
      Stop a Metal capture.
      )pbdoc");
  metal.def("device_info", []() {
    DEPRECATE("mx.metal.device_info", "mx.device_info");
    return mx::device_info(mx::Device(mx::Device::gpu, 0));
  });
}
