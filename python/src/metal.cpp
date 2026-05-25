// Copyright © 2023-2024 Apple Inc.
#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/backend/metal/device.h"
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
  metal.def(
      "enable_profiling",
      &mx::metal::enable_profiling,
      R"pbdoc(
      Enable kernel-level GPU profiling.

      When enabled, each kernel dispatch gets its own command buffer
      so that per-kernel GPU timestamps can be measured.
      )pbdoc");
  metal.def(
      "disable_profiling",
      &mx::metal::disable_profiling,
      R"pbdoc(
      Disable kernel-level GPU profiling.
      )pbdoc");
  metal.def(
      "profiling_enabled",
      &mx::metal::profiling_enabled,
      R"pbdoc(
      Check if kernel-level GPU profiling is enabled.
      )pbdoc");
  metal.def(
      "get_kernel_stats",
      []() {
        auto stats = mx::metal::get_kernel_stats();
        nb::dict result;
        for (auto& [name, s] : stats) {
          nb::dict entry;
          entry["count"] = s.count;
          entry["total_us"] = s.total_us;
          entry["min_us"] = s.min_us;
          entry["max_us"] = s.max_us;
          entry["avg_us"] = s.count > 0 ? s.total_us / s.count : 0.0;
          result[nb::cast(name)] = entry;
        }
        return result;
      },
      R"pbdoc(
      Get per-kernel GPU timing statistics.

      Returns a dict mapping kernel names to their stats:
      ``{name: {count, total_us, min_us, max_us, avg_us}}``.
      )pbdoc");
  metal.def(
      "reset_kernel_stats",
      &mx::metal::reset_kernel_stats,
      R"pbdoc(
      Reset all kernel profiling statistics.
      )pbdoc");
}
