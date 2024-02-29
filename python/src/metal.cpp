// Copyright © 2023 Apple Inc.

#include <pybind11/pybind11.h>

#include "mlx/backend/metal/metal.h"

namespace py = pybind11;
using namespace py::literals;

using namespace mlx::core;

void init_metal(py::module_& m) {
  py::module_ metal = m.def_submodule("metal", "mlx.metal");
  metal.def(
      "is_available",
      &metal::is_available,
      R"pbdoc(
      Check if the Metal back-end is available.
      )pbdoc");
  metal.def(
      "get_active_memory",
      &metal::get_active_memory,
      R"pbdoc(
      Get the actively used memory in bytes.

      Note, this will not always match memory use reported by the system because
      it does not include cached memory buffers.
      )pbdoc");
  metal.def(
      "get_peak_memory",
      &metal::get_peak_memory,
      R"pbdoc(
      Get the peak amount of used memory in bytes.

      The maximum memory used is recorded from the beginning of the program
      execution.
      )pbdoc");
  metal.def(
      "set_memory_limit",
      &metal::set_memory_limit,
      "limit"_a,
      py::kw_only(),
      "relaxed"_a = false,
      R"pbdoc(
      Set the memory limit.

      Memory allocations will wait on scheduled tasks to complete if the limit
      is exceeded. If there are no more scheduled tasks an error will be raised
      if ``relaxed`` is ``False``. Otherwise memory will be allocated
      (including the potential for swap) if ``relaxed`` is ``True``.

      The memory limit defaults to 1.5 times the maximum recommended working set
      size reported by the device.

      Args:
        limit (int): Memory limit in bytes.
        relaxed (bool, optional): If `False`` an error is raised if the limit
          is exceeded. Default: ``True``

      Returns:
        int: The previous memory limit in bytes.
      )pbdoc");
  metal.def(
      "set_gc_limit",
      &metal::set_gc_limit,
      "limit"_a,
      R"pbdoc(
      Set the garbage collection limit.

      If using more than the given limit, free memory will be reclaimed
      from the garbage collector on allocation. To disable the garbage collector,
      set the limit to ``0``.

      The gc limit defaults to .95 times the maximum recommended working set
      size reported by the device.

      Args:
        limit (int): Garbage collection limit in bytes.

      Returns:
        int: The previous garbage collection limit in bytes.
      )pbdoc");
}
