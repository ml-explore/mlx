// Copyright  © 2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "mlx/dist/dist.h"
#include "mlx/dist/ops.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

void init_dist(nb::module_& parent_module) {
  auto m = parent_module.def_submodule(
      "dist", "mlx.core.dist: Communication operations");

  nb::class_<dist::Group>(
      m,
      "Group",
      R"pbcopy(
        An :class:`mlx.core.dist.Group` represents a group of independent mlx
        processes that can communicate.
      )pbcopy")
      .def("rank", &dist::Group::rank, "Get the rank of this process")
      .def("size", &dist::Group::size, "Get the size of the group");

  m.def(
      "is_available",
      &dist::is_available,
      R"pbdoc(
      Check if a communication backend is available.
      )pbdoc");

  m.def(
      "init",
      &dist::init,
      R"pbdoc(
        Initialize the communication backend and create the global communication group.
      )pbdoc");

  m.def(
      "all_reduce_sum",
      [](const array& x, std::optional<std::shared_ptr<dist::Group>> group) {
        return dist::all_reduce_sum(x, group.value_or(nullptr));
      },
      "x"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      nb::sig(
          "def all_reduce_sum(x: array, *, group: Optional[Group] = None) -> array"),
      R"pbdoc(
        All reduce sum.

        Sum the ``x`` arrays from all processes in the group.

        Args:
          x (array): Input array.
          group (Group): The group of processes that will participate in the
            reduction. If set to ``None`` the global group is used. Default:
            ``None``.

        Returns:
          array: The sum of all ``x`` arrays.
      )pbdoc");
}
