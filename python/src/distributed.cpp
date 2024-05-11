// Copyright  © 2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>

#include "mlx/distributed/distributed.h"
#include "mlx/distributed/ops.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

void init_distributed(nb::module_& parent_module) {
  auto m = parent_module.def_submodule(
      "distributed", "mlx.core.distributed: Communication operations");

  nb::class_<distributed::Group>(
      m,
      "Group",
      R"pbcopy(
        An :class:`mlx.core.distributed.Group` represents a group of independent mlx
        processes that can communicate.
      )pbcopy")
      .def("rank", &distributed::Group::rank, "Get the rank of this process")
      .def("size", &distributed::Group::size, "Get the size of the group");

  m.def(
      "is_available",
      &distributed::is_available,
      R"pbdoc(
      Check if a communication backend is available.
      )pbdoc");

  m.def(
      "init",
      &distributed::init,
      R"pbdoc(
        Initialize the communication backend and create the global communication group.
      )pbdoc");

  m.def(
      "all_reduce_sum",
      [](const array& x,
         std::optional<std::shared_ptr<distributed::Group>> group) {
        return distributed::all_reduce_sum(x, group.value_or(nullptr));
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

  m.def(
      "all_gather",
      [](const array& x,
         std::optional<std::shared_ptr<distributed::Group>> group) {
        return distributed::all_gather(x, group.value_or(nullptr));
      },
      "x"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      nb::sig(
          "def all_gather(x: array, *, group: Optional[Group] = None) -> array"),
      R"pbdoc(
        Gather arrays from all processes.

        Gather the ``x`` arrays from all processes in the group and concatenate
        them along the first axis. The arrays should all have the same shape.

        Args:
          x (array): Input array.
          group (Group): The group of processes that will participate in the
            gather. If set to ``None`` the global group is used. Default:
            ``None``.

        Returns:
          array: The concatenation of all ``x`` arrays.
      )pbdoc");
}
