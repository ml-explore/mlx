// Copyright  © 2024 Apple Inc.

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

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
      .def("size", &distributed::Group::size, "Get the size of the group")
      .def(
          "split",
          &distributed::Group::split,
          "color"_a,
          "key"_a = -1,
          nb::sig("def split(self, color: int, key: int = -1) -> Group"),
          R"pbdoc(
            Split the group to subgroups based on the provided color.

            Processes that use the same color go to the same group. The ``key``
            argument defines the rank in the new group. The smaller the key the
            smaller the rank. If the key is negative then the rank in the
            current group is used.

            Args:
              color (int): A value to group processes into subgroups.
              key (int, optional): A key to optionally change the rank ordering
                of the processes.
          )pbdoc")
      .def("barrier", &distributed::Group::barrier, "Make a synhronization point for all nodes in the group");

  m.def(
      "is_available",
      &distributed::is_available,
      R"pbdoc(
      Check if a communication backend is available.
      )pbdoc");

  m.def(
      "init",
      &distributed::init,
      "strict"_a = false,
      nb::sig("def init(strict: bool = False) -> Group"),
      R"pbdoc(
        Initialize the communication backend and create the global communication group.

        Args:
          strict (bool, optional): If set to False it returns a singleton group
            in case ``mx.distributed.is_available()`` returns False otherwise
            it throws a runtime error. Default: ``False``

        Returns:
          Group: The group representing all the launched processes.
      )pbdoc");

  m.def(
      "all_sum",
      &distributed::all_sum,
      "x"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def all_sum(x: array, *, group: Optional[Group] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        All reduce sum.

        Sum the ``x`` arrays from all processes in the group.

        Args:
          x (array): Input array.
          group (Group): The group of processes that will participate in the
            reduction. If set to ``None`` the global group is used. Default:
            ``None``.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

        Returns:
          array: The sum of all ``x`` arrays.
      )pbdoc");

  m.def(
      "all_gather",
      &distributed::all_gather,
      "x"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def all_gather(x: array, *, group: Optional[Group] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Gather arrays from all processes.

        Gather the ``x`` arrays from all processes in the group and concatenate
        them along the first axis. The arrays should all have the same shape.

        Args:
          x (array): Input array.
          group (Group): The group of processes that will participate in the
            gather. If set to ``None`` the global group is used. Default:
            ``None``.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

        Returns:
          array: The concatenation of all ``x`` arrays.
      )pbdoc");

  m.def(
      "send",
      &distributed::send,
      "x"_a,
      "dst"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def send(x: array, dst: int, *, group: Optional[Group] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Send an array from the current process to the process that has rank
        ``dst`` in the group.

        Args:
          x (array): Input array.
          dst (int): Rank of the destination process in the group.
          group (Group): The group of processes that will participate in the
            sned. If set to ``None`` the global group is used. Default:
            ``None``.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

        Returns:
          array: An empty array which when evaluated the send is performed.
      )pbdoc");

  m.def(
      "recv",
      &distributed::recv,
      "shape"_a,
      "dtype"_a,
      "src"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def recv(shape: Sequence[int], dtype: Dtype, src: int, *, group: Optional[Group] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Recv an array with shape ``shape`` and dtype ``dtype`` from process
        with rank ``src``.

        Args:
          shape (Tuple[int]): The shape of the array we are receiving.
          dtype (Dtype): The data type of the array we are receiving.
          src (int): Rank of the source process in the group.
          group (Group): The group of processes that will participate in the
            recv. If set to ``None`` the global group is used. Default:
            ``None``.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

        Returns:
          array: The array that was received from ``src``.
      )pbdoc");

  m.def(
      "recv_like",
      &distributed::recv_like,
      "x"_a,
      "src"_a,
      nb::kw_only(),
      "group"_a = nb::none(),
      "stream"_a = nb::none(),
      nb::sig(
          "def recv_like(x: array, src: int, *, group: Optional[Group] = None, stream: Union[None, Stream, Device] = None) -> array"),
      R"pbdoc(
        Recv an array with shape and type like ``x`` from process with rank
        ``src``.

        It is equivalent to calling ``mx.distributed.recv(x.shape, x.dtype, src)``.

        Args:
          x (array): An array defining the shape and dtype of the array we are
            receiving.
          src (int): Rank of the source process in the group.
          group (Group): The group of processes that will participate in the
            recv. If set to ``None`` the global group is used. Default:
            ``None``.
          stream (Stream, optional): Stream or device. Defaults to ``None``
            in which case the default stream of the default device is used.

        Returns:
          array: The array that was received from ``src``.
      )pbdoc");
}
