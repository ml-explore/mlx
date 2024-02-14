// Copyright © 2023 Apple Inc.

#include <sstream>

#include <pybind11/pybind11.h>

#include "mlx/stream.h"
#include "mlx/utils.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mlx::core;

void init_stream(py::module_& m) {
  py::class_<Stream>(
      m,
      "Stream",
      R"pbdoc(
      A stream for running operations on a given device.
      )pbdoc")
      .def(py::init<int, Device>(), "index"_a, "device"_a)
      .def_readonly("device", &Stream::device)
      .def(
          "__repr__",
          [](const Stream& s) {
            std::ostringstream os;
            os << s;
            return os.str();
          })
      .def("__eq__", [](const Stream& s1, const Stream& s2) {
        return s1 == s2;
      });

  py::implicitly_convertible<Device::DeviceType, Device>();

  m.def(
      "default_stream",
      &default_stream,
      "device"_a,
      R"pbdoc(Get the device's default stream.)pbdoc");
  m.def(
      "set_default_stream",
      &set_default_stream,
      "stream"_a,
      R"pbdoc(
        Set the default stream.

        This will make the given stream the default for the
        streams device. It will not change the default device.

        Args:
          stream (stream): Stream to make the default.
      )pbdoc");
  m.def(
      "new_stream",
      &new_stream,
      "device"_a,
      R"pbdoc(Make a new stream on the given device.)pbdoc");
}
