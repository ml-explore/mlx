// Copyright © 2023 Apple Inc.

#include <sstream>

#include <pybind11/pybind11.h>

#include "mlx/device.h"
#include "mlx/utils.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mlx::core;

void init_device(py::module_& m) {
  auto device_class = py::class_<Device>(m, "Device");
  py::enum_<Device::DeviceType>(m, "DeviceType")
      .value("cpu", Device::DeviceType::cpu)
      .value("gpu", Device::DeviceType::gpu)
      .export_values()
      .def(
          "__eq__",
          [](const Device::DeviceType& d1, const Device& d2) {
            return d1 == d2;
          },
          py::prepend());

  device_class.def(py::init<Device::DeviceType, int>(), "type"_a, "index"_a = 0)
      .def_readonly("type", &Device::type)
      .def(
          "__repr__",
          [](const Device& d) {
            std::ostringstream os;
            os << d;
            return os.str();
          })
      .def("__eq__", [](const Device& d1, const Device& d2) {
        return d1 == d2;
      });

  py::implicitly_convertible<Device::DeviceType, Device>();

  m.def("default_device", &default_device);
  m.def("set_default_device", &set_default_device, "device"_a);
}
