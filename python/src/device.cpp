// Copyright © 2023-2024 Apple Inc.

#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "mlx/device.h"
#include "mlx/utils.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

void init_device(nb::module_& m) {
  auto device_class = nb::class_<Device>(
      m, "Device", R"pbdoc(A device to run operations on.)pbdoc");
  nb::enum_<Device::DeviceType>(m, "DeviceType")
      .value("cpu", Device::DeviceType::cpu)
      .value("gpu", Device::DeviceType::gpu)
      .export_values()
      .def("__eq__", [](const Device::DeviceType& d, const nb::object& other) {
        if (!nb::isinstance<Device>(other) &&
            !nb::isinstance<Device::DeviceType>(other)) {
          return false;
        }
        return d == nb::cast<Device>(other);
      });

  device_class.def(nb::init<Device::DeviceType, int>(), "type"_a, "index"_a = 0)
      .def_ro("type", &Device::type)
      .def(
          "__repr__",
          [](const Device& d) {
            std::ostringstream os;
            os << d;
            return os.str();
          })
      .def("__eq__", [](const Device& d, const nb::object& other) {
        if (!nb::isinstance<Device>(other) &&
            !nb::isinstance<Device::DeviceType>(other)) {
          return false;
        }
        return d == nb::cast<Device>(other);
      });

  nb::implicitly_convertible<Device::DeviceType, Device>();

  m.def(
      "default_device",
      &default_device,
      R"pbdoc(Get the default device.)pbdoc");
  m.def(
      "set_default_device",
      &set_default_device,
      "device"_a,
      R"pbdoc(Set the default device.)pbdoc");
}
