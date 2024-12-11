// Copyright © 2023-2024 Apple Inc.

#include <sstream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "mlx/device.h"
#include "mlx/utils.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

void init_device(nb::module_& m) {
  auto device_class = nb::class_<mx::Device>(
      m, "Device", R"pbdoc(A device to run operations on.)pbdoc");
  nb::enum_<mx::Device::DeviceType>(m, "DeviceType")
      .value("cpu", mx::Device::DeviceType::cpu)
      .value("gpu", mx::Device::DeviceType::gpu)
      .export_values()
      .def(
          "__eq__",
          [](const mx::Device::DeviceType& d, const nb::object& other) {
            if (!nb::isinstance<mx::Device>(other) &&
                !nb::isinstance<mx::Device::DeviceType>(other)) {
              return false;
            }
            return d == nb::cast<mx::Device>(other);
          });

  device_class
      .def(nb::init<mx::Device::DeviceType, int>(), "type"_a, "index"_a = 0)
      .def_ro("type", &mx::Device::type)
      .def(
          "__repr__",
          [](const mx::Device& d) {
            std::ostringstream os;
            os << d;
            return os.str();
          })
      .def("__eq__", [](const mx::Device& d, const nb::object& other) {
        if (!nb::isinstance<mx::Device>(other) &&
            !nb::isinstance<mx::Device::DeviceType>(other)) {
          return false;
        }
        return d == nb::cast<mx::Device>(other);
      });

  nb::implicitly_convertible<mx::Device::DeviceType, mx::Device>();

  m.def(
      "default_device",
      &mx::default_device,
      R"pbdoc(Get the default device.)pbdoc");
  m.def(
      "set_default_device",
      &mx::set_default_device,
      "device"_a,
      R"pbdoc(Set the default device.)pbdoc");
}
