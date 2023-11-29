#include <sstream>

#include <pybind11/pybind11.h>

#include "mlx/stream.h"
#include "mlx/utils.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mlx::core;

void init_stream(py::module_& m) {
  py::class_<Stream>(m, "Stream")
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

  m.def("default_stream", &default_stream, "device"_a);
  m.def("set_default_stream", &set_default_stream, "stream"_a);
  m.def("new_stream", &new_stream, "device"_a);
}
