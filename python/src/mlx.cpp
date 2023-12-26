// Copyright © 2023 Apple Inc.

#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

namespace py = pybind11;

void init_array(py::module_&);
void init_device(py::module_&);
void init_stream(py::module_&);
void init_metal(py::module_&);
void init_ops(py::module_&);
void init_transforms(py::module_&);
void init_random(py::module_&);
void init_fft(py::module_&);

PYBIND11_MODULE(core, m) {
  m.doc() = "mlx: A framework for machine learning on Apple silicon.";

  auto reprlib_fix = py::module_::import("mlx._reprlib_fix");

  init_device(m);
  init_stream(m);
  init_array(m);
  init_metal(m);
  init_ops(m);
  init_transforms(m);
  init_random(m);
  init_fft(m);
  m.attr("__version__") = TOSTRING(_VERSION_);
}
