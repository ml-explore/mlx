#include <pybind11/pybind11.h>

#include "mlx/backend/metal/metal.h"

namespace py = pybind11;

using namespace mlx::core;

void init_metal(py::module_& m) {
  py::module_ metal = m.def_submodule("metal", "mlx.metal");
  metal.def("is_available", &metal::is_available);
}
