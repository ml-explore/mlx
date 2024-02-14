
#include "mlx/utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>

namespace py = pybind11;
using namespace py::literals;
using namespace mlx::core;

// Slightly different from the original, with python context on init we are not
// in the context yet. Only create the inner context on enter then delete on
// exit.
class PyStreamContextManager {
 public:
  PyStreamContextManager(StreamOrDevice s) : _s(s) {}

  void enter() {
    _inner = StreamContextManager(_s);
  }

  void exit() {
    _inner.reset();
  }

 private:
  StreamOrDevice _s;
  std::optional<StreamContextManager> _inner = {};
};

void init_utils(py::module_& m) {
  py::class_<PyStreamContextManager>(m, "StreamContextManager")
      .def(py::init<StreamOrDevice>(), "s"_a)
      .def("__enter__", &PyStreamContextManager::enter)
      .def("__exit__", &PyStreamContextManager::exit);
}