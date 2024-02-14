
#include "mlx/utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>

namespace py = pybind11;
using namespace py::literals;
using namespace mlx::core;

// Slightly different from the original, with python context on init we are not
// in the context yet. Only create the inner context on enter then de   lete on
// exit.
class PyStreamContextManager {
 public:
  PyStreamContextManager(StreamOrDevice s) : _s(s), _inner(nullptr) {}

  void enter() {
    _inner = new StreamContextManager(_s);
  }

  void exit() {
    if (_inner != nullptr) {
      delete _inner;
    }
  }

 private:
  StreamOrDevice _s;
  StreamContextManager* _inner;
};

void init_utils(py::module_& m) {
  py::class_<PyStreamContextManager>(m, "StreamContextManager")
      .def(py::init<StreamOrDevice>(), "s"_a)
      .def("__enter__", [](PyStreamContextManager& scm) { scm.enter(); })
      .def(
          "__exit__",
          [](PyStreamContextManager& scm,
             const std::optional<py::type>& exc_type,
             const std::optional<py::object>& exc_value,
             const std::optional<py::object>& traceback) { scm.exit(); });
}