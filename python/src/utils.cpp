
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
      _inner = nullptr;
    }
  }

 private:
  StreamOrDevice _s;
  StreamContextManager* _inner;
};

void init_utils(py::module_& m) {
  py::class_<PyStreamContextManager>(m, "StreamContextManager", R"pbdoc(
        A context manager for setting the current device and stream.
    
        This class is a Python wrapper around the C++ class
        `mlx::core::StreamContextManager`. It is used to set the current device
        and stream for the duration of a Python context block.
    
        Example:
        ```python
        import mlx.utils as mlx
    
        # Create a context manager for the current device and stream.
        with mlx.StreamContextManager(mx.cpu):
            # Run some code that uses the current device and stream.
            pass
        ```
    
        Args:
            s: The stream or device to set as the current device and stream.
  )pbdoc")
      .def(py::init<StreamOrDevice>(), "s"_a)
      .def("__enter__", [](PyStreamContextManager& scm) { scm.enter(); })
      .def(
          "__exit__",
          [](PyStreamContextManager& scm,
             const std::optional<py::type>& exc_type,
             const std::optional<py::object>& exc_value,
             const std::optional<py::object>& traceback) { scm.exit(); });
}