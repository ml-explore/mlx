
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
class PyStreamContext {
 public:
  PyStreamContext(StreamOrDevice s) : _s(s), _inner(nullptr) {}

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
  py::class_<PyStreamContext>(m, "StreamContext", R"pbdoc(
        A context manager for setting the current device and stream.
        
        Args:
            s: The stream or device to set as the current device and stream.

        Example:
        .. code-block::python
          import mlx.core as mx
      
          # Create a context manager for the current device and stream.
          with mx.StreamContext(mx.cpu):
              # Run some code that uses the current device and stream.
              pass
  )pbdoc")
      .def(py::init<StreamOrDevice>(), "s"_a)
      .def("__enter__", [](PyStreamContext& scm) { scm.enter(); })
      .def(
          "__exit__",
          [](PyStreamContext& scm,
             const std::optional<py::type>& exc_type,
             const std::optional<py::object>& exc_value,
             const std::optional<py::object>& traceback) { scm.exit(); });
  m.def(
      "stream",
      [](StreamOrDevice s) { return PyStreamContext(s); },
      "s"_a,
      R"pbdoc(
        Wrap around the Context-manager StreamContext that selects a given stream.

        Args:
            s: The stream or device to set as the current device and stream.
        
        Returns:
            A context manager for setting the current device and stream.
      )pbdoc");
}