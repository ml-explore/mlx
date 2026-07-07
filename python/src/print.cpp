#include <cstdint>
#include <cstring>
#include <sstream>

#include <nanobind/typing.h>

#include "mlx/utils.h"
#include "python/src/utils.h"

#include "mlx/mlx.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

struct PrintOptionsContext {
  mx::PrintOptions old_options;
  mx::PrintOptions new_options;
  PrintOptionsContext(mx::PrintOptions p) : new_options(p) {}
  PrintOptionsContext& enter() {
    old_options = mx::get_global_formatter().format_options;
    mx::set_printoptions(new_options);
    return *this;
  }
  void exit(nb::args) {
    mx::set_printoptions(old_options);
  }
};

void init_print(nb::module_& m) {
  // Set Python print formatting options
  mx::get_global_formatter().capitalize_bool = true;
  // Expose printing options to Python: allow setting global precision.
  nb::class_<mx::PrintOptions>(m, "PrintOptions")
      .def(nb::init<int>(), "precision"_a = -1)
      .def_rw("precision", &mx::PrintOptions::precision);

  m.def(
      "set_printoptions",
      [](int precision) { mx::set_printoptions({precision}); },
      "precision"_a = mx::get_global_formatter().format_options.precision,
      R"pbdoc(
        Set global printing precision for array formatting.

        Example:
            >>> print(x)  # Uses default precision
            >>> mx.set_printoptions(precision=3):
            >>> print(x)  # Uses precision of 3
            >>> print(x)  # Uses precision of 3 (again)

        Args:
            precision (int): Number of decimal places.
        )pbdoc");
  m.def(
      "get_printoptions",
      []() { return mx::get_global_formatter().format_options; },
      R"pbdoc(
        Get global printing precision for array formatting.

        Returns:
        PrintOptions: The format options used for printing arrays.
        )pbdoc");

  nb::class_<PrintOptionsContext>(m, "_PrintOptionsContext")
      .def(nb::init<mx::PrintOptions>())
      .def("__enter__", &PrintOptionsContext::enter)
      .def("__exit__", &PrintOptionsContext::exit);

  m.def(
      "printoptions",
      [](int precision) { return PrintOptionsContext({precision}); },
      "precision"_a = mx::get_global_formatter().format_options.precision,
      R"pbdoc(
        Context manager for setting print options temporarily.

        Example:
            >>> print(x)  # Uses default precision
            >>> with mx.printoptions(precision=3):
            >>>     print(x)  # Uses precision of 3
            >>> print(x)  # Back to default precision


        Args:
            precision (int): Number of decimal places. Use -1 for default
        )pbdoc");
}
