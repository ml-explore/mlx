// Copyright © 2024 Apple Inc.
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <fstream>

#include "mlx/array.h"
#include "mlx/export.h"
#include "mlx/graph_utils.h"
#include "python/src/trees.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

template <typename T>
bool check_arrs(const T& iterable) {
  for (auto it = iterable.begin(); it != iterable.end(); ++it) {
    if (!nb::isinstance<mx::array>(*it)) {
      return false;
    }
  }
  return true;
};

bool valid_inputs(const nb::args& inputs) {
  if (inputs.size() > 0 && nb::isinstance<mx::array>(inputs[0])) {
    return check_arrs(inputs);
  } else if (inputs.size() > 1) {
    return false;
  } else if (inputs.size() == 1) {
    if (nb::isinstance<nb::list>(inputs[0])) {
      return check_arrs(nb::cast<nb::list>(inputs[0]));
    } else if (nb::isinstance<nb::tuple>(inputs[0])) {
      return check_arrs(nb::cast<nb::tuple>(inputs[0]));
    } else {
      return false;
    }
  }
  return true;
}

bool valid_outputs(const nb::object& outputs) {
  if (nb::isinstance<mx::array>(outputs)) {
    return true;
  } else if (nb::isinstance<nb::list>(outputs)) {
    return check_arrs(nb::cast<nb::list>(outputs));
  } else if (nb::isinstance<nb::tuple>(outputs)) {
    return check_arrs(nb::cast<nb::tuple>(outputs));
  }
  return false;
}

void init_export(nb::module_& m) {
  m.def(
      "export_function",
      [](const std::string& path,
         const nb::callable& fun,
         const nb::args& arrays,
         bool shapeless) {
        if (!valid_inputs(arrays)) {
          throw std::invalid_argument(
              "[export_function] Inputs can be either a variable number "
              "of arrays or a single tuple or list of arrays.");
        }

        std::vector<mx::array> inputs = tree_flatten(arrays, true);
        auto wrapped_fun = [&fun, &arrays](const std::vector<mx::array>& inputs)
            -> std::vector<mx::array> {
          auto outputs = fun(*tree_unflatten(arrays, inputs));
          if (!valid_outputs(outputs)) {
            throw std::invalid_argument(
                "[export_function] Outputs can be either a variable number "
                "of arrays or a single tuple or list of arrays.");
          }
          return tree_flatten(outputs, true);
        };
        mx::export_function(path, wrapped_fun, inputs, shapeless);
      },
      "path"_a,
      "fun"_a,
      "arrays"_a,
      nb::kw_only(),
      "shapeless"_a = false,
      nb::sig(
          "def export_function(path: str, fun: Callable, **arrays: array, *, shapeless: bool = False)"),
      R"pbdoc(
        Export a function to a file.

        Args:
            path (str): Path to export the function to.
            fun (Callable): A function which takes as input zero or more
              :class:`array` and returns one or more :class:`array`.
            *arrays (array): Array inputs to the function.
            shapeless (bool, optional): Whether or not the function allows
              changing the shapes of inputs.
      )pbdoc");
  m.def(
      "import_function",
      [](const std::string& path) {
        return nb::cpp_function(
            [fn = mx::import_function(path)](const nb::args& arrays) {
              if (!valid_inputs(arrays)) {
                throw std::invalid_argument(
                    "[import_function::call] Inputs can be either a variable "
                    "number of arrays or a single tuple or list of arrays.");
              }
              return nb::tuple(nb::cast(fn(tree_flatten(arrays, true))));
            });
      },
      "path"_a,
      nb::sig("def import_function(path: str) -> Callable"),
      R"pbdoc(
        Import a function from a file.

        Args:
            path (str): Path to import the function from.

        Returns:
            Callable: The imported function.
      )pbdoc");
  m.def(
      "export_to_dot",
      [](nb::object file, const nb::args& args) {
        std::vector<mx::array> arrays = tree_flatten(args);
        if (nb::isinstance<nb::str>(file)) {
          std::ofstream out(nb::cast<std::string>(file));
          mx::export_to_dot(out, arrays);
        } else if (nb::hasattr(file, "write")) {
          std::ostringstream out;
          mx::export_to_dot(out, arrays);
          auto write = file.attr("write");
          write(out.str());
        } else {
          throw std::invalid_argument(
              "[export_to_dot] Accepts file-like objects or strings "
              "to be used as filenames.");
        }
      },
      "file"_a,
      "args"_a);
}
