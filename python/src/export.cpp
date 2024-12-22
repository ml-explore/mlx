// Copyright © 2024 Apple Inc.
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
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

std::pair<std::vector<mx::array>, std::map<std::string, mx::array>>
validate_and_extract_inputs(
    const nb::args& args,
    const nb::kwargs& kwargs,
    const std::string& prefix) {
  auto maybe_throw = [&prefix](bool valid) {
    if (!valid) {
      throw std::invalid_argument(
          prefix +
          " Inputs can either be a variable "
          "number of positional and keyword arrays or a single tuple "
          "and/or dictionary of arrays.");
    }
  };
  std::vector<mx::array> args_;
  std::map<std::string, mx::array> kwargs_;
  if (args.size() == 0) {
    // No args so kwargs must be keyword arrays
    maybe_throw(nb::try_cast(kwargs, kwargs_));
  } else if (args.size() > 0 && nb::isinstance<mx::array>(args[0])) {
    // Args are positional arrays and kwargs are keyword arrays
    maybe_throw(nb::try_cast(args, args_));
    maybe_throw(nb::try_cast(kwargs, kwargs_));
  } else if (args.size() == 1) {
    // - args[0] can be a tuple or list or arrays or a dict
    //   with string keys and array values
    // - kwargs should be empty
    maybe_throw(kwargs.size() == 0);
    if (!nb::try_cast(args[0], args_)) {
      maybe_throw(nb::try_cast(args[0], kwargs_));
    }
  } else if (args.size() == 2) {
    // - args[0] can be a tuple or list of arrays
    // - args[1] can be a dict of string keys with array values.
    // - kwargs should be empty
    maybe_throw(kwargs.size() == 0);
    maybe_throw(nb::try_cast(args[0], args_));
    maybe_throw(nb::try_cast(args[1], kwargs_));
  } else {
    maybe_throw(false);
  }
  return {args_, kwargs_};
}

auto wrap_export_function(const nb::callable& fun) {
  return [fun](
             const std::vector<mx::array>& args_,
             const std::map<std::string, mx::array>& kwargs_) {
    auto kwargs = nb::dict();
    kwargs.update(nb::cast(kwargs_));
    auto args = nb::tuple(nb::cast(args_));
    auto outputs = fun(*args, **kwargs);
    std::vector<mx::array> outputs_;
    if (nb::isinstance<mx::array>(outputs)) {
      outputs_.push_back(nb::cast<mx::array>(outputs));
    } else if (!nb::try_cast(outputs, outputs_)) {
      throw std::invalid_argument(
          "[export_function] Outputs can be either a single array "
          "a tuple or list of arrays.");
    }
    return outputs_;
  };
}

void init_export(nb::module_& m) {
  m.def(
      "export_function",
      [](const std::string& path,
         const nb::callable& fun,
         const nb::args& args,
         bool shapeless,
         const nb::kwargs& kwargs) {
        auto [args_, kwargs_] =
            validate_and_extract_inputs(args, kwargs, "[export_function]");
        mx::export_function(
            path, wrap_export_function(fun), args_, kwargs_, shapeless);
      },
      "path"_a,
      "fun"_a,
      "args"_a,
      nb::kw_only(),
      "shapeless"_a = false,
      "kwargs"_a,
      nb::sig(
          "def export_function(path: str, fun: Callable, *args, *, shapeless: bool = False, **kwargs)"),
      R"pbdoc(
        Export a function to a file.

        To export ``fun`` Example input arrays must be provided. The arrays
        can be either variable ``*args`` and ``**kwargs`` or a tuple of arrays and/or
        dictionary of string keys with array values.

        Args:
            path (str): Path to export the function to.
            fun (Callable): A function which takes as input zero or more
              :class:`array` and returns one or more :class:`array`.
            *args (array): Example array inputs to the function.
            shapeless (bool, optional): Whether or not the function allows
              changing the shapes of inputs.
            **kwargs (array): Additional example keyword array inputs to the function.
      )pbdoc");
  m.def(
      "import_function",
      [](const std::string& path) {
        return nb::cpp_function(
            [fn = mx::import_function(path)](
                const nb::args& args, const nb::kwargs& kwargs) {
              auto [args_, kwargs_] = validate_and_extract_inputs(
                  args, kwargs, "[import_function::call]");
              return nb::tuple(nb::cast(fn(args_, kwargs_)));
            });
      },
      "path"_a,
      R"pbdoc(
        Import a function from a file.

        The imported function can be called either with variable ``*args`` and
        ``**kwargs`` or with a tuple of arrays and/or dictionary of string
        keys with array values.

        Args:
            path (str): Path to import the function from.

        Returns:
            Callable: The imported function.

        Example:
          >>> fn = mx.import("function.mlxfn")
          >>> out = fn(a, b, x=x, y=y)[0]
          >>>
          >>> out = fn((a, b), {"x": x, "y": y}[0]
      )pbdoc");

  nb::class_<mx::FunctionExporter>(m, "FunctionExporter")
      .def("close", &mx::FunctionExporter::close)
      .def(
          "__enter__", [](mx::FunctionExporter& exporter) { return &exporter; })
      .def(
          "__exit__",
          [](mx::FunctionExporter& exporter,
             const std::optional<nb::object>&,
             const std::optional<nb::object>&,
             const std::optional<nb::object>&) { exporter.close(); },
          "exc_type"_a = nb::none(),
          "exc_value"_a = nb::none(),
          "traceback"_a = nb::none())
      .def(
          "__call__",
          [](mx::FunctionExporter& exporter,
             const nb::args& args,
             const nb::kwargs& kwargs) {
            auto [args_, kwargs_] =
                validate_and_extract_inputs(args, kwargs, "[export_function]");
            exporter(args_, kwargs_);
          });

  m.def(
      "exporter",
      [](const std::string& path, const nb::callable& fun, bool shapeless) {
        return mx::exporter(path, wrap_export_function(fun), shapeless);
      },
      "path"_a,
      "fun"_a,
      nb::kw_only(),
      "shapeless"_a = false);

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
