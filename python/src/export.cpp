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
      [](const std::string& file,
         const nb::callable& fun,
         const nb::args& args,
         bool shapeless,
         const nb::kwargs& kwargs) {
        auto [args_, kwargs_] =
            validate_and_extract_inputs(args, kwargs, "[export_function]");
        mx::export_function(
            file, wrap_export_function(fun), args_, kwargs_, shapeless);
      },
      "file"_a,
      "fun"_a,
      "args"_a,
      nb::kw_only(),
      "shapeless"_a = false,
      "kwargs"_a,
      R"pbdoc(
        Export a function to a file.

        Example input arrays must be provided to export a function. The example
        inputs can be variable ``*args`` and ``**kwargs`` or a tuple of arrays
        and/or dictionary of string keys with array values.

        .. warning::

          This is part of an experimental API which is likely to
          change in future versions of MLX. Functions exported with older
          versions of MLX may not be compatible with future versions.

        Args:
            file (str): File path to export the function to.
            fun (Callable): A function which takes as input zero or more
              :class:`array` and returns one or more :class:`array`.
            *args (array): Example array inputs to the function.
            shapeless (bool, optional): Whether or not the function allows
              inputs with variable shapes. Default: ``False``.
            **kwargs (array): Additional example keyword array inputs to the
              function.

        Example:

          .. code-block:: python

            def fun(x, y):
                return x + y

            x = mx.array(1)
            y = mx.array([1, 2, 3])
            mx.export_function("fun.mlxfn", fun, x, y=y)
      )pbdoc");
  m.def(
      "import_function",
      [](const std::string& file) {
        return nb::cpp_function(
            [fn = mx::import_function(file)](
                const nb::args& args, const nb::kwargs& kwargs) {
              auto [args_, kwargs_] = validate_and_extract_inputs(
                  args, kwargs, "[import_function::call]");
              return nb::tuple(nb::cast(fn(args_, kwargs_)));
            });
      },
      "file"_a,
      nb::sig("def import_function(file: str) -> Callable"),
      R"pbdoc(
        Import a function from a file.

        The imported function can be called either with ``*args`` and
        ``**kwargs`` or with a tuple of arrays and/or dictionary of string
        keys with array values. Imported functions always return a tuple of
        arrays.

        .. warning::

          This is part of an experimental API which is likely to
          change in future versions of MLX. Functions exported with older
          versions of MLX may not be compatible with future versions.

        Args:
            file (str): The file path to import the function from.

        Returns:
            Callable: The imported function.

        Example:
          >>> fn = mx.import_function("function.mlxfn")
          >>> out = fn(a, b, x=x, y=y)[0]
          >>>
          >>> out = fn((a, b), {"x": x, "y": y}[0]
      )pbdoc");

  nb::class_<mx::FunctionExporter>(
      m,
      "FunctionExporter",
      R"pbdoc(
       A context managing class for exporting multiple traces of the same
       function to a file.

       Make an instance of this class by calling fun:`mx.exporter`.
      )pbdoc")
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
      [](const std::string& file, const nb::callable& fun, bool shapeless) {
        return mx::exporter(file, wrap_export_function(fun), shapeless);
      },
      "file"_a,
      "fun"_a,
      nb::kw_only(),
      "shapeless"_a = false,
      R"pbdoc(
        Make a callable object to export multiple traces of a function to a file.

        .. warning::

          This is part of an experimental API which is likely to
          change in future versions of MLX. Functions exported with older
          versions of MLX may not be compatible with future versions.

        Args:
            file (str): File path to export the function to.
            shapeless (bool, optional): Whether or not the function allows
              inputs with variable shapes. Default: ``False``.

        Example:

          .. code-block:: python

            def fun(*args):
                return sum(args)

            with mx.exporter("fun.mlxfn", fun) as exporter:
                exporter(mx.array(1))
                exporter(mx.array(1), mx.array(2))
                exporter(mx.array(1), mx.array(2), mx.array(3))
      )pbdoc");
  m.def(
      "export_to_dot",
      [](nb::object file, const nb::args& args, const nb::kwargs& kwargs) {
        std::vector<mx::array> arrays = tree_flatten(args);
        mx::NodeNamer namer;
        for (const auto& n : kwargs) {
          namer.set_name(
              nb::cast<mx::array>(n.second), nb::cast<std::string>(n.first));
        }
        if (nb::isinstance<nb::str>(file)) {
          std::ofstream out(nb::cast<std::string>(file));
          mx::export_to_dot(out, std::move(namer), arrays);
        } else if (nb::hasattr(file, "write")) {
          std::ostringstream out;
          mx::export_to_dot(out, std::move(namer), arrays);
          auto write = file.attr("write");
          write(out.str());
        } else {
          throw std::invalid_argument(
              "[export_to_dot] Accepts file-like objects or strings "
              "to be used as filenames.");
        }
      },
      "file"_a,
      "args"_a,
      "kwargs"_a,
      R"pbdoc(
        Export a graph to DOT format for visualization.

        A variable number of output arrays can be provided for exporting
        The graph exported will recursively include all unevaluated inputs of
        the provided outputs.

        Args:
            file (str): The file path to export to.
            *args (array): The output arrays.
            **kwargs (dict[str, array]): Provide some names for arrays in the
              graph to make the result easier to parse.

        Example:
          >>> a = mx.array(1) + mx.array(2)
          >>> mx.export_to_dot("graph.dot", a)
          >>> x = mx.array(1)
          >>> y = mx.array(2)
          >>> mx.export_to_dot("graph.dot", x + y, x=x, y=y)
      )pbdoc");
}
