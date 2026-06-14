// Copyright © 2024 Apple Inc.
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <fstream>

#include "mlx/array.h"
#include "mlx/export.h"
#include "mlx/graph_utils.h"
#include "python/src/small_vector.h"
#include "python/src/trees.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

std::pair<mx::Args, mx::Kwargs> validate_and_extract_inputs(
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
  mx::Args args_;
  mx::Kwargs kwargs_;
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

int py_function_exporter_tp_traverse(
    PyObject* self,
    visitproc visit,
    void* arg);

class PyFunctionExporter {
 public:
  PyFunctionExporter(mx::FunctionExporter exporter, nb::handle dep)
      : exporter_(std::move(exporter)), dep_(dep) {}
  ~PyFunctionExporter() {
    nb::gil_scoped_acquire gil;
  }
  PyFunctionExporter(const PyFunctionExporter&) = delete;
  PyFunctionExporter& operator=(const PyFunctionExporter&) = delete;
  PyFunctionExporter& operator=(const PyFunctionExporter&&) = delete;
  PyFunctionExporter(PyFunctionExporter&& other)
      : exporter_(std::move(other.exporter_)), dep_(std::move(other.dep_)) {}

  void close() {
    exporter_.close();
  }
  void operator()(const mx::Args& args, const mx::Kwargs& kwargs) {
    exporter_(args, kwargs);
  }

  friend int py_function_exporter_tp_traverse(PyObject*, visitproc, void*);

 private:
  mx::FunctionExporter exporter_;
  nb::handle dep_;
};

int py_function_exporter_tp_traverse(
    PyObject* self,
    visitproc visit,
    void* arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) {
    return 0;
  }
  auto* p = nb::inst_ptr<PyFunctionExporter>(self);
  Py_VISIT(p->dep_.ptr());
  return 0;
}

PyType_Slot py_function_exporter_slots[] = {
    {Py_tp_traverse, (void*)py_function_exporter_tp_traverse},
    {0, 0}};

auto wrap_export_function(nb::callable fun) {
  return
      [fun = std::move(fun)](const mx::Args& args_, const mx::Kwargs& kwargs_) {
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
      [](nb::object& file_or_callback,
         const nb::callable& fun,
         const nb::args& args,
         bool shapeless,
         const std::optional<std::unordered_map<std::string, std::string>>&
             metadata,
         const nb::kwargs& kwargs) {
        auto [args_, kwargs_] =
            validate_and_extract_inputs(args, kwargs, "[export_function]");
        if (nb::isinstance<nb::str>(file_or_callback)) {
          mx::export_function(
              nb::cast<std::string>(file_or_callback),
              wrap_export_function(fun),
              args_,
              kwargs_,
              shapeless,
              metadata.value_or(
                  std::unordered_map<std::string, std::string>{}));
        } else {
          if (metadata && !metadata->empty()) {
            throw std::invalid_argument(
                "[export_function] The metadata argument is only supported "
                "when exporting to a file, not when using a callback.");
          }
          auto callback = nb::cast<nb::callable>(file_or_callback);
          auto wrapped_callback =
              [callback](const mx::ExportCallbackInput& input) {
                return callback(input);
              };
          mx::export_function(
              callback, wrap_export_function(fun), args_, kwargs_, shapeless);
        }
      },
      nb::arg(),
      "fun"_a,
      "args"_a,
      nb::kw_only(),
      "shapeless"_a = false,
      "metadata"_a = nb::none(),
      "kwargs"_a,
      nb::sig(
          "def export_function(file_or_callback: Union[str, Callable], fun: Callable, *args, shapeless: bool = False, metadata: Optional[dict[str, str]] = None, **kwargs) -> None"),
      R"pbdoc(
        Export an MLX function.

        Example input arrays must be provided to export a function. The example
        inputs can be variable ``*args`` and ``**kwargs`` or a tuple of arrays
        and/or dictionary of string keys with array values.

        .. warning::

          This is part of an experimental API which is likely to
          change in future versions of MLX. Functions exported with older
          versions of MLX may not be compatible with future versions.

        Args:
            file_or_callback (str or Callable): Either a file path to export
              the function to or a callback.
            fun (Callable): A function which takes as input zero or more
              :class:`array` and returns one or more :class:`array`.
            *args (array): Example array inputs to the function.
            shapeless (bool, optional): Whether or not the function allows
              inputs with variable shapes. Default: ``False``.
            metadata (dict, optional): A dictionary of string keys and string
              values to save alongside the function. Only supported when
              exporting to a file. The metadata can be read back with
              :func:`import_function`. Default: ``None``.
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
      [](const std::string& file, bool return_metadata) -> nb::object {
        auto imported = mx::import_function(file);
        auto metadata = imported.metadata();
        auto fn = nb::cpp_function(
            [imported = std::move(imported)](
                const nb::args& args, const nb::kwargs& kwargs) {
              auto [args_, kwargs_] = validate_and_extract_inputs(
                  args, kwargs, "[import_function::call]");
              return nb::tuple(nb::cast(imported(args_, kwargs_)));
            });
        if (return_metadata) {
          return nb::make_tuple(fn, nb::cast(metadata));
        }
        return fn;
      },
      "file"_a,
      nb::kw_only(),
      "return_metadata"_a = false,
      nb::sig(
          "def import_function(file: str, *, return_metadata: bool = False) -> Union[Callable, tuple[Callable, dict[str, str]]]"),
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
            return_metadata (bool, optional): If ``True`` also return the
              metadata that was saved with the function as a dictionary of
              string keys and values. Default: ``False``.

        Returns:
            Callable: The imported function. If ``return_metadata`` is
            ``True`` a tuple of the imported function and a dictionary of
            metadata is returned instead.

        Example:
          >>> fn = mx.import_function("function.mlxfn")
          >>> out = fn(a, b, x=x, y=y)[0]
          >>>
          >>> out = fn((a, b), {"x": x, "y": y}[0]
      )pbdoc");

  nb::class_<PyFunctionExporter>(
      m,
      "FunctionExporter",
      nb::type_slots(py_function_exporter_slots),
      R"pbdoc(
       A context managing class for exporting multiple traces of the same
       function to a file.

       Make an instance of this class by calling fun:`mx.exporter`.
      )pbdoc")
      .def("close", &PyFunctionExporter::close)
      .def("__enter__", [](PyFunctionExporter& exporter) { return &exporter; })
      .def(
          "__exit__",
          [](PyFunctionExporter& exporter,
             const std::optional<nb::object>&,
             const std::optional<nb::object>&,
             const std::optional<nb::object>&) { exporter.close(); },
          "exc_type"_a = nb::none(),
          "exc_value"_a = nb::none(),
          "traceback"_a = nb::none())
      .def(
          "__call__",
          [](PyFunctionExporter& exporter,
             const nb::args& args,
             const nb::kwargs& kwargs) {
            auto [args_, kwargs_] =
                validate_and_extract_inputs(args, kwargs, "[export_function]");
            exporter(args_, kwargs_);
          });

  m.def(
      "exporter",
      [](const std::string& file,
         nb::callable fun,
         bool shapeless,
         const std::optional<std::unordered_map<std::string, std::string>>&
             metadata) {
        return PyFunctionExporter{
            mx::exporter(
                file,
                wrap_export_function(fun),
                shapeless,
                metadata.value_or(
                    std::unordered_map<std::string, std::string>{})),
            fun};
      },
      "file"_a,
      "fun"_a,
      nb::kw_only(),
      "shapeless"_a = false,
      "metadata"_a = nb::none(),
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
            metadata (dict, optional): A dictionary of string keys and string
              values to save alongside the function. The metadata can be read
              back with :func:`import_function`. Default: ``None``.

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
        std::vector<mx::array> arrays =
            tree_flatten(nb::make_tuple(args, kwargs));
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
