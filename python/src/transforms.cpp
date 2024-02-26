// Copyright © 2023-2024 Apple Inc.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>

#include "mlx/array.h"
#include "mlx/compile.h"
#include "mlx/graph_utils.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"
#include "python/src/trees.h"

namespace py = pybind11;
using namespace py::literals;
using namespace mlx::core;

using IntOrVec = std::variant<int, std::vector<int>>;
using StrOrVec = std::variant<std::string, std::vector<std::string>>;

template <typename T>
std::vector<T> to_vector(const std::variant<T, std::vector<T>>& v) {
  std::vector<T> vals;
  if (auto pv = std::get_if<T>(&v); pv) {
    vals.push_back(*pv);
  } else {
    vals = std::get<std::vector<T>>(v);
  }
  return vals;
}

auto validate_argnums_argnames(
    const std::optional<IntOrVec>& argnums,
    const StrOrVec& argnames) {
  auto vec_names = to_vector(argnames);

  if (!argnums.has_value()) {
    // argnums was not provided and argnames was empty
    if (vec_names.empty()) {
      return std::make_pair(std::vector<int>{0}, vec_names);
    } else {
      return std::make_pair(std::vector<int>{}, vec_names);
    }
  }

  return std::make_pair(to_vector(*argnums), vec_names);
}

auto py_value_and_grad(
    const py::function& fun,
    std::vector<int> argnums,
    std::vector<std::string> argnames,
    const std::string& error_msg_tag,
    bool scalar_func_only) {
  // Sanitize argnums
  if (argnums.size() == 0 && argnames.size() == 0) {
    throw std::invalid_argument(
        error_msg_tag + " Gradient wrt no argument requested");
  }
  if (argnums.size() > 0) {
    std::sort(argnums.begin(), argnums.end());
    if (argnums[0] < 0) {
      std::ostringstream msg;
      msg << error_msg_tag
          << " Can't compute the gradient of negative argument index "
          << argnums[0];
      throw std::invalid_argument(msg.str());
    }
  }

  return [fun, argnums, argnames, error_msg_tag, scalar_func_only](
             const py::args& args, const py::kwargs& kwargs) {
    // Sanitize the input
    if (argnums.size() > 0 && argnums.back() >= args.size()) {
      std::ostringstream msg;
      msg << error_msg_tag << " Can't compute the gradient of argument index "
          << argnums.back() << " because the function is called with only "
          << args.size() << " arguments.";
      throw std::invalid_argument(msg.str());
    }

    for (auto& key : argnames) {
      if (!kwargs.contains(key)) {
        std::ostringstream msg;
        msg << error_msg_tag
            << " Can't compute the gradient of keyword argument '" << key
            << "' because the function is called with the "
            << "following keyword arguments {";
        for (auto item : kwargs) {
          msg << item.first.cast<std::string>() << ",";
        }
        msg << "}";
        throw std::invalid_argument(msg.str());
      }
    }

    // Collect the arrays
    std::vector<array> arrays;
    std::vector<int> counts(1, 0);
    for (auto i : argnums) {
      auto argsi = tree_flatten(args[i]);
      arrays.insert(arrays.end(), argsi.begin(), argsi.end());
      counts.push_back(argsi.size());
    }
    for (auto& key : argnames) {
      auto argsk = tree_flatten(kwargs[key.c_str()]);
      arrays.insert(arrays.end(), argsk.begin(), argsk.end());
      counts.push_back(argsk.size());
    }
    std::partial_sum(counts.cbegin(), counts.cend(), counts.begin());
    std::vector<int> gradient_indices(arrays.size());
    std::iota(gradient_indices.begin(), gradient_indices.end(), 0);

    // value_out will hold the output of the python function in order to be
    // able to reconstruct the python tree of extra return values
    py::object py_value_out;
    auto value_and_grads = value_and_grad(
        [&fun,
         &args,
         &kwargs,
         &argnums,
         &argnames,
         &counts,
         &py_value_out,
         &error_msg_tag,
         scalar_func_only](const std::vector<array>& a) {
          // Copy the arguments
          py::args args_cpy = py::tuple(args.size());
          py::kwargs kwargs_cpy = py::kwargs();
          int j = 0;
          for (int i = 0; i < args.size(); ++i) {
            if (j < argnums.size() && i == argnums[j]) {
              args_cpy[i] = tree_unflatten(args[i], a, counts[j]);
              j++;
            } else {
              args_cpy[i] = args[i];
            }
          }
          for (auto& key : argnames) {
            kwargs_cpy[key.c_str()] =
                tree_unflatten(kwargs[key.c_str()], a, counts[j]);
            j++;
          }
          for (auto item : kwargs) {
            if (kwargs_cpy.contains(item.first)) {
              continue;
            }
            kwargs_cpy[item.first] = item.second;
          }

          // Call the python function
          py_value_out = fun(*args_cpy, **kwargs_cpy);

          // Validate the return value of the python function
          if (!py::isinstance<array>(py_value_out)) {
            if (scalar_func_only) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be a "
                  << "scalar array; but " << py_value_out.get_type()
                  << " was returned.";
              throw std::invalid_argument(msg.str());
            }
            if (!py::isinstance<py::tuple>(py_value_out)) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be either a "
                  << "scalar array or a tuple with the first value being a "
                  << "scalar array (Union[array, Tuple[array, Any, ...]]); but "
                  << py_value_out.get_type() << " was returned.";
              throw std::invalid_argument(msg.str());
            }
            py::tuple ret = py::cast<py::tuple>(py_value_out);
            if (ret.size() == 0) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be either a "
                  << "scalar array or a non-empty tuple. The first value should be a "
                  << "scalar array and the rest can be anything. Instead, "
                  << "we got an empty tuple.";
              throw std::invalid_argument(msg.str());
            }
            if (!py::isinstance<array>(ret[0])) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be either a "
                  << "scalar array or a tuple with the first value being a "
                  << "scalar array (Union[array, Tuple[array, Any, ...]]); but it "
                  << "was a tuple with the first value being of type "
                  << ret[0].get_type() << " .";
              throw std::invalid_argument(msg.str());
            }
          }

          return tree_flatten(py_value_out, false);
        },
        gradient_indices)(arrays);

    auto value = value_and_grads.first;
    auto gradients = value_and_grads.second;

    // Put the gradients back in their container.
    // We have the following cases:
    //
    // 1. Single python positional argument has a gradient (eg argnums=[0])
    // 2. Many python positional arguments have gradients (eg argnums=[0, 1])
    // 3. A python keyword argument has gradients
    //
    // In case 1 we return the original python variable but with the gradients.
    // In case 2 we return a tuple of the above.
    // In case 3 we return a tuple containing a tuple and dict (sth like
    // (tuple(), dict(x=mx.array(5))) ).
    py::object positional_grads;
    py::object keyword_grads;
    py::object py_grads;

    // Collect the gradients for the positional arguments
    if (argnums.size() == 1) {
      positional_grads = tree_unflatten(args[argnums[0]], gradients, counts[0]);
    } else if (argnums.size() > 1) {
      py::tuple grads_(argnums.size());
      for (int i = 0; i < argnums.size(); i++) {
        grads_[i] = tree_unflatten(args[argnums[i]], gradients, counts[i]);
      }
      positional_grads = py::cast<py::object>(grads_);
    } else {
      positional_grads = py::none();
    }

    // No keyword argument gradients so return the tuple of gradients
    if (argnames.size() == 0) {
      py_grads = positional_grads;
    } else {
      py::dict grads_;
      for (int i = 0; i < argnames.size(); i++) {
        auto& k = argnames[i];
        grads_[k.c_str()] = tree_unflatten(
            kwargs[k.c_str()], gradients, counts[i + argnums.size()]);
      }
      keyword_grads = py::cast<py::object>(grads_);

      py_grads =
          py::cast<py::object>(py::make_tuple(positional_grads, keyword_grads));
    }

    // Put the values back in the container
    py::object return_value = tree_unflatten(py_value_out, value);
    return std::make_pair(return_value, py_grads);
  };
}

auto py_vmap(
    const py::function& fun,
    const py::object& in_axes,
    const py::object& out_axes) {
  return [fun, in_axes, out_axes](const py::args& args) {
    auto axes_to_flat_tree = [](const py::object& tree,
                                const py::object& axes) {
      auto tree_axes = tree_map(
          {tree, axes},
          [](const std::vector<py::object>& inputs) { return inputs[1]; });
      std::vector<int> flat_axes;
      tree_visit(tree_axes, [&flat_axes](py::handle obj) {
        if (obj.is_none()) {
          flat_axes.push_back(-1);
        } else if (py::isinstance<py::int_>(obj)) {
          flat_axes.push_back(py::cast<int>(py::cast<py::int_>(obj)));
        } else {
          throw std::invalid_argument("[vmap] axis must be int or None.");
        }
      });
      return flat_axes;
    };

    // Inputs must be array or tree of arrays
    auto inputs = tree_flatten(args, true);
    auto flat_in_axes = axes_to_flat_tree(args, in_axes);

    // py_value_out will hold the output of the python function in order to be
    // able to reconstruct the python tree of extra return values
    py::object py_outputs;

    auto vmap_fn =
        [&fun, &args, &inputs, &py_outputs](const std::vector<array>& a) {
          // Call the python function
          py_outputs = fun(*tree_unflatten(args, a));

          // Flatten the outputs
          return tree_flatten(py_outputs, true);
        };

    auto [trace_inputs, trace_outputs] =
        detail::vmap_trace(vmap_fn, inputs, flat_in_axes);

    auto flat_out_axes = axes_to_flat_tree(py_outputs, out_axes);

    // Perform the vmap
    auto outputs = detail::vmap_replace(
        inputs, trace_inputs, trace_outputs, flat_in_axes, flat_out_axes);

    // Put the outputs back in the container
    return tree_unflatten(py_outputs, outputs);
  };
}

std::unordered_map<size_t, py::object>& tree_cache() {
  // This map is used to Cache the tree structure of the outputs
  static std::unordered_map<size_t, py::object> tree_cache_;
  return tree_cache_;
}

struct PyCompiledFun {
  py::function fun;
  size_t fun_id;
  py::object captured_inputs;
  py::object captured_outputs;
  bool shapeless;
  size_t num_outputs{0};

  PyCompiledFun(
      const py::function& fun,
      py::object inputs,
      py::object outputs,
      bool shapeless)
      : fun(fun),
        fun_id(reinterpret_cast<size_t>(fun.ptr())),
        captured_inputs(inputs),
        captured_outputs(outputs),
        shapeless(shapeless) {}

  PyCompiledFun(const PyCompiledFun&) = delete;
  PyCompiledFun& operator=(const PyCompiledFun&) = delete;
  PyCompiledFun& operator=(PyCompiledFun&& other) = delete;
  PyCompiledFun(PyCompiledFun&& other)
      : fun(std::move(other.fun)), fun_id(reinterpret_cast<size_t>(fun.ptr())) {
    other.fun_id = 0;
    captured_inputs = std::move(other.captured_inputs);
    captured_outputs = std::move(other.captured_outputs);
    shapeless = other.shapeless;
    num_outputs = other.num_outputs;
  };

  py::object operator()(const py::args& args, const py::kwargs& kwargs) {
    // Flat array inputs
    std::vector<array> inputs;

    // Compilation constants
    std::vector<uint64_t> constants;

    auto flatten_with_constants = [&](py::object tree) {
      tree_visit(tree, [&](py::handle obj) {
        if (py::isinstance<array>(obj)) {
          inputs.push_back(py::cast<array>(obj));
        } else if (py::isinstance<py::str>(obj)) {
          auto r = py::hash(obj);
          constants.push_back(*reinterpret_cast<uint64_t*>(&r));
        } else if (py::isinstance<py::int_>(obj)) {
          auto r = obj.cast<int64_t>();
          constants.push_back(*reinterpret_cast<uint64_t*>(&r));
        } else if (py::isinstance<py::float_>(obj)) {
          auto r = obj.cast<double>();
          constants.push_back(*reinterpret_cast<uint64_t*>(&r));
        } else {
          std::ostringstream msg;
          msg << "[compile] Function arguments must be trees of arrays "
              << "or constants (floats, ints, or strings), but received "
              << "type " << obj.get_type() << ".";
          throw std::invalid_argument(msg.str());
        }
      });
    };
    flatten_with_constants(args);
    int num_args = inputs.size();
    flatten_with_constants(kwargs);

    auto compile_fun = [this, &args, &kwargs, num_args](
                           const std::vector<array>& a) {
      // Put tracers into captured inputs
      std::vector<array> flat_in_captures;
      std::vector<array> trace_captures;
      if (!py::isinstance<py::none>(captured_inputs)) {
        flat_in_captures = tree_flatten(captured_inputs, false);
        trace_captures.insert(
            trace_captures.end(), a.end() - flat_in_captures.size(), a.end());
        tree_fill(captured_inputs, trace_captures);
      }

      auto tree_outputs =
          fun(*tree_unflatten(args, a), **tree_unflatten(kwargs, a, num_args));
      auto [outputs, py_outputs] =
          tree_flatten_with_structure(std::move(tree_outputs), false);

      tree_cache().insert({fun_id, py_outputs});

      num_outputs = outputs.size();
      if (!py::isinstance<py::none>(captured_outputs)) {
        auto flat_out_captures = tree_flatten(captured_outputs, false);
        outputs.insert(
            outputs.end(),
            std::make_move_iterator(flat_out_captures.begin()),
            std::make_move_iterator(flat_out_captures.end()));
      }

      // Replace tracers with originals in captured inputs
      if (!py::isinstance<py::none>(captured_inputs)) {
        tree_replace(captured_inputs, trace_captures, flat_in_captures);
      }
      return outputs;
    };

    if (!py::isinstance<py::none>(captured_inputs)) {
      auto flat_in_captures = tree_flatten(captured_inputs, false);
      inputs.insert(
          inputs.end(),
          std::make_move_iterator(flat_in_captures.begin()),
          std::make_move_iterator(flat_in_captures.end()));
    }

    // Compile and call
    auto outputs =
        detail::compile(compile_fun, fun_id, shapeless, constants)(inputs);
    if (!py::isinstance<py::none>(captured_outputs)) {
      std::vector<array> captures(
          std::make_move_iterator(outputs.begin() + num_outputs),
          std::make_move_iterator(outputs.end()));
      tree_fill(captured_outputs, captures);
    }

    // Put the outputs back in the container
    py::object py_outputs = tree_cache().at(fun_id);
    return tree_unflatten_from_structure(py_outputs, outputs);
  };

  ~PyCompiledFun() {
    py::gil_scoped_acquire gil;

    tree_cache().erase(fun_id);
    detail::compile_erase(fun_id);
    fun.release().dec_ref();
    captured_inputs.release().dec_ref();
    captured_outputs.release().dec_ref();
  }
};

class PyCheckpointedFun {
 public:
  PyCheckpointedFun(py::function fun) : fun_(std::move(fun)) {}

  ~PyCheckpointedFun() {
    py::gil_scoped_acquire gil;

    fun_.release().dec_ref();
  }

  struct InnerFunction {
    py::object fun_;
    py::object args_structure_;
    std::weak_ptr<py::object> output_structure_;

    InnerFunction(
        py::object fun,
        py::object args_structure,
        std::weak_ptr<py::object> output_structure)
        : fun_(std::move(fun)),
          args_structure_(std::move(args_structure)),
          output_structure_(output_structure) {}
    ~InnerFunction() {
      py::gil_scoped_acquire gil;

      fun_.release().dec_ref();
      args_structure_.release().dec_ref();
    }

    std::vector<array> operator()(const std::vector<array>& inputs) {
      auto args = py::cast<py::tuple>(
          tree_unflatten_from_structure(args_structure_, inputs));
      auto [outputs, output_structure] =
          tree_flatten_with_structure(fun_(*args[0], **args[1]), false);
      if (auto s = output_structure_.lock()) {
        *s = output_structure;
      }
      return outputs;
    }
  };

  py::object operator()(const py::args& args, const py::kwargs& kwargs) {
    auto output_structure = std::make_shared<py::object>();
    auto full_args = py::make_tuple(args, kwargs);
    auto [inputs, args_structure] =
        tree_flatten_with_structure(full_args, false);

    auto outputs = checkpoint(
        InnerFunction(fun_, args_structure, output_structure))(inputs);

    return tree_unflatten_from_structure(*output_structure, outputs);
  }

 private:
  py::function fun_;
};

void init_transforms(py::module_& m) {
  py::options options;
  options.disable_function_signatures();

  m.def(
      "eval",
      [](const py::args& args) {
        std::vector<array> arrays = tree_flatten(args, false);
        {
          py::gil_scoped_release nogil;
          eval(arrays);
        }
      },
      R"pbdoc(
        eval(*args) -> None

        Evaluate an :class:`array` or tree of :class:`array`.

        Args:
            *args (arrays or trees of arrays): Each argument can be a single array
              or a tree of arrays. If a tree is given the nodes can be a Python
              :class:`list`, :class:`tuple` or :class:`dict`. Leaves which are not
              arrays are ignored.
      )pbdoc");
  m.def(
      "jvp",
      [](const py::function& fun,
         const std::vector<array>& primals,
         const std::vector<array>& tangents) {
        auto vfun = [&fun](const std::vector<array>& primals) {
          py::args args = py::tuple(primals.size());
          for (int i = 0; i < primals.size(); ++i) {
            args[i] = primals[i];
          }
          auto out = fun(*args);
          if (py::isinstance<array>(out)) {
            return std::vector<array>{py::cast<array>(out)};
          } else {
            return py::cast<std::vector<array>>(out);
          }
        };
        return jvp(vfun, primals, tangents);
      },
      "fun"_a,
      "primals"_a,
      "tangents"_a,
      R"pbdoc(
        jvp(fun: function, primals: List[array], tangents: List[array]) -> Tuple[List[array], List[array]]


        Compute the Jacobian-vector product.

        This computes the product of the Jacobian of a function ``fun`` evaluated
        at ``primals`` with the ``tangents``.

        Args:
            fun (function): A function which takes a variable number of :class:`array`
              and returns a single :class:`array` or list of :class:`array`.
            primals (list(array)): A list of :class:`array` at which to
              evaluate the Jacobian.
            tangents (list(array)): A list of :class:`array` which are the
              "vector" in the Jacobian-vector product. The ``tangents`` should be the
              same in number, shape, and type as the inputs of ``fun`` (i.e. the ``primals``).

        Returns:
            list(array): A list of the Jacobian-vector products which
            is the same in number, shape, and type of the inputs to ``fun``.
      )pbdoc");
  m.def(
      "vjp",
      [](const py::function& fun,
         const std::vector<array>& primals,
         const std::vector<array>& cotangents) {
        auto vfun = [&fun](const std::vector<array>& primals) {
          py::args args = py::tuple(primals.size());
          for (int i = 0; i < primals.size(); ++i) {
            args[i] = primals[i];
          }
          auto out = fun(*args);
          if (py::isinstance<array>(out)) {
            return std::vector<array>{py::cast<array>(out)};
          } else {
            return py::cast<std::vector<array>>(out);
          }
        };
        return vjp(vfun, primals, cotangents);
      },
      "fun"_a,
      "primals"_a,
      "cotangents"_a,
      R"pbdoc(
        vjp(fun: function, primals: List[array], cotangents: List[array]) -> Tuple[List[array], List[array]]

        Compute the vector-Jacobian product.

        Computes the product of the ``cotangents`` with the Jacobian of a
        function ``fun`` evaluated at ``primals``.

        Args:
          fun (function): A function which takes a variable number of :class:`array`
            and returns a single :class:`array` or list of :class:`array`.
          primals (list(array)): A list of :class:`array` at which to
            evaluate the Jacobian.
          cotangents (list(array)): A list of :class:`array` which are the
            "vector" in the vector-Jacobian product. The ``cotangents`` should be the
            same in number, shape, and type as the outputs of ``fun``.

        Returns:
            list(array): A list of the vector-Jacobian products which
            is the same in number, shape, and type of the outputs of ``fun``.
      )pbdoc");
  m.def(
      "value_and_grad",
      [](const py::function& fun,
         const std::optional<IntOrVec>& argnums,
         const StrOrVec& argnames) {
        auto [argnums_vec, argnames_vec] =
            validate_argnums_argnames(argnums, argnames);
        return py::cpp_function(py_value_and_grad(
            fun, argnums_vec, argnames_vec, "[value_and_grad]", false));
      },
      "fun"_a,
      "argnums"_a = std::nullopt,
      "argnames"_a = std::vector<std::string>{},
      R"pbdoc(
        value_and_grad(fun: function, argnums: Optional[Union[int, List[int]]] = None, argnames: Union[str, List[str]] = []) -> function

        Returns a function which computes the value and gradient of ``fun``.

        The function passed to :func:`value_and_grad` should return either
        a scalar loss or a tuple in which the first element is a scalar
        loss and the remaining elements can be anything.

        .. code-block:: python

            import mlx.core as mx

            def mse(params, inputs, targets):
                outputs = forward(params, inputs)
                lvalue = (outputs - targets).square().mean()
                return lvalue

            # Returns lvalue, dlvalue/dparams
            lvalue, grads = mx.value_and_grad(mse)(params, inputs, targets)

            def lasso(params, inputs, targets, a=1.0, b=1.0):
                outputs = forward(params, inputs)
                mse = (outputs - targets).square().mean()
                l1 = mx.abs(outputs - targets).mean()

                loss = a*mse + b*l1

                return loss, mse, l1

            (loss, mse, l1), grads = mx.value_and_grad(lasso)(params, inputs, targets)

        Args:
            fun (function): A function which takes a variable number of
              :class:`array` or trees of :class:`array` and returns
              a scalar output :class:`array` or a tuple the first element
              of which should be a scalar :class:`array`.
            argnums (int or list(int), optional): Specify the index (or indices)
              of the positional arguments of ``fun`` to compute the gradient
              with respect to. If neither ``argnums`` nor ``argnames`` are
              provided ``argnums`` defaults to ``0`` indicating ``fun``'s first
              argument.
            argnames (str or list(str), optional): Specify keyword arguments of
              ``fun`` to compute gradients with respect to. It defaults to [] so
              no gradients for keyword arguments by default.

        Returns:
            function: A function which returns a tuple where the first element
            is the output of `fun` and the second element is the gradients w.r.t.
            the loss.
      )pbdoc");
  m.def(
      "grad",
      [](const py::function& fun,
         const std::optional<IntOrVec>& argnums,
         const StrOrVec& argnames) {
        auto [argnums_vec, argnames_vec] =
            validate_argnums_argnames(argnums, argnames);
        auto fn =
            py_value_and_grad(fun, argnums_vec, argnames_vec, "[grad]", true);
        return py::cpp_function(
            [fn](const py::args& args, const py::kwargs& kwargs) {
              return fn(args, kwargs).second;
            });
      },
      "fun"_a,
      "argnums"_a = std::nullopt,
      "argnames"_a = std::vector<std::string>{},
      R"pbdoc(
        grad(fun: function, argnums: Optional[Union[int, List[int]]] = None, argnames: Union[str, List[str]] = []) -> function

        Returns a function which computes the gradient of ``fun``.

        Args:
            fun (function): A function which takes a variable number of
              :class:`array` or trees of :class:`array` and returns
              a scalar output :class:`array`.
            argnums (int or list(int), optional): Specify the index (or indices)
              of the positional arguments of ``fun`` to compute the gradient
              with respect to. If neither ``argnums`` nor ``argnames`` are
              provided ``argnums`` defaults to ``0`` indicating ``fun``'s first
              argument.
            argnames (str or list(str), optional): Specify keyword arguments of
              ``fun`` to compute gradients with respect to. It defaults to [] so
              no gradients for keyword arguments by default.

        Returns:
            function: A function which has the same input arguments as ``fun`` and
            returns the gradient(s).
      )pbdoc");
  m.def(
      "vmap",
      [](const py::function& fun,
         const py::object& in_axes,
         const py::object& out_axes) {
        return py::cpp_function(py_vmap(fun, in_axes, out_axes));
      },
      "fun"_a,
      "in_axes"_a = 0,
      "out_axes"_a = 0,
      R"pbdoc(
        vmap(fun: function, in_axes: object = 0, out_axes: object = 0) -> function

        Returns a vectorized version of ``fun``.

        Args:
            fun (function): A function which takes a variable number of
              :class:`array` or a tree of :class:`array` and returns
              a variable number of :class:`array` or a tree of :class:`array`.
            in_axes (int, optional): An integer or a valid prefix tree of the
              inputs to ``fun`` where each node specifies the vmapped axis. If
              the value is ``None`` then the corresponding input(s) are not vmapped.
              Defaults to ``0``.
            out_axes (int, optional): An integer or a valid prefix tree of the
              outputs of ``fun`` where each node specifies the vmapped axis. If
              the value is ``None`` then the corresponding outputs(s) are not vmapped.
              Defaults to ``0``.

        Returns:
            function: The vectorized function.
      )pbdoc");
  m.def(
      "export_to_dot",
      [](py::object file, const py::args& args) {
        std::vector<array> arrays = tree_flatten(args);
        if (py::isinstance<py::str>(file)) {
          std::ofstream out(py::cast<std::string>(file));
          export_to_dot(out, arrays);
        } else if (py::hasattr(file, "write")) {
          std::ostringstream out;
          export_to_dot(out, arrays);
          auto write = file.attr("write");
          write(out.str());
        } else {
          throw std::invalid_argument(
              "export_to_dot accepts file-like objects or strings to be used as filenames");
        }
      },
      "file"_a);
  m.def(
      "compile",
      [](const py::function& fun,
         const py::object& inputs,
         const py::object& outputs,
         bool shapeless) {
        return py::cpp_function(PyCompiledFun{fun, inputs, outputs, shapeless});
      },
      "fun"_a,
      "inputs"_a = std::nullopt,
      "outputs"_a = std::nullopt,
      "shapeless"_a = false,
      R"pbdoc(
        compile(fun: function) -> function

        Returns a compiled function which produces the same output as ``fun``.

        Args:
            fun (function): A function which takes a variable number of
              :class:`array` or trees of :class:`array` and returns
              a variable number of :class:`array` or trees of :class:`array`.
            inputs (list or dict, optional): These inputs will be captured during
              the function compilation along with the inputs to ``fun``. The ``inputs``
              can be a :obj:`list` or a :obj:`dict` containing arbitrarily nested
              lists, dictionaries, or arrays. Leaf nodes that are not
              :obj:`array` are ignored. Default: ``None``
            outputs (list or dict, optional): These outputs will be captured and
              updated in a compiled function. The ``outputs`` can be a
              :obj:`list` or a :obj:`dict` containing arbitrarily nested lists,
              dictionaries, or arrays. Leaf nodes that are not :obj:`array` are ignored.
              Default: ``None``
            shapeless (bool, optional): A function compiled with the ``shapeless``
              option enabled will not be recompiled when the input shape changes. Not all
              functions can be compiled with ``shapeless`` enabled. Attempting to compile
              such functions with shapeless enabled will throw. Note, changing the number
              of dimensions or type of any input will result in a recompilation even with
              ``shapeless`` set to ``True``. Default: ``False``

        Returns:
            function: A compiled function which has the same input arguments
            as ``fun`` and returns the the same output(s).
      )pbdoc");
  m.def(
      "disable_compile",
      &disable_compile,
      R"pbdoc(
        disable_compile() -> None

        Globally disable compilation. Setting the environment variable
        ``MLX_DISABLE_COMPILE`` can also be used to disable compilation.
      )pbdoc");
  m.def(
      "enable_compile",
      &enable_compile,
      R"pbdoc(
        enable_compile() -> None

        Globally enable compilation. This will override the environment
        variable ``MLX_DISABLE_COMPILE`` if set.
      )pbdoc");
  m.def(
      "checkpoint",
      [](py::function fun) { return py::cpp_function(PyCheckpointedFun{fun}); },
      "fun"_a);

  // Register static Python object cleanup before the interpreter exits
  auto atexit = py::module_::import("atexit");
  atexit.attr("register")(py::cpp_function([]() { tree_cache().clear(); }));
}
