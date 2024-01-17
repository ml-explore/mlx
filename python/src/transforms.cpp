// Copyright © 2023-2024 Apple Inc.
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>

#include "mlx/array.h"
#include "mlx/graph_utils.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"

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

void tree_visit(py::object tree, std::function<void(py::handle)> visitor) {
  std::function<void(py::handle)> recurse;
  recurse = [&](py::handle subtree) {
    if (py::isinstance<py::list>(subtree) ||
        py::isinstance<py::tuple>(subtree)) {
      for (auto item : subtree) {
        recurse(item);
      }
    } else if (py::isinstance<py::dict>(subtree)) {
      for (auto item : py::cast<py::dict>(subtree)) {
        recurse(item.second);
      }
    } else {
      visitor(subtree);
    }
  };

  recurse(tree);
}

template <typename T, typename U, typename V>
void validate_subtrees(const std::vector<py::object>& subtrees) {
  int len = py::cast<T>(subtrees[0]).size();
  for (auto& subtree : subtrees) {
    if ((py::isinstance<T>(subtree) && py::cast<T>(subtree).size() != len) ||
        py::isinstance<U>(subtree) || py::isinstance<V>(subtree)) {
      throw std::invalid_argument(
          "[tree_map] Additional input tree is not a valid prefix of the first tree.");
    }
  }
}

py::object tree_map(
    const std::vector<py::object>& trees,
    std::function<py::object(const std::vector<py::object>&)> transform) {
  std::function<py::object(const std::vector<py::object>&)> recurse;

  recurse = [&](const std::vector<py::object>& subtrees) {
    if (py::isinstance<py::list>(subtrees[0])) {
      py::list l;
      std::vector<py::object> items(subtrees.size());
      validate_subtrees<py::list, py::tuple, py::dict>(subtrees);
      for (int i = 0; i < py::cast<py::list>(subtrees[0]).size(); ++i) {
        for (int j = 0; j < subtrees.size(); ++j) {
          if (py::isinstance<py::list>(subtrees[j])) {
            items[j] = py::cast<py::list>(subtrees[j])[i];
          } else {
            items[j] = subtrees[j];
          }
        }
        l.append(recurse(items));
      }
      return py::cast<py::object>(l);
    } else if (py::isinstance<py::tuple>(subtrees[0])) {
      //  Check the rest of the subtrees
      std::vector<py::object> items(subtrees.size());
      int len = py::cast<py::tuple>(subtrees[0]).size();
      py::tuple l(len);
      validate_subtrees<py::tuple, py::list, py::dict>(subtrees);
      for (int i = 0; i < len; ++i) {
        for (int j = 0; j < subtrees.size(); ++j) {
          if (py::isinstance<py::tuple>(subtrees[j])) {
            items[j] = py::cast<py::tuple>(subtrees[j])[i];
          } else {
            items[j] = subtrees[j];
          }
        }
        l[i] = recurse(items);
      }
      return py::cast<py::object>(l);
    } else if (py::isinstance<py::dict>(subtrees[0])) {
      std::vector<py::object> items(subtrees.size());
      validate_subtrees<py::dict, py::list, py::tuple>(subtrees);
      py::dict d;
      for (auto item : py::cast<py::dict>(subtrees[0])) {
        for (int j = 0; j < subtrees.size(); ++j) {
          if (py::isinstance<py::dict>(subtrees[j])) {
            auto subdict = py::cast<py::dict>(subtrees[j]);
            if (!subdict.contains(item.first)) {
              throw std::invalid_argument(
                  "[tree_map] Tree is not a valid prefix tree of the first tree.");
            }
            items[j] = subdict[item.first];
          } else {
            items[j] = subtrees[j];
          }
        }
        d[item.first] = recurse(items);
      }
      return py::cast<py::object>(d);
    } else {
      return transform(subtrees);
    }
  };
  return recurse(trees);
}

py::object tree_map(
    py::object tree,
    std::function<py::object(py::handle)> transform) {
  return tree_map({tree}, [&](std::vector<py::object> inputs) {
    return transform(inputs[0]);
  });
}

std::vector<array> tree_flatten(py::object tree, bool strict = true) {
  std::vector<array> flat_tree;

  tree_visit(tree, [&](py::handle obj) {
    if (py::isinstance<array>(obj)) {
      flat_tree.push_back(py::cast<array>(obj));
    } else if (strict) {
      throw std::invalid_argument("Argument is not an array");
    }
  });

  return flat_tree;
}

py::object tree_unflatten(
    py::object tree,
    const std::vector<array>& values,
    int index = 0) {
  return tree_map(tree, [&](py::handle obj) {
    if (py::isinstance<array>(obj)) {
      return py::cast(values[index++]);
    } else {
      return py::cast<py::object>(obj);
    }
  });
}

py::object tree_unflatten_none(
    py::object tree,
    const std::vector<array>& values,
    int index = 0) {
  return tree_map(tree, [&](py::handle obj) {
    if (py::isinstance<py::none>(obj)) {
      return py::cast(values[index++]);
    } else {
      return py::cast<py::object>(obj);
    }
  });
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

  PyCompiledFun(const py::function& fun)
      : fun(fun), fun_id(reinterpret_cast<size_t>(fun.ptr())) {}

  PyCompiledFun(const PyCompiledFun&) = delete;
  PyCompiledFun& operator=(const PyCompiledFun&) = delete;
  PyCompiledFun& operator=(PyCompiledFun&& other) = delete;
  PyCompiledFun(PyCompiledFun&& other) {
    fun = other.fun;
    other.fun_id = 0;
    fun_id = reinterpret_cast<size_t>(fun.ptr());
  };

  py::object operator()(const py::args& args) {
    // TODO, awni, I think this cast is ok??

    auto compile_fun = [this, &args](const std::vector<array>& a) {
      // Call the python function
      py::object py_outputs = this->fun(*tree_unflatten(args, a));

      // Flatten the outputs
      auto outputs = tree_flatten(py_outputs, true);

      py_outputs =
          tree_map(py_outputs, [](const py::handle& x) { return py::none(); });
      tree_cache().insert({this->fun_id, py_outputs});
      return outputs;
    };

    // Inputs must be array or tree of arrays
    auto inputs = tree_flatten(args, true);

    // Get globally enclosed arrays so we don't compile through them
    // c.f. https://github.com/python/cpython/blob/main/Lib/inspect.py#L1638
    if (py::hasattr(fun, "__globals__")) {
      py::dict globals = py::getattr(fun, "__globals__");
      auto co_names = py::getattr(py::getattr(fun, "__code__"), "co_names");
      for (auto& n : co_names) {
        if (py::cast<bool>(globals.attr("__contains__")(n))) {
          auto global_inputs =
              tree_flatten(globals.attr("__getitem__")(n), false);
          std::move(
              std::begin(global_inputs),
              std::end(global_inputs),
              std::back_inserter(inputs));
        }
      }
    }

    // Get locally enclosed arrays so we don't compile through them
    if (py::hasattr(fun, "__closure__")) {
      auto closures = py::getattr(fun, "__closure__");
      if (py::isinstance<py::tuple>(closures)) {
        for (auto& closure : closures) {
          auto enclosed_inputs =
              tree_flatten(py::getattr(closure, "cell_contents"), false);
          std::move(
              std::begin(enclosed_inputs),
              std::end(enclosed_inputs),
              std::back_inserter(inputs));
        }
      }
    }

    // Compile and call
    auto outputs = detail::compile(compile_fun, fun_id)(inputs);

    // Put the outputs back in the container
    py::object py_outputs = tree_cache().at(fun_id);
    return tree_unflatten_none(py_outputs, outputs);
  };

  ~PyCompiledFun() {
    detail::compile_erase(fun_id);
  }
};

void init_transforms(py::module_& m) {
  py::options options;
  options.disable_function_signatures();

  m.def(
      "eval",
      [](const py::args& args) {
        std::vector<array> arrays = tree_flatten(args);
        eval(arrays);
      },
      R"pbdoc(
        eval(*args) -> None

        Evaluate an :class:`array` or tree of :class:`array`.

        Args:
            *args (arrays or trees of arrays): Each argument can be a single array
              or a tree of arrays. If a tree is given the nodes can be a Python
              :class:`list`, :class:`tuple` or :class:`dict` but the leafs must all be
              an :class:`array`.
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
      [](const py::function& fun) {
        return py::cpp_function(PyCompiledFun{fun});
      },
      "fun"_a,
      R"pbdoc(
        compile(fun: function) -> function

        Returns a compiled function which produces the same output as ``fun``.

        Args:
            fun (function): A function which takes a variable number of
              :class:`array` or trees of :class:`array` and returns
              a variable number of :class:`array` or trees of :class:`array`.

        Returns:
            function: A compiled function which has the same input arguments
            as ``fun`` and returns the the same output(s).
      )pbdoc");
  m.def(
      "disable_compiler",
      &disable_compiler,
      R"pbdoc(
        disable_compiler() -> None

        Globally disable compilation. Setting the environment variable
        ``MLX_DISABLE_COMPILER`` can also be used to disable compilation.
      )pbdoc");
  m.def(
      "enable_compiler",
      &enable_compiler,
      R"pbdoc(
        enable_compiler() -> None

        Globally enable compilation. This will override the environment
        variable ``MLX_DISABLE_COMPILER`` if set.
      )pbdoc");

  // Register static Python object cleanup before the interpreter exits
  auto atexit = py::module_::import("atexit");
  atexit.attr("register")(py::cpp_function([]() { tree_cache().clear(); }));
}
