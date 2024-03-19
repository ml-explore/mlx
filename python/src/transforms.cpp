// Copyright © 2023-2024 Apple Inc.
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

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

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

using IntOrVec = std::variant<int, std::vector<int>>;
using StrOrVec = std::variant<std::string, std::vector<std::string>>;

inline std::string type_name_str(const nb::handle& o) {
  return nb::cast<std::string>(nb::type_name(o.type()));
}

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
    const nb::callable& fun,
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
             const nb::args& args, const nb::kwargs& kwargs) {
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
          msg << nb::cast<std::string>(item.first) << ",";
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
    nb::object py_value_out;
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
          nb::list args_cpy;
          nb::kwargs kwargs_cpy = nb::kwargs();
          int j = 0;
          for (int i = 0; i < args.size(); ++i) {
            if (j < argnums.size() && i == argnums[j]) {
              args_cpy.append(tree_unflatten(args[i], a, counts[j]));
              j++;
            } else {
              args_cpy.append(args[i]);
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
          if (!nb::isinstance<array>(py_value_out)) {
            if (scalar_func_only) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be a "
                  << "scalar array; but " << type_name_str(py_value_out)
                  << " was returned.";
              throw std::invalid_argument(msg.str());
            }
            if (!nb::isinstance<nb::tuple>(py_value_out)) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be either a "
                  << "scalar array or a tuple with the first value being a "
                  << "scalar array (Union[array, Tuple[array, Any, ...]]); but "
                  << type_name_str(py_value_out) << " was returned.";
              throw std::invalid_argument(msg.str());
            }
            nb::tuple ret = nb::cast<nb::tuple>(py_value_out);
            if (ret.size() == 0) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be either a "
                  << "scalar array or a non-empty tuple. The first value should be a "
                  << "scalar array and the rest can be anything. Instead, "
                  << "we got an empty tuple.";
              throw std::invalid_argument(msg.str());
            }
            if (!nb::isinstance<array>(ret[0])) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be either a "
                  << "scalar array or a tuple with the first value being a "
                  << "scalar array (Union[array, Tuple[array, Any, ...]]); but it "
                  << "was a tuple with the first value being of type "
                  << type_name_str(ret[0]) << " .";
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
    nb::object positional_grads;
    nb::object keyword_grads;
    nb::object py_grads;

    // Collect the gradients for the positional arguments
    if (argnums.size() == 1) {
      positional_grads = tree_unflatten(args[argnums[0]], gradients, counts[0]);
    } else if (argnums.size() > 1) {
      nb::list grads_;
      for (int i = 0; i < argnums.size(); i++) {
        grads_.append(tree_unflatten(args[argnums[i]], gradients, counts[i]));
      }
      positional_grads = nb::tuple(grads_);
    } else {
      positional_grads = nb::none();
    }

    // No keyword argument gradients so return the tuple of gradients
    if (argnames.size() == 0) {
      py_grads = positional_grads;
    } else {
      nb::dict grads_;
      for (int i = 0; i < argnames.size(); i++) {
        auto& k = argnames[i];
        grads_[k.c_str()] = tree_unflatten(
            kwargs[k.c_str()], gradients, counts[i + argnums.size()]);
      }
      keyword_grads = grads_;

      py_grads = nb::make_tuple(positional_grads, keyword_grads);
    }

    // Put the values back in the container
    nb::object return_value = tree_unflatten(py_value_out, value);
    return std::make_pair(return_value, py_grads);
  };
}

auto py_vmap(
    const nb::callable& fun,
    const nb::object& in_axes,
    const nb::object& out_axes) {
  return [fun, in_axes, out_axes](const nb::args& args) {
    auto axes_to_flat_tree = [](const nb::object& tree,
                                const nb::object& axes) {
      auto tree_axes = tree_map(
          {tree, axes},
          [](const std::vector<nb::object>& inputs) { return inputs[1]; });
      std::vector<int> flat_axes;
      tree_visit(tree_axes, [&flat_axes](nb::handle obj) {
        if (obj.is_none()) {
          flat_axes.push_back(-1);
        } else if (nb::isinstance<nb::int_>(obj)) {
          flat_axes.push_back(nb::cast<int>(nb::cast<nb::int_>(obj)));
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
    nb::object py_outputs;

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

std::unordered_map<size_t, nb::object>& tree_cache() {
  // This map is used to Cache the tree structure of the outputs
  static std::unordered_map<size_t, nb::object> tree_cache_;
  return tree_cache_;
}

struct PyCompiledFun {
  nb::callable fun;
  size_t fun_id;
  nb::object captured_inputs;
  nb::object captured_outputs;
  bool shapeless;
  mutable size_t num_outputs{0};

  PyCompiledFun(
      const nb::callable& fun,
      nb::object inputs,
      nb::object outputs,
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

  nb::object call_impl(const nb::args& args, const nb::kwargs& kwargs) {
    // Flat array inputs
    std::vector<array> inputs;

    // Compilation constants which includes the tree structure of the arguments
    std::vector<uint64_t> constants;

    // Reserve some large primes to signify the presence of an array, a list or
    // a dict in order to encode the structure of the pytree. We choose primes
    // to reduce slightly the chances of these numbers occurring by a
    // multiplication as values in the constants list.
    constexpr uint64_t array_identifier = 18446744073709551557UL;
    constexpr uint64_t list_identifier = 18446744073709551533UL;
    constexpr uint64_t dict_identifier = 18446744073709551521UL;

    // Flatten the tree with hashed constants and structure
    std::function<void(nb::handle)> recurse;
    recurse = [&](nb::handle obj) {
      if (nb::isinstance<nb::list>(obj)) {
        auto l = nb::cast<nb::list>(obj);
        constants.push_back(list_identifier);
        for (int i = 0; i < l.size(); ++i) {
          recurse(l[i]);
        }
      } else if (nb::isinstance<nb::tuple>(obj)) {
        auto l = nb::cast<nb::tuple>(obj);
        constants.push_back(list_identifier);
        for (auto item : obj) {
          recurse(item);
        }
      } else if (nb::isinstance<nb::dict>(obj)) {
        auto d = nb::cast<nb::dict>(obj);
        constants.push_back(dict_identifier);
        for (auto item : d) {
          auto r = item.first.attr("__hash__");
          constants.push_back(*reinterpret_cast<uint64_t*>(&r));
          recurse(item.second);
        }
      } else if (nb::isinstance<array>(obj)) {
        inputs.push_back(nb::cast<array>(obj));
        constants.push_back(array_identifier);
      } else if (nb::isinstance<nb::str>(obj)) {
        auto r = obj.attr("__hash__");
        constants.push_back(*reinterpret_cast<uint64_t*>(&r));
      } else if (nb::isinstance<nb::int_>(obj)) {
        auto r = nb::cast<int64_t>(obj);
        constants.push_back(*reinterpret_cast<uint64_t*>(&r));
      } else if (nb::isinstance<nb::float_>(obj)) {
        auto r = nb::cast<double>(obj);
        constants.push_back(*reinterpret_cast<uint64_t*>(&r));
      } else {
        std::ostringstream msg;
        msg << "[compile] Function arguments must be trees of arrays "
            << "or constants (floats, ints, or strings), but received "
            << "type " << type_name_str(obj) << ".";
        throw std::invalid_argument(msg.str());
      }
    };

    recurse(args);
    int num_args = inputs.size();
    recurse(kwargs);
    auto compile_fun = [this, &args, &kwargs, num_args](
                           const std::vector<array>& a) {
      // Put tracers into captured inputs
      std::vector<array> flat_in_captures;
      std::vector<array> trace_captures;
      if (!captured_inputs.is_none()) {
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
      if (!captured_outputs.is_none()) {
        auto flat_out_captures = tree_flatten(captured_outputs, false);
        outputs.insert(
            outputs.end(),
            std::make_move_iterator(flat_out_captures.begin()),
            std::make_move_iterator(flat_out_captures.end()));
      }

      // Replace tracers with originals in captured inputs
      if (!captured_inputs.is_none()) {
        tree_replace(captured_inputs, trace_captures, flat_in_captures);
      }
      return outputs;
    };

    if (!captured_inputs.is_none()) {
      auto flat_in_captures = tree_flatten(captured_inputs, false);
      inputs.insert(
          inputs.end(),
          std::make_move_iterator(flat_in_captures.begin()),
          std::make_move_iterator(flat_in_captures.end()));
    }

    // Compile and call
    auto outputs =
        detail::compile(compile_fun, fun_id, shapeless, constants)(inputs);
    if (!captured_outputs.is_none()) {
      std::vector<array> captures(
          std::make_move_iterator(outputs.begin() + num_outputs),
          std::make_move_iterator(outputs.end()));
      tree_fill(captured_outputs, captures);
    }

    // Put the outputs back in the container
    nb::object py_outputs = tree_cache().at(fun_id);
    return tree_unflatten_from_structure(py_outputs, outputs);
  }

  nb::object operator()(const nb::args& args, const nb::kwargs& kwargs) const {
    return const_cast<PyCompiledFun*>(this)->call_impl(args, kwargs);
  };

  ~PyCompiledFun() {
    nb::gil_scoped_acquire gil;

    tree_cache().erase(fun_id);
    detail::compile_erase(fun_id);
    fun.release().dec_ref();
    captured_inputs.release().dec_ref();
    captured_outputs.release().dec_ref();
  }
};

class PyCheckpointedFun {
 public:
  PyCheckpointedFun(nb::callable fun) : fun_(std::move(fun)) {}

  ~PyCheckpointedFun() {
    nb::gil_scoped_acquire gil;

    fun_.release().dec_ref();
  }

  struct InnerFunction {
    nb::object fun_;
    nb::object args_structure_;
    std::weak_ptr<nb::object> output_structure_;

    InnerFunction(
        nb::object fun,
        nb::object args_structure,
        std::weak_ptr<nb::object> output_structure)
        : fun_(std::move(fun)),
          args_structure_(std::move(args_structure)),
          output_structure_(output_structure) {}
    ~InnerFunction() {
      nb::gil_scoped_acquire gil;

      fun_.release().dec_ref();
      args_structure_.release().dec_ref();
    }

    std::vector<array> operator()(const std::vector<array>& inputs) {
      auto args = nb::cast<nb::tuple>(
          tree_unflatten_from_structure(args_structure_, inputs));
      auto [outputs, output_structure] =
          tree_flatten_with_structure(fun_(*args[0], **args[1]), false);
      if (auto s = output_structure_.lock()) {
        *s = output_structure;
      }
      return outputs;
    }
  };

  nb::object call_impl(const nb::args& args, const nb::kwargs& kwargs) {
    auto output_structure = std::make_shared<nb::object>();
    auto full_args = nb::make_tuple(args, kwargs);
    auto [inputs, args_structure] =
        tree_flatten_with_structure(full_args, false);

    auto outputs = checkpoint(
        InnerFunction(fun_, args_structure, output_structure))(inputs);

    return tree_unflatten_from_structure(*output_structure, outputs);
  }

  nb::object operator()(const nb::args& args, const nb::kwargs& kwargs) const {
    return const_cast<PyCheckpointedFun*>(this)->call_impl(args, kwargs);
  }

 private:
  nb::callable fun_;
};

void init_transforms(nb::module_& m) {
  m.def(
      "eval",
      [](const nb::args& args) {
        std::vector<array> arrays = tree_flatten(args, false);
        {
          nb::gil_scoped_release nogil;
          eval(arrays);
        }
      },
      nb::arg(),
      nb::sig("def eval(*args) -> None"),
      R"pbdoc(
        Evaluate an :class:`array` or tree of :class:`array`.

        Args:
            *args (arrays or trees of arrays): Each argument can be a single array
              or a tree of arrays. If a tree is given the nodes can be a Python
              :class:`list`, :class:`tuple` or :class:`dict`. Leaves which are not
              arrays are ignored.
      )pbdoc");
  m.def(
      "jvp",
      [](const nb::callable& fun,
         const std::vector<array>& primals,
         const std::vector<array>& tangents) {
        auto vfun = [&fun](const std::vector<array>& primals) {
          auto out = fun(*nb::cast(primals));
          if (nb::isinstance<array>(out)) {
            return std::vector<array>{nb::cast<array>(out)};
          } else {
            return nb::cast<std::vector<array>>(out);
          }
        };
        return jvp(vfun, primals, tangents);
      },
      "fun"_a,
      "primals"_a,
      "tangents"_a,
      nb::sig(
          "def jvp(fun: callable, primals: List[array], tangents: List[array]) -> Tuple[List[array], List[array]]"),
      R"pbdoc(
        Compute the Jacobian-vector product.

        This computes the product of the Jacobian of a function ``fun`` evaluated
        at ``primals`` with the ``tangents``.

        Args:
            fun (callable): A function which takes a variable number of :class:`array`
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
      [](const nb::callable& fun,
         const std::vector<array>& primals,
         const std::vector<array>& cotangents) {
        auto vfun = [&fun](const std::vector<array>& primals) {
          auto out = fun(*nb::cast(primals));
          if (nb::isinstance<array>(out)) {
            return std::vector<array>{nb::cast<array>(out)};
          } else {
            return nb::cast<std::vector<array>>(out);
          }
        };
        return vjp(vfun, primals, cotangents);
      },
      "fun"_a,
      "primals"_a,
      "cotangents"_a,
      nb::sig(
          "def vjp(fun: callable, primals: List[array], cotangents: List[array]) -> Tuple[List[array], List[array]]"),
      R"pbdoc(
        Compute the vector-Jacobian product.

        Computes the product of the ``cotangents`` with the Jacobian of a
        function ``fun`` evaluated at ``primals``.

        Args:
          fun (callable): A function which takes a variable number of :class:`array`
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
      [](const nb::callable& fun,
         const std::optional<IntOrVec>& argnums,
         const StrOrVec& argnames) {
        auto [argnums_vec, argnames_vec] =
            validate_argnums_argnames(argnums, argnames);
        return nb::cpp_function(py_value_and_grad(
            fun, argnums_vec, argnames_vec, "[value_and_grad]", false));
      },
      "fun"_a,
      "argnums"_a = nb::none(),
      "argnames"_a = std::vector<std::string>{},
      nb::sig(
          "def value_and_grad(fun: callable, argnums: Optional[Union[int, List[int]]] = None, argnames: Union[str, List[str]] = []) -> callable"),
      R"pbdoc(
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
            fun (callable): A function which takes a variable number of
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
            callable: A function which returns a tuple where the first element
            is the output of `fun` and the second element is the gradients w.r.t.
            the loss.
      )pbdoc");
  m.def(
      "grad",
      [](const nb::callable& fun,
         const std::optional<IntOrVec>& argnums,
         const StrOrVec& argnames) {
        auto [argnums_vec, argnames_vec] =
            validate_argnums_argnames(argnums, argnames);
        auto fn =
            py_value_and_grad(fun, argnums_vec, argnames_vec, "[grad]", true);
        return nb::cpp_function(
            [fn](const nb::args& args, const nb::kwargs& kwargs) {
              return fn(args, kwargs).second;
            });
      },
      "fun"_a,
      "argnums"_a = nb::none(),
      "argnames"_a = std::vector<std::string>{},
      nb::sig(
          "def grad(fun: callable, argnums: Optional[Union[int, List[int]]] = None, argnames: Union[str, List[str]] = []) -> callable"),
      R"pbdoc(
        Returns a function which computes the gradient of ``fun``.

        Args:
            fun (callable): A function which takes a variable number of
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
            callable: A function which has the same input arguments as ``fun`` and
            returns the gradient(s).
      )pbdoc");
  m.def(
      "vmap",
      [](const nb::callable& fun,
         const nb::object& in_axes,
         const nb::object& out_axes) {
        return nb::cpp_function(py_vmap(fun, in_axes, out_axes));
      },
      "fun"_a,
      "in_axes"_a = 0,
      "out_axes"_a = 0,
      nb::sig(
          "def vmap(fun: callable, in_axes: object = 0, out_axes: object = 0) -> callable"),
      R"pbdoc(
        Returns a vectorized version of ``fun``.

        Args:
            fun (callable): A function which takes a variable number of
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
            callable: The vectorized function.
      )pbdoc");
  m.def(
      "export_to_dot",
      [](nb::object file, const nb::args& args) {
        std::vector<array> arrays = tree_flatten(args);
        if (nb::isinstance<nb::str>(file)) {
          std::ofstream out(nb::cast<std::string>(file));
          export_to_dot(out, arrays);
        } else if (nb::hasattr(file, "write")) {
          std::ostringstream out;
          export_to_dot(out, arrays);
          auto write = file.attr("write");
          write(out.str());
        } else {
          throw std::invalid_argument(
              "export_to_dot accepts file-like objects or strings to be used as filenames");
        }
      },
      "file"_a,
      "args"_a);
  m.def(
      "compile",
      [](const nb::callable& fun,
         const nb::object& inputs,
         const nb::object& outputs,
         bool shapeless) {
        //  Try to get the name
        auto n = fun.attr("__name__");
        auto name = n.is_none() ? "compiled" : nb::cast<std::string>(n);

        // Try to get the signature
        std::ostringstream sig;
        sig << "def " << name;
        auto inspect = nb::module_::import_("inspect");
        if (nb::cast<bool>(inspect.attr("isroutine")(fun))) {
          sig << nb::cast<std::string>(
              inspect.attr("signature")(fun).attr("__str__")());
        } else {
          sig << "(*args, **kwargs)";
        }

        // Try to get the doc string
        auto d = inspect.attr("getdoc")(fun);
        std::string doc =
            d.is_none() ? "MLX compiled function." : nb::cast<std::string>(d);

        auto sig_str = sig.str();
        return nb::cpp_function(
            PyCompiledFun{fun, inputs, outputs, shapeless},
            nb::name(name.c_str()),
            nb::sig(sig_str.c_str()),
            doc.c_str());
      },
      "fun"_a,
      "inputs"_a = nb::none(),
      "outputs"_a = nb::none(),
      "shapeless"_a = false,
      R"pbdoc(
        Returns a compiled function which produces the same output as ``fun``.

        Args:
            fun (callable): A function which takes a variable number of
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
            callable: A compiled function which has the same input arguments
            as ``fun`` and returns the the same output(s).
      )pbdoc");
  m.def(
      "disable_compile",
      &disable_compile,
      R"pbdoc(
        Globally disable compilation. Setting the environment variable
        ``MLX_DISABLE_COMPILE`` can also be used to disable compilation.
      )pbdoc");
  m.def(
      "enable_compile",
      &enable_compile,
      R"pbdoc(
        Globally enable compilation. This will override the environment
        variable ``MLX_DISABLE_COMPILE`` if set.
      )pbdoc");
  m.def(
      "checkpoint",
      [](nb::callable fun) { return nb::cpp_function(PyCheckpointedFun{fun}); },
      "fun"_a);

  // Register static Python object cleanup before the interpreter exits
  auto atexit = nb::module_::import_("atexit");
  atexit.attr("register")(nb::cpp_function([]() { tree_cache().clear(); }));
}
