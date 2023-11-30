// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <future>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "mlx/backend/metal/metal.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"
#include "mlx/utils.h"

namespace mlx::core {

void simplify(const std::vector<array>& outputs) {
  std::function<void(const array&)> recurse;
  std::queue<array> tape;
  std::unordered_set<std::uintptr_t> cache;
  std::unordered_map<std::uintptr_t, std::vector<std::pair<array, int>>>
      parents_map;

  // Helpers to identify identical scalars
  std::map<std::pair<uint64_t, Dtype::Val>, array> scalars;
  auto is_scalar = [](const array& a) {
    return a.is_evaled() && a.ndim() == 0;
  };
  auto get_scalar_rep = [](const array& a) {
    uint64_t v = 0;
    int dtype;
    switch (a.dtype().size) {
      case 1:
        v = *a.data<uint8_t>();
        break;
      case 4:
        v = *a.data<uint32_t>();
        break;
      case 8:
        v = *a.data<uint64_t>();
        break;
    }
    return std::make_pair(v, a.dtype().val);
  };

  // DFS the graph to log the parents
  recurse = [&](const array& a) {
    auto id = a.id();
    if (cache.find(id) != cache.end()) {
      return;
    }
    for (int i = 0; i < a.inputs().size(); i++) {
      auto& in = a.inputs()[i];
      parents_map[in.id()].push_back({a, i});
      recurse(in);
    }
    cache.insert(id);
    tape.push(a);
    if (is_scalar(a)) {
      scalars.insert({get_scalar_rep(a), a});
    }
  };
  for (auto& a : outputs) {
    recurse(a);
  }

  // Helper that fuses two arrays in the graph by setting the parents of the
  // source to point to the destination
  auto fuse = [&](array& dst, array& src) {
    auto src_parents = parents_map.find(src.id());
    if (src_parents == parents_map.end()) {
      return;
    }

    auto& pairs = parents_map[dst.id()];
    for (auto& parent : src_parents->second) {
      parent.first.editable_inputs()[parent.second] = dst;
      pairs.push_back(parent);
    }
  };

  // Walk the graph
  cache.clear();

  // Depth-1 array equivalence check.
  auto array_equivalent = [](const array& a, const array& b) {
    if (!a.has_primitive() || !b.has_primitive()) {
      return false;
    }
    const auto& pa = a.primitive();
    const auto& pb = b.primitive();
    if (typeid(pa) != typeid(pb)) {
      return false;
    }

    if (a.inputs().size() != b.inputs().size()) {
      return false;
    }

    for (int i = 0; i < a.inputs().size(); i++) {
      if (a.inputs()[i].id() != b.inputs()[i].id()) {
        return false;
      }
    }

    return pa.is_equivalent(pb);
  };

  while (!tape.empty()) {
    auto arr = std::move(tape.front());
    tape.pop();

    if (cache.find(arr.id()) != cache.end()) {
      continue;
    }

    // Check if we can fuse scalars
    if (is_scalar(arr)) {
      auto scalar = scalars.find(get_scalar_rep(arr));
      if (scalar->second.id() != arr.id()) {
        fuse(scalar->second, arr);
        arr = scalar->second;
      }
    }

    // Check if we can fuse the parents of this array
    auto parents = parents_map.find(arr.id());
    if (parents != parents_map.end()) {
      std::vector<bool> mask(parents->second.size(), false);
      auto N = parents->second.size();
      for (int i = 0; i < N; i++) {
        if (mask[i]) {
          continue;
        }
        for (int j = i + 1; j < N; j++) {
          if (mask[j]) {
            continue;
          }
          auto& src = parents->second[j].first;
          auto& dst = parents->second[i].first;
          if (src.id() != dst.id() && array_equivalent(src, dst)) {
            cache.insert(src.id());
            fuse(dst, src);
            mask[j] = true;
          }
        }
      }
    }
  }
}

void eval(const std::vector<array>& outputs, bool retain_graph /* = false */) {
  if (!retain_graph) {
    for (auto& out : outputs) {
      if (out.has_primitive() && out.is_tracer()) {
        throw std::invalid_argument(
            "[eval] Illegal to eval an array during "
            "function transform without graph retention.");
      }
    }
  }
  std::function<void(const array&)> recurse;
  std::queue<array> tape;
  std::unordered_set<std::uintptr_t> cache;
  std::unordered_map<std::uintptr_t, std::shared_future<void>> deps;

  recurse = [&](const array& a) {
    auto id = a.id();
    if (cache.find(id) != cache.end()) {
      return;
    }
    for (auto in : a.inputs()) {
      recurse(in);
      // If one of the inputs is being computed on a different
      // stream, we need to manage the dependency.
      if (!in.is_evaled()) {
        if (a.primitive().stream() != in.primitive().stream()) {
          deps.insert({in.id(), std::shared_future<void>{}});
        }
      }
    }
    cache.insert(id);
    if (!a.is_evaled() || (!retain_graph && a.has_primitive())) {
      if (!a.has_primitive()) {
        throw std::invalid_argument(
            "[eval] Attempting to eval an array without a primitive.");
      }
      tape.push(a);
    }
  };

  for (auto& arr : outputs) {
    if (!arr.is_evaled() || (!retain_graph && arr.has_primitive())) {
      recurse(arr);
      // Insert a dependency for every output to synchronize
      // with at the end.
      if (!arr.is_evaled()) {
        deps.insert({arr.id(), std::shared_future<void>{}});
      }
    }
  }

  while (!tape.empty()) {
    auto arr = std::move(tape.front());
    tape.pop();
    if (arr.is_evaled()) {
      if (!retain_graph && arr.has_primitive()) {
        arr.detach();
      }
      continue;
    }

    auto stream = arr.primitive().stream();
    std::vector<std::shared_future<void>> arr_deps;
    for (auto& in : arr.inputs()) {
      if (auto it = deps.find(in.id()); it != deps.end()) {
        arr_deps.push_back(it->second);
      }
    }
    std::shared_ptr<std::promise<void>> p{nullptr};
    if (auto it = deps.find(arr.id()); it != deps.end()) {
      p = std::make_unique<std::promise<void>>();
      it->second = p->get_future().share();
    }

    if (arr.primitive().device() == Device::gpu) {
      if (!metal::is_available()) {
        throw std::runtime_error("Metal GPU is not available.");
      }
      scheduler::enqueue(
          stream,
          metal::make_task(
              arr, std::move(arr_deps), std::move(p), retain_graph));
    } else {
      auto task = [retain_graph,
                   arr,
                   stream,
                   arr_deps = std::move(arr_deps),
                   p = std::move(p)]() mutable {
        for (auto& d : arr_deps) {
          d.wait();
        }
        scheduler::notify_new_task(stream);
        arr.primitive().eval_cpu(arr.inputs(), arr);
        if (!retain_graph) {
          arr.detach();
        }
        if (p) {
          p->set_value();
        }
        scheduler::notify_task_completion(stream);
      };
      scheduler::enqueue(stream, std::move(task));
    }
  }
  for (auto& arr : outputs) {
    if (auto it = deps.find(arr.id()); it != deps.end()) {
      it->second.wait();
    }
  }
}

std::pair<std::vector<array>, std::vector<array>> vjp(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& primals,
    const std::vector<array>& cotans) {
  // Make tracers from given primals
  std::vector<array> primals_;
  for (auto& p : primals) {
    auto s = p.has_primitive() ? p.primitive().stream()
                               : default_stream(default_device());
    primals_.push_back(copy(p, s)); // Does not do a deep copy
    primals_.back().set_tracer(true);
  }

  // Pass tracer primals through the function
  // Any variables that depend on the primals are marked as tracers
  auto outputs = fun(primals_);

  // Map outputs to passed cotans while ignoring the outputs
  // that have stop_gradient called on them
  int cotan_index = 0;
  std::vector<std::pair<int, int>> output_cotan_pairs;
  for (int i = 0; i < outputs.size(); ++i) {
    auto& out = outputs[i];
    if (out.has_primitive()) {
      if (auto& p = out.primitive(); typeid(p) == typeid(StopGradient)) {
        continue;
      }
    }
    if (cotan_index >= cotans.size()) {
      throw std::invalid_argument(
          "[vjp] Number of outputs with gradient does not match number of cotangents.");
    }
    if (out.shape() != cotans[cotan_index].shape()) {
      throw std::invalid_argument(
          "[vjp] Output shape does not match shape of cotangent.");
    }
    output_cotan_pairs.emplace_back(i, cotan_index++);
  }

  // Topologically sort the compute graph, record outputs
  // in the tape if a gradient is needed.
  std::unordered_set<std::uintptr_t> cache;
  std::unordered_set<std::uintptr_t> calc_grad;
  for (auto& primal : primals_) {
    primal.set_tracer(false);
    calc_grad.insert(primal.id());
    cache.insert(primal.id());
  }

  std::vector<array> tape;

  std::function<void(array&)> recurse;
  recurse = [&](auto& a) {
    auto id = a.id();
    a.set_tracer(false);

    // Check if visited and add to cache if not
    if (auto inserted = cache.insert(id); !inserted.second) {
      return;
    }

    for (auto& input : a.editable_inputs()) {
      recurse(input);
    }

    // Stop grad
    if (a.has_primitive() && typeid(a.primitive()) == typeid(StopGradient)) {
      return;
    }

    // Calculate gradient if any inputs require gradient
    for (auto& input : a.inputs()) {
      if (calc_grad.find(input.id()) != calc_grad.end()) {
        tape.push_back(a);
        calc_grad.insert(id);
        break;
      }
    }
  };

  for (auto& out : outputs) {
    recurse(out);
  }

  // Run the tape backwards, computing vector-jacobian
  // products for each primitive
  std::unordered_map<std::uintptr_t, array> cotan_map;
  for (auto [out_idx, cotan_idx] : output_cotan_pairs) {
    cotan_map.insert({outputs[out_idx].id(), cotans[cotan_idx]});
  }
  for (auto it = tape.rbegin(); it != tape.rend(); ++it) {
    auto& a = *it;

    // Get the arguments whose gradients are needed
    std::vector<int> argnums;
    for (int i = 0; i < a.inputs().size(); ++i) {
      if (calc_grad.find(a.inputs()[i].id()) != calc_grad.end()) {
        argnums.push_back(i);
      }
    }

    auto cotan_it = cotan_map.find(a.id());
    if (cotan_it == cotan_map.end()) {
      continue;
    }

    auto cotan = cotan_map.extract(cotan_it).mapped();
    auto vjps = a.primitive().vjp(a.inputs(), cotan, argnums);
    auto s = a.primitive().stream();
    // Accumulate the vector-jacobian products for each input
    for (int i = 0; i < argnums.size(); ++i) {
      auto in_id = a.inputs()[argnums[i]].id();
      if (auto cotan_it = cotan_map.find(in_id); cotan_it != cotan_map.end()) {
        cotan_it->second = add(cotan_it->second, vjps[i], s);
      } else {
        cotan_map.insert({in_id, vjps[i]});
      }
    }
  }

  std::vector<array> vjps;
  for (auto& primal : primals_) {
    if (auto cotan_it = cotan_map.find(primal.id());
        cotan_it != cotan_map.end()) {
      vjps.push_back(cotan_it->second);
    } else {
      auto s = primal.has_primitive() ? primal.primitive().stream()
                                      : default_stream(default_device());
      vjps.push_back(zeros_like(primal, s));
    }
  }
  return {outputs, vjps};
}

std::pair<array, array> vjp(
    const std::function<array(const array&)>& fun,
    const array& primal,
    const array& cotan) {
  auto vec_fun = [fun](const std::vector<array>& inputs) {
    return std::vector<array>{fun(inputs[0])};
  };
  auto [outputs, vjps] = vjp(vec_fun, {primal}, {cotan});
  return {outputs[0], vjps[0]};
}

std::pair<std::vector<array>, std::vector<array>> jvp(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& primals,
    const std::vector<array>& tangents) {
  if (primals.size() != tangents.size()) {
    throw std::invalid_argument(
        "[jvp] Number of inputs does not match number of tangents.");
  }
  for (int i = 0; i < primals.size(); ++i) {
    if (primals[i].shape() != tangents[i].shape()) {
      throw std::invalid_argument(
          "[jvp] Input shape does not match shape of tangent.");
    }
  }

  std::vector<array> primals_;
  for (auto& p : primals) {
    auto s = p.has_primitive() ? p.primitive().stream()
                               : default_stream(default_device());
    primals_.push_back(copy(p, s)); // Does not do a deep copy
    primals_.back().set_tracer(true);
  }
  auto outputs = fun(primals_);

  // Topologically sort the compute graph, record outputs
  // in the tape if a gradient is needed.
  std::unordered_set<std::uintptr_t> cache;
  std::unordered_set<std::uintptr_t> calc_grad;
  for (auto& primal : primals_) {
    primal.set_tracer(false);
    calc_grad.insert(primal.id());
    cache.insert(primal.id());
  }

  std::vector<array> tape;

  std::function<void(array&)> recurse;
  recurse = [&](auto& a) {
    auto id = a.id();
    a.set_tracer(false);

    // Check if visited and add to cache if not
    if (auto inserted = cache.insert(id); !inserted.second) {
      return;
    }

    for (auto& input : a.editable_inputs()) {
      recurse(input);
    }

    // Stop grad
    if (a.has_primitive() && typeid(a.primitive()) == typeid(StopGradient)) {
      return;
    }

    // Calculate gradient if any inputs require gradient
    for (auto& input : a.inputs()) {
      if (calc_grad.find(input.id()) != calc_grad.end()) {
        tape.push_back(a);
        calc_grad.insert(id);
        break;
      }
    }
  };

  for (auto& out : outputs) {
    recurse(out);
  }
  std::unordered_map<std::uintptr_t, array> tan_map;
  for (int i = 0; i < primals_.size(); ++i) {
    tan_map.insert({primals_[i].id(), tangents[i]});
  }

  for (auto& a : tape) {
    // Get the arguments used in the jvp
    std::vector<int> argnums;
    std::vector<array> tangents;
    for (int i = 0; i < a.inputs().size(); ++i) {
      if (auto it = tan_map.find(a.inputs()[i].id()); it != tan_map.end()) {
        argnums.push_back(i);
        tangents.push_back(it->second);
      }
    }

    auto jvp = a.primitive().jvp(a.inputs(), tangents, argnums);
    tan_map.insert({a.id(), jvp});
  }

  std::vector<array> jvps;
  for (auto& out : outputs) {
    if (auto it = tan_map.find(out.id()); it != tan_map.end()) {
      jvps.push_back(it->second);
    } else {
      auto s = out.has_primitive() ? out.primitive().stream()
                                   : default_stream(default_device());
      jvps.push_back(zeros_like(out, s));
    }
  }
  return {outputs, jvps};
}

std::pair<array, array> jvp(
    const std::function<array(const array&)>& fun,
    const array& primal,
    const array& tangent) {
  auto vec_fun = [fun](const std::vector<array>& inputs) {
    return std::vector<array>{fun(inputs[0])};
  };
  auto [outputs, jvps] = jvp(vec_fun, {primal}, {tangent});
  return {outputs[0], jvps[0]};
}

ValueAndGradFn value_and_grad(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<int>& argnums) {
  if (argnums.empty()) {
    throw std::invalid_argument("[grad] Must specify at least one argument.");
  }
  return [fun, argnums](const std::vector<array>& inputs) {
    std::set<int> args;
    for (auto& arg : argnums) {
      args.insert(arg < 0 ? arg + inputs.size() : arg);
    }
    if (args.size() != argnums.size()) {
      throw std::invalid_argument(
          "[grad] Repeat argument number not allowed in grad.");
    }
    if (*args.begin() < 0 || *args.rbegin() >= inputs.size()) {
      std::ostringstream msg;
      msg << "[grad] Invalid argument number for function with "
          << inputs.size() << " inputs.";
      throw std::invalid_argument(msg.str());
    }

    auto gfun = [&fun, &inputs, &args](const std::vector<array>& ginputs) {
      std::vector<array> inputs_(inputs);
      auto argit = args.begin();
      for (int i = 0; i < ginputs.size(); ++i) {
        inputs_[*argit] = ginputs[i];
        ++argit;
      }
      auto outputs = fun(inputs_);
      for (int i = 1; i < outputs.size(); i++) {
        auto& out = outputs[i];
        auto s = out.has_primitive() ? out.primitive().stream()
                                     : default_stream(default_device());
        outputs[i] = stop_gradient(out, s);
      }
      return outputs;
    };

    std::vector<array> ginputs;
    for (auto arg : args) {
      ginputs.push_back(inputs[arg]);
    }
    // Set the incoming gradient as int32 so that it will be promoted to the
    // appropriate floating point type op(int, floatXX) -> floatXX for most ops
    auto [outputs, grads] = vjp(gfun, ginputs, {array(1)});
    return std::make_pair(outputs, grads);
  };
}

namespace detail {

std::pair<std::vector<array>, std::vector<array>> vmap_trace(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& inputs,
    const std::vector<int>& in_axes) {
  if (in_axes.size() != inputs.size()) {
    throw std::invalid_argument(
        "[vmap] The number of in axes must match the number of inputs.");
  }

  // Run the function on placeholder inputs
  // to get the original graph
  std::vector<array> s_inputs;
  for (int i = 0; i < inputs.size(); ++i) {
    if (in_axes[i] != -1) {
      if (inputs[i].ndim() == 0) {
        throw std::invalid_argument(
            "[vmap] Cannot vmap an input with zero dimensions.");
      }
      if (in_axes[i] > inputs[i].ndim()) {
        std::ostringstream msg;
        msg << "[vmap] Axis " << in_axes[i] << " invalid for input with "
            << inputs[i].ndim() << " dimensions.";
        throw std::invalid_argument(msg.str());
      }

      std::vector<int> shape = inputs[i].shape();
      shape.erase(shape.begin() + in_axes[i]);
      array in(shape, inputs[i].dtype(), nullptr, {});
      s_inputs.push_back(in);
      s_inputs.back().set_tracer(true);
    } else {
      s_inputs.push_back(inputs[i]);
    }
  }
  return {s_inputs, fun(s_inputs)};
}

std::vector<array> vmap_replace(
    const std::vector<array>& inputs,
    const std::vector<array>& s_inputs,
    const std::vector<array>& s_outputs,
    const std::vector<int>& in_axes,
    const std::vector<int>& out_axes) {
  if (out_axes.size() != s_outputs.size()) {
    throw std::invalid_argument(
        "[vmap] The number of out axes must match the number of outputs.");
  }

  std::unordered_map<std::uintptr_t, std::pair<array, int>> tmap;
  std::unordered_set<std::uintptr_t> needs_vmap;
  for (int i = 0; i < s_inputs.size(); ++i) {
    if (in_axes[i] != -1) {
      tmap.insert({s_inputs[i].id(), {inputs[i], in_axes[i]}});
      needs_vmap.insert(s_inputs[i].id());
    }
  }

  // Topologically sort the graph
  std::unordered_set<std::uintptr_t> cache;
  for (int i = 0; i < s_inputs.size(); ++i) {
    auto in = s_inputs[i];
    if (in_axes[i] != -1) {
      in.set_tracer(false);
    }
    cache.insert(in.id());
  }
  std::vector<array> tape;

  std::function<void(const array&)> recurse;

  recurse = [&](const array& a) {
    // Stop at inputs to the vmap function
    auto id = a.id();
    if (cache.find(id) != cache.end()) {
      return;
    }
    for (auto& input : a.inputs()) {
      recurse(input);
    }
    cache.insert(id);
    for (auto& input : a.inputs()) {
      if (needs_vmap.find(input.id()) != needs_vmap.end()) {
        needs_vmap.insert(id);
        tape.push_back(a);
        tape.back().set_tracer(false);
        break;
      }
    }
  };

  for (auto& out : s_outputs) {
    recurse(out);
  }

  // Transform each primitive in the graph with
  // its vmap implementation
  for (auto& a : tape) {
    std::vector<array> v_inputs;
    std::vector<int> v_axes;
    for (auto& in : a.inputs()) {
      auto map_it = tmap.find(in.id());
      if (map_it != tmap.end()) {
        v_inputs.push_back(map_it->second.first);
        v_axes.push_back(map_it->second.second);
      } else {
        v_inputs.push_back(in);
        v_axes.push_back(-1);
      }
    }
    auto out_and_axis = a.primitive().vmap(v_inputs, v_axes);
    tmap.insert({a.id(), out_and_axis});
  }

  // Populate the outputs and make sure all the output axes are
  // in the right place
  std::vector<array> outputs;
  for (int i = 0; i < s_outputs.size(); ++i) {
    auto map_it = tmap.find(s_outputs[i].id());
    if (map_it != tmap.end()) {
      auto& [out, vdim] = map_it->second;
      if (vdim != out_axes[i]) {
        if (out_axes[i] >= out.ndim()) {
          std::ostringstream msg;
          msg << "[vmap] Axis " << out_axes[i] << " invalid for output with "
              << out.ndim() << " dimensions.";
          throw std::invalid_argument(msg.str());
        }
        std::vector<int> reorder(out.ndim());
        std::iota(reorder.begin(), reorder.end(), 0);
        reorder.erase(reorder.begin() + vdim);
        reorder.insert(reorder.begin() + out_axes[i], vdim);
        out = transpose(out, reorder);
      }
      outputs.push_back(out);
    } else {
      outputs.push_back(s_outputs[i]);
    }
  }
  return outputs;
}

} // namespace detail

std::function<std::vector<array>(const std::vector<array>&)> vmap(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<int>& in_axes /* = {} */,
    const std::vector<int>& out_axes /* = {} */) {
  auto infer_axes = [](auto axes) {
    return !axes.empty() &&
        std::all_of(axes.begin(), axes.end(), [](int ax) { return ax < 0; });
  };
  if (infer_axes(in_axes) != infer_axes(out_axes)) {
    throw std::invalid_argument(
        "[vmap] Input (or output) axes must be "
        "specified if output (or input) axes are.");
  }
  auto vfun = [fun, in_axes = in_axes, out_axes = out_axes](
                  const std::vector<array>& inputs) mutable {
    if (in_axes.size() == 0) {
      in_axes.resize(inputs.size(), 0);
    }

    auto [trace_inputs, trace_outputs] =
        detail::vmap_trace(fun, inputs, in_axes);

    if (out_axes.size() == 0) {
      out_axes.resize(trace_outputs.size(), 0);
    }

    return detail::vmap_replace(
        inputs, trace_inputs, trace_outputs, in_axes, out_axes);
  };

  return vfun;
}

std::function<array(const array&, const array&)> vmap(
    const std::function<array(const array&, const array&)>& fun,
    int in_axis_a /* = 0 */,
    int in_axis_b /* = 0 */,
    int out_axis /* = 0 */) {
  auto vfun = vmap(
      [in_axis_a, in_axis_b, out_axis, fun](const std::vector<array>& inputs) {
        return std::vector<array>{fun(inputs[0], inputs[1])};
      },
      {in_axis_a, in_axis_b},
      {out_axis});
  return [vfun](const array& a, const array& b) { return vfun({a, b})[0]; };
}

std::function<array(const array&)> vmap(
    const std::function<array(const array&)>& fun,
    int in_axis /* = 0 */,
    int out_axis /* = 0 */) {
  auto vfun = vmap(
      [in_axis, out_axis, fun](const std::vector<array>& inputs) {
        return std::vector<array>{fun(inputs[0])};
      },
      {in_axis},
      {out_axis});
  return [vfun](const array& a) { return vfun({a})[0]; };
}

} // namespace mlx::core
