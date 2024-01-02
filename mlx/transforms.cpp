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

// Initialize the static tracing counter from transforms_impl.h .
//
// This is used to implement the in_tracing() function the returns true if we
// are currently under a function transformation.
int detail::InTracing::tracing_counter{0};

void simplify(const std::vector<array>& outputs) {
  // Some notes about how this function works
  //
  // Step 1: Traverse the graph and build a tape. During the graph
  // traversal we:
  //      - Build a map of inputs to their parents.
  //      - Record scalar inputs in a map in order to fuse them.
  // Step 2: Process the tape. A node in the tape has inputs and outputs.
  //      - Scalar inputs are replaced with their canoncial scalar
  //      - We check each inputs output nodes. Every output node that matches
  //        the current node gets fused into the current node.
  std::function<void(const GraphNode&)> recurse;
  std::queue<GraphNode> tape;
  std::unordered_set<std::uintptr_t> cache;
  std::unordered_map<std::uintptr_t, std::vector<std::pair<GraphNode, int>>>
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

  // DFS the graph to build the tape, and log parents and scalars
  recurse = [&](const GraphNode& a) {
    // Ignore leaf nodes
    if (!a.has_primitive()) {
      return;
    }

    auto id = a.id();
    if (cache.find(id) != cache.end()) {
      return;
    }
    for (int i = 0; i < a.inputs().size(); i++) {
      auto& in = a.inputs()[i];
      parents_map[in.id()].push_back({a, i});
      recurse(in.graph_node());
    }
    cache.insert(id);

    tape.push(a);
    for (auto& i : a.inputs()) {
      if (is_scalar(i)) {
        scalars.insert({get_scalar_rep(i), i});
      }
    }
  };
  for (auto& o : outputs) {
    if (is_scalar(o)) {
      scalars.insert({get_scalar_rep(o), o});
    }
    recurse(o.graph_node());
  }

  // Helper that fuses two nodes in the graph by setting the parents of the
  // source to point to the destination
  auto fuse = [&](GraphNode& dst, GraphNode& src) {
    for (int i = 0; i < src.outputs().size(); ++i) {
      auto src_parents = parents_map.find(src.outputs()[i].id());
      if (src_parents == parents_map.end()) {
        continue;
      }

      auto& pairs = parents_map[dst.outputs()[i].id()];
      for (auto& parent : src_parents->second) {
        parent.first.inputs()[parent.second] = dst.outputs()[i];
        pairs.push_back(parent);
      }
    }
  };

  cache.clear();

  // Depth-1 equivalence check.
  auto node_equivalent = [](const GraphNode& a, const GraphNode& b) {
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
    auto g = std::move(tape.front());
    tape.pop();
    if (cache.find(g.id()) != cache.end()) {
      continue;
    }

    // Check if we can fuse scalars
    for (int i = 0; i < g.inputs().size(); ++i) {
      auto& arr = g.inputs()[i];
      if (is_scalar(arr)) {
        auto scalar = scalars.find(get_scalar_rep(arr));
        if (scalar->second.id() != arr.id()) {
          arr = scalar->second;
          auto parents = parents_map.find(scalar->second.id());
          parents->second.push_back({g, i});
        }
      }
    }

    // Check if we can fuse nodes themselves
    // This node along with all of its inputs output nodes
    for (auto& arr : g.inputs()) {
      if (auto parents = parents_map.find(arr.id());
          parents != parents_map.end()) {
        // Check all pairs of parents and look for nodes
        // that are fusable.
        auto N = parents->second.size();
        std::vector<bool> mask(N, false);
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
            if (src.id() != dst.id() && node_equivalent(src, dst)) {
              cache.insert(src.id());
              fuse(dst, src);
              mask[j] = true;
            }
          }
        }
      }
    }
  }
}

void eval(const std::vector<array>& outputs) {
  std::function<void(const array&)> recurse;
  std::queue<array> tape;
  std::unordered_set<std::uintptr_t> cache;
  std::unordered_map<std::uintptr_t, std::shared_future<void>> deps;

  recurse = [&](const GraphNode& a) {
    auto id = a.id();
    if (cache.find(id) != cache.end()) {
      return;
    }
    for (auto in_arr : a.inputs()) {
      auto& in = in_arr.graph_node();

      recurse(in);
      // If one of the inputs is being computed on a different
      // stream, we need to manage the dependency.
      if (!in_arr.is_evaled()) {
        if (a.primitive().stream() != in.primitive().stream()) {
          deps.insert({in.id(), std::shared_future<void>{}});
        }
      }
    }
    cache.insert(id);
    if (!a.is_evaled() || (!a.is_tracer() && a.has_primitive())) {
      if (!a.has_primitive()) {
        throw std::invalid_argument(
            "[eval] Attempting to eval an array without a primitive.");
      }
      tape.push(a);
    }
  };

  for (auto& arr : outputs) {
    if (!arr.is_evaled() || (!arr.is_tracer() && arr.has_primitive())) {
      recurse(arr);
      // Insert a dependency for every output to synchronize
      // with at the end.
      if (!arr.is_evaled() && deps.find(g.id()) == deps.end()) {
        deps.insert({g.id(), std::shared_future<void>{}});
      }
    }
  }

  while (!tape.empty()) {
    auto g = std::move(tape.front());
    tape.pop();
    if (arr.is_evaled()) {
      if (!arr.is_tracer() && arr.has_primitive()) {
        arr.detach();
      }
      continue;
    }

    auto stream = g.primitive().stream();
    std::vector<std::shared_future<void>> g_deps;
    for (auto& in : g.inputs()) {
      if (in.has_primitive()) {
        if (auto it = deps.find(in.graph_node().id()); it != deps.end()) {
          g_deps.push_back(it->second);
        }
      }
    }
    std::shared_ptr<std::promise<void>> p{nullptr};
    if (auto it = deps.find(g.id()); it != deps.end()) {
      p = std::make_unique<std::promise<void>>();
      it->second = p->get_future().share();
    }

    if (g.primitive().device() == Device::gpu) {
      if (!metal::is_available()) {
        throw std::runtime_error("Metal GPU is not available.");
      }
      scheduler::enqueue(
          stream, metal::make_task(arr, std::move(arr_deps), std::move(p)));
    } else {
      auto task = [arr,
                   stream,
                   g = std::move(g),
                   deps = std::move(g_deps),
                   p = std::move(p)]() mutable {
        for (auto& d : deps) {
          d.wait();
        }
        scheduler::notify_new_task(stream);
        arr.primitive().eval_cpu(arr.inputs(), arr);
        if (!arr.is_tracer()) {
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
    if (auto it = deps.find(arr.graph_node().id()); it != deps.end()) {
      it->second.wait();
      deps.erase(it);
    }
  }
}

std::pair<std::vector<array>, std::vector<array>> vjp(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& primals,
    const std::vector<array>& cotans) {
  // Set the global tracing flag.
  detail::InTracing in_tracing;

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

  // Topologically sort the compute graph, add graph nodes
  // to the tape which need a gradient.
  std::unordered_set<std::uintptr_t> cache;
  std::unordered_set<std::uintptr_t> calc_grad;
  for (auto& primal : primals_) {
    primal.set_tracer(false);
    calc_grad.insert(primal.id());
    cache.insert(primal.graph_node().id());
  }

  std::vector<GraphNode> tape;

  std::function<void(const GraphNode&)> recurse;
  recurse = [&](const auto& a) {
    // Check if visited and add to cache if not
    if (auto inserted = cache.insert(a.id()); !inserted.second) {
      return;
    }

    for (auto input : a.inputs()) {
      input.set_tracer(false);
      recurse(input.graph_node());
    }

    // Stop grad
    if (a.has_primitive()) {
      if (auto& p = a.primitive(); typeid(p) == typeid(StopGradient)) {
        return;
      }
    }

    // Calculate gradient if any inputs require gradient
    for (auto& input : a.inputs()) {
      if (calc_grad.find(input.id()) != calc_grad.end()) {
        tape.push_back(a);
        for (auto& output : a.outputs()) {
          calc_grad.insert(output.id());
        }
        break;
      }
    }
  };

  for (auto out : outputs) {
    out.set_tracer(false);
    recurse(out.graph_node());
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

    auto has_cotans = std::any_of(
        a.outputs().cbegin(), a.outputs().cend(), [&cotan_map](auto& o) {
          return cotan_map.find(o.id()) != cotan_map.end();
        });
    if (!has_cotans) {
      continue;
    }

    auto s = a.primitive().stream();
    std::vector<array> cotangents;
    for (auto& o : a.outputs()) {
      if (auto cotan_it = cotan_map.find(o.id()); cotan_it != cotan_map.end()) {
        cotangents.push_back(cotan_map.extract(cotan_it).mapped());
      } else {
        cotangents.push_back(zeros_like(o, s));
      }
    }
    auto vjps = a.primitive().vjp(a.inputs(), cotangents, argnums);
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

  // Set the global tracing flag.
  detail::InTracing in_tracing;

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

  std::vector<GraphNode> tape;

  std::function<void(const GraphNode&)> recurse;
  recurse = [&](const auto& a) {
    // Check if visited and add to cache if not
    if (auto inserted = cache.insert(a.id()); !inserted.second) {
      return;
    }

    for (auto input : a.inputs()) {
      input.set_tracer(false);
      recurse(input.graph_node());
    }

    // Stop grad
    if (a.has_primitive()) {
      if (auto& p = a.primitive(); typeid(p) == typeid(StopGradient)) {
        return;
      }
    }

    // Calculate gradient if any inputs require gradient
    for (auto& input : a.inputs()) {
      if (calc_grad.find(input.id()) != calc_grad.end()) {
        tape.push_back(a);
        for (auto& output : a.outputs()) {
          calc_grad.insert(output.id());
        }
        break;
      }
    }
  };

  for (auto out : outputs) {
    out.set_tracer(false);
    recurse(out.graph_node());
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

    // TODO, this should be one jvp per output argument
    auto jvps = a.primitive().jvp(a.inputs(), tangents, argnums);
    for (int i = 0; i < jvps.size(); ++i) {
      // for (int i = 0; i < argnums.size(); ++i) {
      tan_map.insert({a.outputs()[i].id(), jvps[i]});
      // tan_map.insert({a.outputs()[argnums[i]].id(), jvps[i]});
    }
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
  // Set the global tracing flag
  InTracing in_tracing;

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
      // TODO, awni, check this works with graph node
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
  std::unordered_set<std::uintptr_t> cache;
  for (int i = 0; i < s_inputs.size(); ++i) {
    auto in = s_inputs[i];
    if (in_axes[i] != -1) {
      tmap.insert({in.id(), {inputs[i], in_axes[i]}});
      needs_vmap.insert(in.id());
      in.set_tracer(false);
    }
    cache.insert(in.graph_node().id());
  }

  // Topologically sort the graph
  std::vector<GraphNode> tape;

  std::function<void(const GraphNode&)> recurse;

  recurse = [&](const GraphNode& a) {
    auto id = a.id();
    if (cache.find(id) != cache.end()) {
      return;
    }
    cache.insert(id);
    // Recurse on any child nodes
    for (auto& input : a.inputs()) {
      recurse(input.graph_node());
    }
    // If any input needs a vmap, then the outputs also need
    // a vmap
    for (auto& input : a.inputs()) {
      if (needs_vmap.find(input.id()) != needs_vmap.end()) {
        tape.push_back(a);
        for (auto o : a.outputs()) {
          needs_vmap.insert(o.id());
          o.set_tracer(false);
        }
        break;
      }
    }
  };

  for (auto& out : s_outputs) {
    if (out.has_primitive()) {
      recurse(out.graph_node());
    }
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
    auto [v_outputs, v_out_axes] = a.primitive().vmap(v_inputs, v_axes);
    // For each primitives outputs add its id, the vout id and the vax
    for (int i = 0; i < v_outputs.size(); ++i) {
      tmap.insert({a.outputs()[i].id(), {v_outputs[i], v_out_axes[i]}});
    }
  }

  // Populate the outputs and make sure all the output axes are
  // in the right place
  std::vector<array> outputs;
  for (int i = 0; i < s_outputs.size(); ++i) {
    if (auto map_it = tmap.find(s_outputs[i].id()); map_it != tmap.end()) {
      auto& [out, vdim] = map_it->second;
      if (vdim != out_axes[i]) {
        if (out_axes[i] >= out.ndim()) {
          std::ostringstream msg;
          msg << "[vmap] Axis " << out_axes[i] << " invalid for output with "
              << out.ndim() << " dimensions.";
          throw std::invalid_argument(msg.str());
        }
        out = moveaxis(out, vdim, out_axes[i]);
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
