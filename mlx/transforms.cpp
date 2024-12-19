// Copyright © 2023-2024 Apple Inc.
#include <algorithm>
#include <deque>
#include <future>
#include <numeric>
#include <set>
#include <sstream>
#include <stack>
#include <unordered_map>
#include <unordered_set>

#include "mlx/backend/metal/metal_impl.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"
#include "mlx/utils.h"

namespace mlx::core {

/* This class is only meant to be used in eval
 * for synchronizing with the main thread. */
class Synchronizer : public Primitive {
 public:
  explicit Synchronizer(Stream stream) : Primitive(stream) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {}
  void eval_gpu(const std::vector<array>&, std::vector<array>&) override {}

  DEFINE_PRINT(Synchronize);
};

// Initialize the static tracing members from transforms_impl.h
//
// These are used to implement the in_tracing() function the returns true if we
// are currently under a function transformation and the retain_graph()
// function which returns true if we are forced to retain the graph during
// evaluation.
std::vector<bool> detail::InTracing::trace_stack{};
int detail::RetainGraph::tracing_counter{0};

array eval_impl(std::vector<array> outputs, bool async) {
  std::deque<array> tape;

  // stream events to use for synchronization
  std::unordered_map<uint32_t, Event> events;

  // Make an effort to choose a good output stream
  Stream stream = default_stream(default_device());
  for (auto& o : outputs) {
    if (o.status() == array::Status::unscheduled && o.has_primitive()) {
      stream = o.primitive().stream();
      break;
    }
  }

  std::unordered_set<uintptr_t> needs_signal;

  auto synchronizer = array(
      {}, bool_, std::make_shared<Synchronizer>(stream), std::move(outputs));
  needs_signal.insert(synchronizer.id());

  // Make an event for the synchronizer stream
  events.emplace(stream.index, Event{stream});

  {
    // Record the degree of each input
    std::unordered_map<std::uintptr_t, int> cache;

    std::stack<std::pair<std::reference_wrapper<array>, int>> dfs;
    dfs.emplace(synchronizer, 0);
    while (!dfs.empty()) {
      auto& [a_ref, idx] = dfs.top();
      auto& a = a_ref.get();

      if (idx < a.inputs().size()) {
        // Add an input, and continue
        auto& in = a.inputs()[idx++];

        // Ignore arrays already scheduled
        if (in.status() == array::Status::scheduled) {
          continue;
        }

        if (in.status() == array::Status::unscheduled) {
          if (async && in.is_tracer()) {
            throw std::invalid_argument(
                "[async_eval] Not allowed inside a graph transformation.");
          }
          if (!in.has_primitive()) {
            if (in.is_tracer()) {
              throw std::invalid_argument(
                  "[eval] Attempting to eval an array during function"
                  " transformations like compile or vmap is not allowed.");
            }
            throw std::runtime_error(
                "[eval] Attempting to eval an array without a primitive.\n"
                "If you are compiling a function, make sure all the inputs "
                "and outputs are captured:\n"
                "https://ml-explore.github.io/mlx/build/html/usage/compile.html#pure-functions.\n"
                "If you are not using compile, this may be a bug. "
                "Please file an issue here:\n"
                "https://github.com/ml-explore/mlx/issues.");
          }
          if (a.primitive().stream() != in.primitive().stream()) {
            needs_signal.insert(in.id());
          }
        }

        // All siblings have the same degree
        auto cache_it = cache.find(in.id());
        if (cache_it == cache.end()) {
          dfs.emplace(in, 0);
          cache.insert({in.id(), 1});
          for (auto& s : in.siblings()) {
            cache.insert({s.id(), 1});
          }
        } else {
          cache_it->second++;
          for (auto& s : in.siblings()) {
            cache[s.id()]++;
          }
        }
        continue;
      }
      if ((a.status() != array::Status::unscheduled) && !a.is_tracer() &&
          a.has_primitive()) {
        // If the array is evaluated and is no longer a tracer, detach it
        a.detach();
      }
      dfs.pop();
    }

    // Build the tape in BFS order with a width limit
    int max_width = env::bfs_max_width();
    dfs = std::stack<std::pair<std::reference_wrapper<array>, int>>();
    tape.push_back(synchronizer);
    for (int i = 0; !cache.empty() && (i < tape.size() || !dfs.empty());) {
      auto& a = (i >= tape.size()) ? dfs.top().first.get() : tape[i];
      int j = 0;
      if (i >= tape.size()) {
        j = dfs.top().second;
        dfs.pop();
      } else {
        i++;
      }
      for (; j < a.inputs().size(); ++j) {
        auto& in = a.inputs()[j];
        if (in.status() != array::Status::unscheduled) {
          continue;
        }

        // If the width limit is exceeded, push the array on the stack
        // and go down a level
        if ((tape.size() - i) >= max_width) {
          dfs.emplace(a, j);
          break;
        }

        auto it = cache.find(in.id());
        it->second -= 1;

        if (it->second != 0) {
          for (auto& s : in.siblings()) {
            cache[s.id()] -= 1;
          }
          continue;
        }

        // Remove input and siblings from cache
        cache.erase(it);
        for (auto& s : in.siblings()) {
          cache.erase(s.id());
        }

        tape.push_back(in);
      }
    }
  }

  while (!tape.empty()) {
    auto arr = std::move(tape.back());
    tape.pop_back();

    auto stream = arr.primitive().stream();

    // Lookup corresponding event and increment counter
    auto e = events.find(stream.index);
    if (e == events.end()) {
      e = events.emplace(stream.index, Event{stream}).first;
    }
    e->second.set_value(e->second.value() + 1);
    arr.attach_event(e->second);
    for (auto& s : arr.siblings()) {
      s.attach_event(e->second);
    }

    // Set the status of the array and siblings.
    arr.set_status(array::Status::scheduled);
    for (auto& s : arr.siblings()) {
      s.set_status(array::Status::scheduled);
    }

    std::vector<std::shared_future<void>> arr_deps;
    bool signal = needs_signal.find(arr.id()) != needs_signal.end();

    if (arr.primitive().device() == Device::gpu) {
      if (!metal::is_available()) {
        throw std::runtime_error("Metal GPU is not available.");
      }
      scheduler::enqueue(stream, metal::make_task(std::move(arr), signal));
    } else {
      auto task = [arr = std::move(arr), stream, signal]() mutable {
        for (auto& input : arr.inputs()) {
          if (input.event().valid() &&
              input.event().stream() != arr.primitive().stream()) {
            input.event().wait();
          }
        }
        scheduler::notify_new_task(stream);
        auto outputs = arr.outputs();
        arr.primitive().eval_cpu(arr.inputs(), outputs);
        if (!arr.is_tracer()) {
          arr.detach();
        }
        for (auto& out : outputs) {
          out.set_status(array::Status::available);
        }

        if (signal) {
          arr.event().signal();
        }

        scheduler::notify_task_completion(stream);
      };
      scheduler::enqueue(stream, std::move(task));
    }
  }
  return synchronizer;
}

void async_eval(std::vector<array> outputs) {
  if (outputs.empty()) {
    return;
  }

  if (std::none_of(outputs.begin(), outputs.end(), [](array& x) {
        return x.status() == array::Status::unscheduled;
      })) {
    return;
  }

  eval_impl(std::move(outputs), true);
}

void eval(std::vector<array> outputs) {
  if (outputs.empty()) {
    return;
  }

  if (std::none_of(outputs.begin(), outputs.end(), [](array& x) {
        return x.status() == array::Status::unscheduled;
      })) {
    for (auto& x : outputs) {
      x.wait();
    }
    return;
  }

  eval_impl(std::move(outputs), false).event().wait();
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
      std::ostringstream msg;
      msg << "[vjp] Number of outputs to compute gradients for ("
          << outputs.size() << ") does not match number of cotangents ("
          << cotans.size() << ").";
      throw std::invalid_argument(msg.str());
    }
    if (out.shape() != cotans[cotan_index].shape()) {
      std::ostringstream msg;
      msg << "[vjp] Output shape " << out.shape()
          << " does not match cotangent shape " << cotans[cotan_index].shape()
          << ".";
      if (outputs.size() == 1 && out.size() == 1) {
        msg << " If you are using grad your function must return a scalar.";
      }
      throw std::invalid_argument(msg.str());
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
    cache.insert(primal.id());
  }

  std::vector<array> tape;

  std::function<void(array&)> recurse;
  recurse = [&](auto& a) {
    // Check if visited and add to cache if not
    if (auto inserted = cache.insert(a.id()); !inserted.second) {
      return;
    }
    a.set_tracer(false);
    for (auto& s : a.siblings()) {
      s.set_tracer(false);
      cache.insert(s.id());
    }

    for (auto& input : a.inputs()) {
      recurse(input);
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
        calc_grad.insert(a.id());
        for (auto& s : a.siblings()) {
          calc_grad.insert(s.id());
        }
        break;
      }
    }
  };

  for (auto out : outputs) {
    recurse(out);
  }

  // Run the tape backwards, computing vector-jacobian
  // products for each primitive
  std::unordered_map<std::uintptr_t, array> cotan_map;
  for (auto [out_idx, cotan_idx] : output_cotan_pairs) {
    auto& o = outputs[out_idx];
    auto s = o.has_primitive() ? o.primitive().stream()
                               : default_stream(default_device());
    cotan_map.insert({o.id(), astype(cotans[cotan_idx], o.dtype(), s)});
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

    // Check if any of the array or its siblings have cotangents,
    // if not, we can skip this primitive
    auto outputs = a.outputs();
    bool has_cotans =
        std::any_of(outputs.cbegin(), outputs.cend(), [&cotan_map](auto& s) {
          return cotan_map.find(s.id()) != cotan_map.end();
        });
    if (!has_cotans) {
      continue;
    }

    auto s = a.primitive().stream();
    std::vector<array> cotangents{};
    for (auto& o : outputs) {
      if (auto cotan_it = cotan_map.find(o.id()); cotan_it != cotan_map.end()) {
        cotangents.push_back(cotan_map.extract(cotan_it).mapped());
      } else {
        cotangents.push_back(zeros_like(o, s));
      }
    }

    std::vector<array> vjps;
    {
      detail::RetainGraph retain;
      vjps = a.primitive().vjp(a.inputs(), cotangents, argnums, outputs);
    }
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
  // Set the global tracing flag.
  detail::InTracing in_tracing;

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
    // Check if visited and add to cache if not
    if (auto inserted = cache.insert(a.id()); !inserted.second) {
      return;
    }
    a.set_tracer(false);
    for (auto& s : a.siblings()) {
      s.set_tracer(false);
      cache.insert(s.id());
    }

    for (auto input : a.inputs()) {
      recurse(input);
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
        calc_grad.insert(a.id());
        for (auto& s : a.siblings()) {
          calc_grad.insert(s.id());
        }
        break;
      }
    }
  };

  for (auto out : outputs) {
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

    auto jvps = a.primitive().jvp(a.inputs(), tangents, argnums);
    auto outputs = a.outputs();
    for (int i = 0; i < jvps.size(); ++i) {
      tan_map.insert({outputs[i].id(), jvps[i]});
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
    // Set the incoming gradient to int32, vjp will cast it to the output type
    auto [outputs, grads] = vjp(gfun, ginputs, {array(1.0f)});
    return std::make_pair(outputs, grads);
  };
}

namespace detail {

std::pair<std::vector<array>, std::vector<array>> vmap_trace(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& inputs,
    const std::vector<int>& in_axes) {
  // Set the global tracing flag.
  detail::InTracing in_tracing;

  if (in_axes.size() != inputs.size()) {
    std::stringstream ss;
    ss << "[vmap] The number of in axes (" << in_axes.size()
       << ") must match the number of inputs (" << inputs.size() << ").";
    throw std::invalid_argument(ss.str());
  }

  // Some error checking and get the vmap axis size
  size_t vmap_ax_size;
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
      vmap_ax_size = inputs[i].shape(in_axes[i]);
    }
  }
  // Check that all vmapped axes have the same size
  for (int i = 0; i < inputs.size(); ++i) {
    if (in_axes[i] != -1) {
      if (size_t in_ax = inputs[i].shape(in_axes[i]); vmap_ax_size != in_ax) {
        std::ostringstream msg;
        msg << "[vmap] Inconsistent axis sizes: " << in_ax << " and "
            << vmap_ax_size << ".";
        throw std::invalid_argument(msg.str());
      }
    }
  }

  // Run the function on placeholder inputs
  // to get the original graph
  std::vector<array> s_inputs;
  for (int i = 0; i < inputs.size(); ++i) {
    if (in_axes[i] != -1) {
      auto shape = inputs[i].shape();
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
    std::stringstream msg;
    msg << "[vmap] The number of out axes (" << out_axes.size()
        << ") must match the number of outputs (" << s_outputs.size() << ").";
    throw std::invalid_argument(msg.str());
  }

  int vmap_size = -1;
  for (int i = 0; i < inputs.size(); ++i) {
    if (in_axes[i] >= 0) {
      vmap_size = inputs[i].shape(in_axes[i]);
      break;
    }
  }
  if (vmap_size == -1) {
    throw std::invalid_argument("At least one of in_axes must be non-None.");
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
    cache.insert(in.id());
  }

  // Topologically sort the graph
  std::vector<array> tape;

  std::function<void(const array&)> recurse;

  recurse = [&](const array& a) {
    auto id = a.id();
    if (cache.find(id) != cache.end()) {
      return;
    }
    cache.insert(id);
    for (auto& s : a.siblings()) {
      cache.insert(s.id());
    }

    // Recurse on inputs
    for (auto& input : a.inputs()) {
      recurse(input);
    }
    // If any input needs a vmap, then the outputs also need
    // a vmap
    for (auto& input : a.inputs()) {
      if (needs_vmap.find(input.id()) != needs_vmap.end()) {
        tape.push_back(a);
        tape.back().set_tracer(false);
        needs_vmap.insert(a.id());
        for (auto s : a.siblings()) {
          needs_vmap.insert(s.id());
          s.set_tracer(false);
        }
        break;
      }
    }
  };

  for (auto& out : s_outputs) {
    if (out.has_primitive()) {
      recurse(out);
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

    // For each primitive's outputs add its id, the vout id and the vax
    auto outputs = a.outputs();
    for (int i = 0; i < v_outputs.size(); ++i) {
      tmap.insert({outputs[i].id(), {v_outputs[i], v_out_axes[i]}});
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
      // When the output has no input dependencies
      // use the size of the vmapped axis in the inputs to expand the output
      array output = expand_dims(s_outputs[i], out_axes[i]);
      output = repeat(output, vmap_size, out_axes[i]);
      outputs.push_back(output);
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

std::function<std::vector<array>(const std::vector<array>&)> custom_function(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    std::optional<std::function<std::vector<array>(
        const std::vector<array>&,
        const std::vector<array>&,
        const std::vector<array>&)>> fun_vjp /* = std::nullopt */,
    std::optional<std::function<std::vector<array>(
        const std::vector<array>&,
        const std::vector<array>&,
        const std::vector<int>&)>> fun_jvp /* = std::nullopt */,
    std::optional<std::function<std::pair<std::vector<array>, std::vector<int>>(
        const std::vector<array>&,
        const std::vector<int>&)>> fun_vmap /* = std::nullopt */) {
  if (!fun_vjp.has_value() && !fun_jvp.has_value() && !fun_vmap.has_value()) {
    return fun;
  }

  return [fun = std::move(fun),
          fun_vjp = std::move(fun_vjp),
          fun_jvp = std::move(fun_jvp),
          fun_vmap = std::move(fun_vmap)](const std::vector<array>& args) {
    // Compute the outputs
    auto outputs = fun(args);
    for (auto& out : outputs) {
      out = stop_gradient(out);
    }

    // Prepare the inputs to the primitive
    // We also add the outputs to the primitive so that it can "run" the forward
    // pass.
    std::vector<array> inputs = args;
    inputs.insert(inputs.end(), outputs.begin(), outputs.end());

    // Compute the stream. Maybe do it in a smarter way at some point in the
    // future.
    Stream s = (outputs[0].has_primitive()) ? outputs[0].primitive().stream()
                                            : default_stream(default_device());

    // Make the output info
    std::vector<Shape> shapes;
    std::vector<Dtype> dtypes;
    for (const auto& out : outputs) {
      shapes.emplace_back(out.shape());
      dtypes.emplace_back(out.dtype());
    }

    return array::make_arrays(
        std::move(shapes),
        dtypes,
        std::make_shared<CustomTransforms>(
            to_stream(s),
            outputs.size(),

            // We use the passed vjp function or compute it from the inputs and
            // passed cotangents. Note that this may be less efficient than
            // using `fun` directly because we may not be able to fully reuse
            // the outputs of the forward pass.
            fun_vjp.value_or(
                [fun](auto primals, auto cotangents, auto outputs) {
                  auto [__, vjps] = vjp(fun, primals, cotangents);
                  return vjps;
                }),

            // We use the passed jvp function or compute it from the primals
            // and tangents. Similarly we can't take full advantage of the
            // argnums so it is best to use `fun` directly if we don't need a
            // custom transform.
            //
            // TODO: Use stop_gradient to make full use of argnums and not
            //       waste computation.
            fun_jvp.value_or([fun](auto primals, auto tangents, auto argnums) {
              std::vector<array> all_tangents;
              for (int i = 0, j = 0; i < primals.size(); i++) {
                if (j < argnums.size() && i == argnums[j]) {
                  all_tangents.emplace_back(tangents[j++]);
                } else {
                  all_tangents.emplace_back(zeros_like(primals[i]));
                }
              }
              auto [__, jvps] = jvp(fun, primals, all_tangents);
              return jvps;
            }),

            // Same as above, we use the passed vmap function or we compute it
            // from `fun`. The output axes is selected to be all 0s which again
            // may be suboptimal but the only thing we can do without any
            // information for `fun`.
            fun_vmap.value_or(
                [fun, out_size = outputs.size()](auto inputs, auto in_axes)
                    -> std::pair<std::vector<array>, std::vector<int>> {
                  std::vector<int> out_axes(out_size, 0);
                  return {vmap(fun, in_axes, out_axes)(inputs), out_axes};
                })),
        inputs);
  };
}

std::function<std::vector<array>(const std::vector<array>&)> custom_vjp(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    std::function<std::vector<array>(
        const std::vector<array>&,
        const std::vector<array>&,
        const std::vector<array>&)> fun_vjp) {
  return custom_function(fun, fun_vjp, std::nullopt, std::nullopt);
}

std::function<std::vector<array>(const std::vector<array>&)> checkpoint(
    std::function<std::vector<array>(const std::vector<array>&)> fun) {
  auto vjp_fun = [fun](
                     const std::vector<array>& primals,
                     const std::vector<array>& cotangents,
                     const std::vector<array>& outputs) -> std::vector<array> {
    auto [__, vjps] = vjp(fun, depends(primals, outputs), cotangents);
    return vjps;
  };

  return custom_vjp(fun, vjp_fun);
}

} // namespace mlx::core
