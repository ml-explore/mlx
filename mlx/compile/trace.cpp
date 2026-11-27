// Copyright © 2023-2024 Apple Inc.

#include <unordered_map>
#include <unordered_set>

#include "mlx/compile_impl.h"
#include "mlx/primitives.h"
#include "mlx/transforms_impl.h"
#include "mlx/utils.h"

namespace mlx::core::detail {

std::tuple<std::vector<array>, std::vector<array>, std::shared_ptr<void>>
compile_trace(
    const ArrayFnWithExtra& fun,
    const std::vector<array>& inputs,
    bool shapeless) {
  // Set the global tracing flag.
  detail::InTracing in_tracing{shapeless};

  // Run the function on placeholder inputs
  // to get compute graph
  std::vector<array> tracer_inputs;
  for (int i = 0; i < inputs.size(); ++i) {
    array in(inputs[i].shape(), inputs[i].dtype(), nullptr, {});
    in.set_tracer(true);
    tracer_inputs.push_back(std::move(in));
  }

  auto output = fun(tracer_inputs);
  return {tracer_inputs, output.first, output.second};
}

// Traverses the graph to build a tape and a map of array ids to their parents
std::pair<std::vector<array>, ParentsMap> compile_dfs(
    const std::vector<array>& inputs,
    std::vector<array>& outputs,
    const std::vector<array>& original_inputs) {
  std::vector<array> tape;
  std::unordered_map<std::uintptr_t, std::vector<std::pair<array, int>>>
      parents_map;
  {
    std::function<void(const array&)> recurse;
    std::unordered_set<std::uintptr_t> input_set;
    std::unordered_set<std::uintptr_t> original_input_set;
    for (int i = 0; i < inputs.size(); ++i) {
      input_set.insert(inputs[i].id());
      original_input_set.insert(original_inputs[i].id());
    }

    // DFS the graph to build the tape, and log parents and scalars
    std::unordered_set<std::uintptr_t> cache;
    recurse = [&](const array& a) {
      auto id = a.id();
      if (original_input_set.find(id) != original_input_set.end()) {
        throw std::invalid_argument(
            "[compile] Attempting to compile a function with uncaptured inputs is not allowed.");
      }
      if (cache.find(id) != cache.end()) {
        return;
      }
      for (int i = 0; i < a.inputs().size(); i++) {
        auto& in = a.inputs()[i];
        parents_map[in.id()].push_back({a, i});
        for (auto& s : a.siblings()) {
          parents_map[in.id()].push_back({s, i});
        }
        // Don't recurse on inputs (but add them to the tape for the purpose
        // of future optimizations)
        if (input_set.find(a.id()) == input_set.end()) {
          recurse(in);
        }
      }
      cache.insert(id);
      for (auto& s : a.siblings()) {
        cache.insert(s.id());
      }
      tape.push_back(a);
    };
    for (auto& a : outputs) {
      recurse(a);
    }
  }

  // Deep copy the tape and parents map while preserving inputs and outputs
  std::vector<array> new_tape;
  std::unordered_set<uintptr_t> io_set;
  std::unordered_map<uintptr_t, array> old_to_new;
  for (auto& o : outputs) {
    old_to_new.insert({o.id(), o});
    io_set.insert(o.id());
    for (auto& s : o.siblings()) {
      old_to_new.insert({s.id(), s});
      io_set.insert(s.id());
    }
  }
  for (auto& i : inputs) {
    io_set.insert(i.id());
    old_to_new.insert({i.id(), i});
  }

  new_tape.reserve(tape.size());
  for (auto& arr : tape) {
    if (!arr.has_primitive() || (io_set.find(arr.id()) != io_set.end())) {
      old_to_new.insert({arr.id(), arr});
      new_tape.push_back(arr);
      continue;
    }
    std::vector<array> inputs;
    inputs.reserve(arr.inputs().size());
    for (auto& i : arr.inputs()) {
      inputs.push_back(old_to_new.find(i.id())->second);
    }
    if (arr.siblings().size() > 0) {
      std::vector<Dtype> types;
      std::vector<Shape> shapes;
      auto out = arr.outputs();
      for (auto& o : out) {
        types.push_back(o.dtype());
        shapes.push_back(o.shape());
      }
      auto as = array::make_arrays(
          std::move(shapes), types, arr.primitive_ptr(), std::move(inputs));
      for (int i = 0; i < out.size(); ++i) {
        as[i].set_backward(out[i].is_backward());
        old_to_new.insert({out[i].id(), as[i]});
      }
      new_tape.push_back(as[arr.sibling_position()]);
    } else {
      auto a = array(
          arr.shape(), arr.dtype(), arr.primitive_ptr(), std::move(inputs));
      a.set_backward(arr.is_backward());
      old_to_new.insert({arr.id(), a});
      new_tape.push_back(a);
    }
  }
  io_set.clear();
  for (auto& o : outputs) {
    if (!(io_set.insert(o.id()).second)) {
      continue;
    }
    for (auto& i : o.inputs()) {
      i = old_to_new.find(i.id())->second;
    }
    for (auto& s : o.siblings()) {
      io_set.insert(s.id());
      for (auto& i : s.inputs()) {
        i = old_to_new.find(i.id())->second;
      }
    }
  }
  tape = std::move(new_tape);

  std::unordered_map<std::uintptr_t, std::vector<std::pair<array, int>>>
      new_parents_map;
  for (auto& [id, vec] : parents_map) {
    for (auto& [a, _] : vec) {
      a = old_to_new.find(a.id())->second;
    }
    new_parents_map[old_to_new.find(id)->second.id()] = std::move(vec);
  }
  parents_map = std::move(new_parents_map);
  return {tape, parents_map};
}

std::vector<array> compile_replace(
    const std::vector<array>& tape,
    const std::vector<array>& trace_inputs,
    const std::vector<array>& trace_outputs,
    const std::vector<array>& inputs,
    bool shapeless) {
  std::unordered_map<uintptr_t, array> trace_to_real;
  for (int i = 0; i < inputs.size(); ++i) {
    trace_to_real.insert({trace_inputs[i].id(), inputs[i]});
  }

  auto is_load = [](const Primitive& p) { return typeid(p) == typeid(Load); };

  for (auto& a : tape) {
    // Arrays in the tape without primitives are either:
    // - inputs, which are already in the map
    // - constants, which can be used directly
    // - a load primitive which has no inputs and will become a constant
    //   after the first eval
    if (!a.has_primitive() || is_load(a.primitive())) {
      trace_to_real.insert({a.id(), a});
    } else {
      // Find real inputs
      std::vector<array> real_inputs;
      for (auto& in : a.inputs()) {
        real_inputs.push_back(trace_to_real.at(in.id()));
      }
      if (a.siblings().empty()) {
        auto shape =
            shapeless ? a.primitive().output_shapes(real_inputs)[0] : a.shape();
        auto real_a = array(
            std::move(shape),
            a.dtype(),
            a.primitive_ptr(),
            std::move(real_inputs));
        trace_to_real.insert({a.id(), std::move(real_a)});
      } else {
        // Ensure the order is correct for multi-output primitives
        std::vector<Dtype> types;
        auto trace_out = a.outputs();
        for (auto& o : trace_out) {
          types.push_back(o.dtype());
        }
        std::vector<Shape> shapes;
        if (shapeless) {
          shapes = a.primitive().output_shapes(real_inputs);
        } else {
          for (auto& o : trace_out) {
            shapes.push_back(o.shape());
          }
        }
        auto real_out = array::make_arrays(
            std::move(shapes), types, a.primitive_ptr(), real_inputs);
        for (int i = 0; i < trace_out.size(); ++i) {
          trace_to_real.insert({trace_out[i].id(), std::move(real_out[i])});
        }
      }
    }
  }

  std::vector<array> outputs;
  for (auto& o : trace_outputs) {
    outputs.push_back(trace_to_real.at(o.id()));
  }
  return outputs;
}

} // namespace mlx::core::detail
