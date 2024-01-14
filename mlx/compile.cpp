// Copyright © 2023 Apple Inc.
#include <iostream> // TODO

#include <unordered_map>
#include <unordered_set>

#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"

namespace mlx::core {

using CompileFn = std::function<std::vector<array>(const std::vector<array>&)>;

template <typename T, typename... U>
size_t getAddress(std::function<T(U...)> f) {
  typedef T(fnType)(U...);
  fnType** fnPointer = f.template target<fnType*>();
  return (size_t)*fnPointer;
}

struct CompilerCache {
  struct CacheEntry {
    std::vector<array> inputs;
    std::vector<array> outputs;
    std::vector<array> tape;
    bool empty{true};
  };

  // Returns a reference to a CacheEntry which can be updated
  // by the caller to avoid copying large tapes / inputs / outputs
  CacheEntry& find(const CompileFn& fn, const std::vector<array>& inputs) {
    // Try to find the entry
    auto inserted = cache_.insert({getAddress(fn), {}});
    auto& entries = inserted.first->second;
    auto is_match = [](const std::vector<array>& in1,
                       const std::vector<array>& in2) {
      if (in1.size() != in2.size()) {
        throw std::runtime_error(
            "[compiler] Got different number of inputs to function,"
            " this should never happen.");
      }
      for (int i = 0; i < in1.size(); ++i) {
        if (in1[i].shape() != in2[i].shape()) {
          return false;
        }
        if (in1[i].dtype() != in2[i].dtype()) {
          return false;
        }
      }
      return true;
    };

    // Loop over entries and check inputs match i.e. shapes and types must be
    // equal. Note this could get really slow if one compiles the same
    // function with many different shapes. May want to store entries in a
    // more easily searchable structure.
    for (auto& entry : entries) {
      // Check the inputs match and return if so
      if (is_match(inputs, entry.inputs)) {
        return entry;
      }
    }
    // Otherwise append a new cache entry
    entries.push_back(CacheEntry{});
    return entries.back();
  };

 private:
  CompilerCache() {}
  friend CompilerCache& compiler_cache();
  std::unordered_map<size_t, std::vector<CacheEntry>> cache_;
};

CompilerCache& compiler_cache() {
  static CompilerCache compiler_cache_;
  return compiler_cache_;
}

std::pair<std::vector<array>, std::vector<array>> compile_trace(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    const std::vector<array>& inputs) {
  // Set the global tracing flag.
  detail::InTracing in_tracing;

  // Run the function on placeholder inputs
  // to get compute graph
  std::vector<array> tracer_inputs;
  for (int i = 0; i < inputs.size(); ++i) {
    array in(inputs[i].shape(), inputs[i].dtype(), nullptr, {});
    in.set_tracer(true);
    tracer_inputs.push_back(std::move(in));
  }
  return {tracer_inputs, fun(tracer_inputs)};
}

std::vector<array> compile_dfs_graph(
    const std::vector<array>& inputs,
    const std::vector<array>& outputs) {
  std::unordered_set<std::uintptr_t> needs_compile;
  std::unordered_set<std::uintptr_t> cache;
  for (int i = 0; i < inputs.size(); ++i) {
    auto in = inputs[i];
    needs_compile.insert(in.id());
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
      if (needs_compile.find(input.id()) != needs_compile.end()) {
        tape.push_back(a);
        needs_compile.insert(a.id());
        for (auto s : a.siblings()) {
          needs_compile.insert(s.id());
        }
        break;
      }
    }
  };

  for (auto& out : outputs) {
    if (out.has_primitive()) {
      recurse(out);
    }
  }
  return tape;
}

std::vector<array> compile_tape_replace(
    const std::vector<array>& tape,
    const std::vector<array>& trace_inputs,
    const std::vector<array>& trace_outputs,
    const std::vector<array>& inputs) {
  std::unordered_map<uintptr_t, array> trace_to_real;
  for (int i = 0; i < inputs.size(); ++i) {
    trace_to_real.insert({trace_inputs[i].id(), inputs[i]});
  }

  // We need a map here of traced inputs to real inputs
  for (auto& a : tape) {
    if (!a.has_primitive()) {
      std::runtime_error(
          "[compile] Something went wrong, no primitive in tape");
    }
    std::vector<array> real_inputs;
    for (auto& in : a.inputs()) {
      real_inputs.push_back(trace_to_real.at(in.id()));
    }
    auto real_a =
        array(a.shape(), a.dtype(), a.primitive_ptr(), std::move(real_inputs));
    trace_to_real.insert({a.id(), std::move(real_a)});
  }
  std::vector<array> outputs;
  for (auto& o : trace_outputs) {
    outputs.push_back(trace_to_real.at(o.id()));
  }
  return outputs;
}

std::function<std::vector<array>(const std::vector<array>&)> compile(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun) {
  //  std::cout << getAddress(fun) << std::endl;
  return [&fun](const std::vector<array>& inputs) {
    // Find a cache entry with the correct inputs
    auto& entry = compiler_cache().find(fun, inputs);

    // No matching cache entry existed, so compile
    if (entry.empty) {
      std::cout << "RECOMPILING? " << std::endl;
      // Mark the entry as not empty since we are about to fill it
      entry.empty = false;

      // Trace te build the graph
      std::tie(entry.inputs, entry.outputs) = compile_trace(fun, inputs);

      // This is a good point to do optimizations:
      // - simplify
      // - kernel fusion to generate new primitives
      // - may make sense to keep the tape from simplify
      //   and pass it around so that we don't have to keep rebuilding it

      // Recurse to build the tape
      entry.tape = compile_dfs_graph(entry.inputs, entry.outputs);
    }

    // At this point we must have a tape, now replace the placeholders
    // with real arrays that can be evaluated
    return compile_tape_replace(
        entry.tape, entry.inputs, entry.outputs, inputs);
  };
}

} // namespace mlx::core
