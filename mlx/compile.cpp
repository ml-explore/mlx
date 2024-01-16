// Copyright © 2023 Apple Inc.
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"

namespace mlx::core {

namespace detail {

using CompileFn = std::function<std::vector<array>(const std::vector<array>&)>;
using ParentsMap =
    std::unordered_map<std::uintptr_t, std::vector<std::pair<array, int>>>;

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
  CacheEntry& find(size_t fun_id, const std::vector<array>& inputs) {
    // Try to find the entry
    auto inserted = cache_.insert({fun_id, {}});
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

  void erase(size_t fun_id) {
    cache_.erase(fun_id);
  }

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

// Traverses the graph to build a tape and a map of array ids to their parents
std::pair<std::vector<array>, ParentsMap> compile_dfs(
    const std::vector<array>& inputs,
    const std::vector<array>& outputs) {
  std::function<void(const array&)> recurse;
  std::vector<array> tape;
  std::unordered_set<std::uintptr_t> cache;
  std::unordered_map<std::uintptr_t, std::vector<std::pair<array, int>>>
      parents_map;
  for (int i = 0; i < inputs.size(); ++i) {
    auto in = inputs[i];
    cache.insert(in.id());
  }

  // DFS the graph to build the tape, and log parents and scalars
  recurse = [&](const array& a) {
    auto id = a.id();
    if (cache.find(id) != cache.end()) {
      return;
    }
    for (int i = 0; i < a.inputs().size(); i++) {
      auto& in = a.inputs()[i];
      parents_map[in.id()].push_back({a, i});
      for (auto& s : a.siblings()) {
        parents_map[in.id()].push_back({s, i});
      }
      recurse(in);
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
  return {tape, parents_map};
}

// Simplify the tape. Note, this function modifies in-place both the tape and
// the parents map to remove orphaned arrays
void compile_simplify(
    std::vector<array>& tape,
    ParentsMap& parents_map,
    const std::vector<array>& outputs,
    int passes) {
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

  for (auto& a : tape) {
    if (is_scalar(a)) {
      scalars.insert({get_scalar_rep(a), a});
    }
  }

  // Helper that fuses two arrays in the graph by setting the parents of the
  // source to point to the destination
  auto fuse = [&](array& dst, array& src) {
    // Canonicalize the order of the primitives outputs
    auto sources = src.outputs();
    auto dests = dst.outputs();
    // For each src parent, point it to the corresponding dest
    for (int i = 0; i < sources.size(); ++i) {
      auto src_parents = parents_map.find(sources[i].id());
      if (src_parents == parents_map.end()) {
        continue;
      }
      auto& pairs = parents_map[dests[i].id()];
      for (auto& parent : src_parents->second) {
        parent.first.inputs()[parent.second] = dests[i];
        pairs.push_back(parent);
      }
      // Remove the source from the map to avoid fusing with it again
      parents_map.erase(src_parents);
    }
  };

  // Depth-1 array equivalence check.
  auto array_equivalent = [](const array& a, const array& b) {
    if (!a.has_primitive() || !b.has_primitive()) {
      return false;
    }
    if (a.primitive_id() == b.primitive_id()) {
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

  // Pass 0: fuse scalars
  std::vector<array> new_tape;
  for (auto& arr : tape) {
    // Check if we can fuse scalars
    if (is_scalar(arr)) {
      auto scalar = scalars.find(get_scalar_rep(arr));
      if (scalar->second.id() != arr.id()) {
        fuse(scalar->second, arr);
        // Don't keep orphaned scalars in the tape
        continue;
      }
    }
    new_tape.push_back(std::move(arr));
  }

  tape = std::move(new_tape);

  std::unordered_set<uintptr_t> output_set;
  for (auto& o : outputs) {
    output_set.insert(o.id());
  }
  // Pass 1..passes: fuse only keeping non-orphaned arrays in the tape
  for (int pass = 0; pass < passes; ++pass) {
    for (auto& arr : tape) {
      // Helper to check if we can fuse the parents of the
      // given array
      auto maybe_fuse_parents = [&](auto& a) {
        auto parents = parents_map.find(a.id());
        if (parents != parents_map.end()) {
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
              if (src.id() != dst.id() && array_equivalent(src, dst)) {
                fuse(dst, src);
                mask[j] = true;
              }
            }
          }
          return false;
        } else {
          return output_set.find(a.id()) == output_set.end();
        }
      };

      bool discard = maybe_fuse_parents(arr);
      for (auto& s : arr.siblings()) {
        discard &= maybe_fuse_parents(s);
      }
      // If an array and its siblings have no parents, and none of them are
      // outputs, it is safe to remove it from the tape
      if (!discard) {
        new_tape.push_back(std::move(arr));
      }
    }
    tape = std::move(new_tape);
  }
}

std::vector<array> compile_replace(
    const std::vector<array>& tape,
    const std::vector<array>& trace_inputs,
    const std::vector<array>& trace_outputs,
    const std::vector<array>& inputs) {
  std::unordered_map<uintptr_t, array> trace_to_real;
  for (int i = 0; i < inputs.size(); ++i) {
    trace_to_real.insert({trace_inputs[i].id(), inputs[i]});
  }

  for (auto& a : tape) {
    // Arrays in the tape without primitives are constants
    // and can be used directly
    if (!a.has_primitive()) {
      trace_to_real.insert({a.id(), a});
    } else {
      std::vector<array> real_inputs;
      for (auto& in : a.inputs()) {
        real_inputs.push_back(trace_to_real.at(in.id()));
      }
      auto real_a = array(
          a.shape(), a.dtype(), a.primitive_ptr(), std::move(real_inputs));
      trace_to_real.insert({a.id(), std::move(real_a)});
    }
  }

  std::vector<array> outputs;
  for (auto& o : trace_outputs) {
    outputs.push_back(trace_to_real.at(o.id()));
  }
  return outputs;
}

std::function<std::vector<array>(const std::vector<array>&)> compile(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun,
    size_t fun_id) {
  return [fun, fun_id](const std::vector<array>& inputs) {
    // Find a cache entry with the correct inputs
    auto& entry = compiler_cache().find(fun_id, inputs);

    // No matching cache entry existed, so compile
    if (entry.empty) {
      // Mark the entry as not empty since we are about to fill it
      entry.empty = false;
      // Trace to build the graph
      std::tie(entry.inputs, entry.outputs) = compile_trace(fun, inputs);

      // DFS the graph and get a tape, and a map of array id to (parent,
      // position in parent inputs)
      std::unordered_map<uintptr_t, std::vector<std::pair<array, int>>>
          parents_map;
      std::tie(entry.tape, parents_map) =
          compile_dfs(entry.inputs, entry.outputs);

      // Simplify the tape
      compile_simplify(entry.tape, parents_map, entry.outputs, /* passes */ 2);

      // This is a good point to do more optimizations, e.g. kernel fusion to
      // generate new primitives. The tape needs to be updated accordingly
    }

    // At this point we must have a tape, now replace the placeholders
    // with real arrays that can be evaluated
    return compile_replace(entry.tape, entry.inputs, entry.outputs, inputs);
  };
}

void compile_erase(size_t fun_id) {
  detail::compiler_cache().erase(fun_id);
}

} // namespace detail

std::function<std::vector<array>(const std::vector<array>&)> compile(
    const std::function<std::vector<array>(const std::vector<array>&)>& fun) {
  auto fun_id = detail::getAddress(fun);
  if (fun_id == 0) {
    throw std::invalid_argument(
        "[compile] Cannot compile a non-addressable function.");
  }
  return detail::compile(fun, fun_id);
}

} // namespace mlx::core
