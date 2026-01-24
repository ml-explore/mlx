// Copyright © 2023-2024 Apple Inc.
#include <cstdlib>
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "mlx/allocator.h"
#include "mlx/backend/common/compiled.h"
#include "mlx/compile.h"
#include "mlx/compile_impl.h"
#include "mlx/fast_primitives.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"
#include "mlx/utils.h"

namespace mlx::core {

constexpr int max_compile_depth = 11;
constexpr int max_compile_arrays = 24;

bool is_unary(const Primitive& p) {
  return (
      typeid(p) == typeid(Abs) || typeid(p) == typeid(ArcCos) ||
      typeid(p) == typeid(ArcCosh) || typeid(p) == typeid(ArcSin) ||
      typeid(p) == typeid(ArcSinh) || typeid(p) == typeid(ArcTan) ||
      typeid(p) == typeid(ArcTanh) || typeid(p) == typeid(AsType) ||
      typeid(p) == typeid(Ceil) || typeid(p) == typeid(Cos) ||
      typeid(p) == typeid(Conjugate) || typeid(p) == typeid(Cosh) ||
      typeid(p) == typeid(Remainder) || typeid(p) == typeid(Erf) ||
      typeid(p) == typeid(ErfInv) || typeid(p) == typeid(Exp) ||
      typeid(p) == typeid(Floor) || typeid(p) == typeid(Log) ||
      typeid(p) == typeid(Log1p) || typeid(p) == typeid(LogicalNot) ||
      typeid(p) == typeid(Negative) || typeid(p) == typeid(Round) ||
      typeid(p) == typeid(Sigmoid) || typeid(p) == typeid(Sign) ||
      typeid(p) == typeid(Sin) || typeid(p) == typeid(Sinh) ||
      typeid(p) == typeid(Square) || typeid(p) == typeid(Sqrt) ||
      typeid(p) == typeid(Tan) || typeid(p) == typeid(Tanh) ||
      typeid(p) == typeid(Expm1) || typeid(p) == typeid(Real) ||
      typeid(p) == typeid(Imag) || typeid(p) == typeid(BitwiseInvert));
}

bool is_binary(const Primitive& p) {
  return (
      typeid(p) == typeid(Add) || typeid(p) == typeid(Divide) ||
      typeid(p) == typeid(Equal) || typeid(p) == typeid(Greater) ||
      typeid(p) == typeid(GreaterEqual) || typeid(p) == typeid(Less) ||
      typeid(p) == typeid(LessEqual) || typeid(p) == typeid(LogicalNot) ||
      typeid(p) == typeid(LogicalAnd) || typeid(p) == typeid(LogicalOr) ||
      typeid(p) == typeid(LogAddExp) || typeid(p) == typeid(Maximum) ||
      typeid(p) == typeid(Minimum) || typeid(p) == typeid(Multiply) ||
      typeid(p) == typeid(NotEqual) || typeid(p) == typeid(Power) ||
      typeid(p) == typeid(Subtract) || typeid(p) == typeid(BitwiseBinary) ||
      typeid(p) == typeid(ArcTan2));
}

bool is_ternary(const Primitive& p) {
  return typeid(p) == typeid(Select);
}

bool is_broadcast(const Primitive& p) {
  return typeid(p) == typeid(Broadcast);
}

bool is_noop(const Primitive& p) {
  return typeid(p) == typeid(Copy) || typeid(p) == typeid(StopGradient);
}

bool is_reduction(const Primitive& p) {
  return typeid(p) == typeid(Reduce) || typeid(p) == typeid(ArgReduce);
}

bool is_fusable(const Primitive& p) {
  return is_unary(p) || is_binary(p) || is_ternary(p) || is_broadcast(p);
}

Compiled::Compiled(
    Stream stream,
    std::vector<array> inputs,
    std::vector<array> outputs,
    std::vector<array> tape,
    std::unordered_set<uintptr_t> constant_ids)
    : Primitive(stream),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)),
      tape_(std::move(tape)),
      constant_ids_(std::move(constant_ids)),
      is_constant_([this](size_t i) {
        return constant_ids_.find(inputs_[i].id()) != constant_ids_.end();
      }) {
  // Build the kernel name.
  NodeNamer namer;
  std::ostringstream os;
  std::ostringstream constant_hasher;

  std::unordered_set<uintptr_t> output_ids;
  for (auto& o : outputs_) {
    output_ids.insert(o.id());
  }

  // Fill the input names. This is not really necessary, I just like having A,
  // B, C, ... as the inputs.
  for (const auto& x : inputs_) {
    namer.get_name(x);
  }

  // The primitives describing the tape. For unary and binary primitives this
  // must be enough to describe the full computation.
  for (const auto& a : tape_) {
    // name and type of output
    os << namer.get_name(a) << kindof(a.dtype()) << a.itemsize();
    // whether or not it's an output
    if (output_ids.find(a.id()) != output_ids.end()) {
      os << "O";
    } else {
      os << "I";
    }
    // computation performed
    os << a.primitive().name();
    // name of inputs to the function
    for (auto& inp : a.inputs()) {
      os << namer.get_name(inp);
    }
  }
  os << "_";

  for (const auto& x : inputs_) {
    if (constant_ids_.find(x.id()) != constant_ids_.end()) {
      os << "C";
      print_constant(constant_hasher, x);
    } else {
      os << (is_scalar(x) ? "S" : "V");
    }
  }
  os << "_";
  for (const auto& x : inputs) {
    if (constant_ids.find(x.id()) != constant_ids.end()) {
      continue;
    }
    os << kindof(x.dtype()) << x.itemsize();
  }
  os << "_" << std::hash<std::string>{}(constant_hasher.str());

  kernel_lib_ = os.str();
}

std::vector<array> Compiled::vjp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&,
    const std::vector<array>&) {
  throw std::runtime_error("[Compiled] Cannot vjp primitive.");
}

std::vector<array> Compiled::jvp(
    const std::vector<array>&,
    const std::vector<array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[Compiled] Cannot jvp primitive.");
}

std::pair<std::vector<array>, std::vector<int>> Compiled::vmap(
    const std::vector<array>&,
    const std::vector<int>&) {
  throw std::runtime_error("[Compiled] Cannot vmap primitive.");
}

bool Compiled::is_equivalent(const Primitive& other) const {
  const Compiled& a_other = static_cast<const Compiled&>(other);
  return std::equal(
      tape_.begin(),
      tape_.end(),
      a_other.tape_.begin(),
      a_other.tape_.end(),
      [](const array& a1, const array& a2) {
        auto& p1 = a1.primitive();
        auto& p2 = a2.primitive();
        return typeid(p1) == typeid(p2) && p1.is_equivalent(p2);
      });
}

const char* Compiled::name() const {
  if (name_.empty()) {
    std::ostringstream os;
    os << "Compiled";
    for (auto& a : tape_) {
      os << a.primitive().name();
    }
    name_ = os.str();
  }
  return name_.c_str();
}

std::vector<Shape> Compiled::output_shapes(const std::vector<array>& inputs) {
  size_t nd = 0;
  for (auto& in : inputs) {
    nd = std::max(nd, in.ndim());
  }
  Shape out_shape(nd, 0);
  for (auto& in : inputs) {
    auto dd = nd - in.ndim();
    for (auto i = dd; i < nd; ++i) {
      out_shape[i] = std::max(out_shape[i], in.shape()[i - dd]);
    }
  }
  // All outputs have the same shape
  return std::vector<Shape>(outputs_.size(), out_shape);
}

namespace detail {

CompileMode& compile_mode() {
  auto get_val = []() {
    if (std::getenv("MLX_DISABLE_COMPILE")) {
      return CompileMode::disabled;
    } else {
      return CompileMode::enabled;
    }
  };
  static CompileMode compile_mode_ = get_val();
  return compile_mode_;
}

// Helper like below but only merges the two provided arrays. If the src has
// siblings then these won't be merged to the dst.
void merge_one(array& dst, array& src, ParentsMap& parents_map) {
  auto src_parents = parents_map.find(src.id());
  if (src_parents == parents_map.end()) {
    return;
  }
  auto& pairs = parents_map[dst.id()];
  for (auto& parent : src_parents->second) {
    parent.first.inputs()[parent.second] = dst;
    pairs.push_back(parent);
  }

  // If src is a parent of dst, remove it from dst's parents
  for (auto it = pairs.begin(); it != pairs.end();) {
    if (it->first.id() == src.id()) {
      it = pairs.erase(it);
    } else {
      it++;
    }
  }
  // Remove the source from the map to avoid fusing with it again
  parents_map.erase(src_parents);
}

// Helper that merges two arrays in the graph by setting the parents of the
// source to point to the destination. The arrays are assumed to be coming from
// equivalent primitives so their siblings are merged as well.
void merge(array& dst, array& src, ParentsMap& parents_map) {
  // Canonicalize the order of the primitives outputs
  auto sources = src.outputs();
  auto dests = dst.outputs();
  // For each src parent, point it to the corresponding dst
  for (int i = 0; i < sources.size(); ++i) {
    merge_one(dests[i], sources[i], parents_map);
  }
}

// Any parent in the divider will continue to refer to `x` but any parent not
// in the divider will refer to a copy of the operation.
array split_one(
    const array& x,
    ParentsMap& parents_map,
    const std::unordered_set<uintptr_t>& divider) {
  array y(x.shape(), x.dtype(), x.primitive_ptr(), x.inputs());

  auto& x_parents = parents_map[x.id()];
  auto& y_parents = parents_map[y.id()];

  for (auto it = x_parents.begin(); it != x_parents.end();) {
    if (divider.find(it->first.id()) != divider.end()) {
      it->first.inputs()[it->second] = y;
      y_parents.emplace_back(std::move(*it));
      it = x_parents.erase(it);
    } else {
      it++;
    }
  }

  return y;
}

template <typename T, typename... U>
std::uintptr_t get_function_address(const std::function<T(U...)>& fun) {
  using FunType = T (*)(U...);
  const FunType* fun_ptr = fun.template target<FunType>();
  if (fun_ptr == nullptr) {
    return 0;
  }
  return reinterpret_cast<std::uintptr_t>(*fun_ptr);
}

class CompilerCache {
 public:
  struct CacheEntry {
    CacheEntry(Stream stream, bool shapeless)
        : stream(stream), shapeless(shapeless) {};
    Stream stream;
    bool shapeless;
    std::vector<array> inputs;
    std::vector<array> outputs;
    std::vector<array> tape;
    bool empty{true};
    std::vector<uint64_t> constants;
    std::shared_ptr<void> extra;
  };

  // Returns a reference to a CacheEntry which can be updated
  // by the caller to avoid copying large tapes / inputs / outputs
  CacheEntry& find(
      std::uintptr_t fun_id,
      const std::vector<array>& inputs,
      bool shapeless,
      const std::vector<uint64_t>& constants) {
    // Find the cache entries for |fun_id|.
    std::vector<CacheEntry>& entries = cache_[fun_id];

    // Compare if 2 arrays have same shape and dtype.
    auto has_same_shape_and_dtype = [shapeless](
                                        const std::vector<array>& in1,
                                        const std::vector<array>& in2) {
      if (in1.size() != in2.size()) {
        return false;
      }
      for (size_t i = 0; i < in1.size(); ++i) {
        if (in1[i].ndim() != in2[i].ndim()) {
          return false;
        }
        if (!shapeless && in1[i].shape() != in2[i].shape()) {
          return false;
        }
        if (in1[i].dtype() != in2[i].dtype()) {
          return false;
        }
      }
      return true;
    };
    // Loop over entries and check:
    // - Default stream and device match the entry's default stream
    // - Inputs match i.e. shapes and types must be equal.
    auto stream = default_stream(default_device());
    for (CacheEntry& entry : entries) {
      // Check that the default stream and device match
      if (entry.stream != stream) {
        continue;
      }
      if (entry.shapeless != shapeless) {
        continue;
      }

      // Check the inputs match and return if so
      if (has_same_shape_and_dtype(inputs, entry.inputs) &&
          constants == entry.constants) {
        return entry;
      }
    }
    // Otherwise append a new cache entry
    entries.push_back(CacheEntry{stream, shapeless});
    return entries.back();
  }

  void erase(std::uintptr_t fun_id) {
    cache_.erase(fun_id);
  }

  void clear() {
    cache_.clear();
  }

 private:
  CompilerCache() {
    // Make sure the allocator is fully
    // initialized before the compiler cache
    allocator::allocator();
  }

  friend CompilerCache& compiler_cache();
  std::unordered_map<std::uintptr_t, std::vector<CacheEntry>> cache_;
};

CompilerCache& compiler_cache() {
  static CompilerCache compiler_cache_;
  return compiler_cache_;
}

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
        old_to_new.insert({out[i].id(), as[i]});
      }
      new_tape.push_back(as[arr.sibling_position()]);
    } else {
      auto a = array(
          arr.shape(), arr.dtype(), arr.primitive_ptr(), std::move(inputs));
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

static inline uint64_t splitmix64(uint64_t x) noexcept {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

struct VecU64Hash {
  size_t operator()(const std::vector<uint64_t>& s) const noexcept {
    uint64_t h =
        0x243f6a8885a308d3ull ^ (uint64_t)s.size() * 0x9e3779b97f4a7c15ull;
    for (uint64_t x : s) {
      h = splitmix64(x ^ splitmix64(h + 0x9e3779b97f4a7c15ull));
    }
    return (size_t)h;
  }
};

// Simplify the tape. Note, this function modifies in-place both the tape,
// the parents map to remove orphaned arrays, and potentially the outputs
void compile_simplify(
    std::vector<array>& tape,
    ParentsMap& parents_map,
    std::vector<array>& outputs,
    int passes) {
  // Helpers to identify identical scalars
  std::map<std::pair<uint64_t, Dtype::Val>, array> scalars;
  auto is_scalar = [](const array& a) {
    // Condition for when it's safe to read an array
    return a.is_available() && a.ndim() == 0;
  };
  auto get_scalar_rep = [](const array& a) {
    uint64_t v = 0;
    switch (a.dtype().size()) {
      case 1:
        v = *a.data<uint8_t>();
        break;
      case 2:
        v = *a.data<uint16_t>();
        break;
      case 4:
        v = *a.data<uint32_t>();
        break;
      case 8:
        v = *a.data<uint64_t>();
        break;
    }
    return std::make_pair(v, a.dtype().val());
  };

  for (auto& a : tape) {
    if (is_scalar(a)) {
      scalars.insert({get_scalar_rep(a), a});
    }
  }

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

  // Merge scalars
  std::vector<array> new_tape;
  for (auto& arr : tape) {
    // Check if we can merge scalars
    if (is_scalar(arr)) {
      auto scalar = scalars.find(get_scalar_rep(arr));
      if (scalar->second.id() != arr.id()) {
        merge(scalar->second, arr, parents_map);
        // Don't keep orphaned scalars in the tape
        continue;
      }
    }
    new_tape.push_back(std::move(arr));
  }
  tape = std::move(new_tape);

  // Remove no-ops
  {
    std::unordered_map<uintptr_t, array> output_map;
    for (auto& o : outputs) {
      output_map.insert({o.id(), o});
    }
    for (auto& arr : tape) {
      if (!arr.has_primitive() || !is_noop(arr.primitive())) {
        new_tape.push_back(std::move(arr));
        continue;
      }
      merge_one(arr.inputs()[0], arr, parents_map);
      if (auto it = output_map.find(arr.id()); it != output_map.end()) {
        it->second = arr.inputs()[0];
      }
    }
    tape = std::move(new_tape);
    for (auto& o : outputs) {
      o = output_map.at(o.id());
    }
  }

  std::unordered_map<std::uintptr_t, uint32_t> tape_order;
  for (uint32_t i = 0; i < tape.size(); ++i) {
    tape_order.insert({tape[i].id(), i});
  }

  std::unordered_set<uintptr_t> output_set;
  for (auto& o : outputs) {
    output_set.insert(o.id());
  }

  // Multi-pass merge only keeping non-orphaned arrays in the tape
  for (int pass = 0; pass < passes; ++pass) {
    for (auto& arr : tape) {
      // Helper to check if we can merge the parents of the
      // given array
      auto maybe_merge_parents = [&](auto& a) {
        auto parents = parents_map.find(a.id());
        if (parents != parents_map.end()) {
          auto N = parents->second.size();
          std::vector<bool> mask(N, false);

          auto try_merge = [&](int dst_idx, int src_idx) {
            if (tape_order[parents->second[src_idx].first.id()] <
                tape_order[parents->second[dst_idx].first.id()]) {
              std::swap(src_idx, dst_idx);
            }
            auto& src = parents->second[src_idx].first;
            auto& dst = parents->second[dst_idx].first;
            if (src.id() != dst.id() && array_equivalent(src, dst) &&
                output_set.find(src.id()) == output_set.end()) {
              merge(dst, src, parents_map);
              mask[src_idx] = true;
            }
          };

          if (N > 100) {
            std::unordered_map<
                std::vector<uint64_t>,
                std::vector<int>,
                VecU64Hash>
                dst_map;
            // Find possibly mergeable groups
            for (int i = 0; i < N; i++) {
              // Make the hash key
              std::vector<uint64_t> key;
              auto& curr = parents->second[i].first;
              key.reserve(curr.inputs().size() + 2);
              for (auto& in : curr.inputs()) {
                key.push_back(in.id());
              }
              auto& p = curr.primitive();
              key.push_back(curr.inputs().size());
              key.push_back(typeid(p).hash_code());
              auto it = dst_map.find(key);
              if (it == dst_map.end()) {
                bool _;
                std::tie(it, _) = dst_map.insert({key, std::vector<int>{}});
              }
              it->second.push_back(i);
            }
            for (auto& [_, group] : dst_map) {
              for (int i = 0; i < group.size(); ++i) {
                if (mask[group[i]]) {
                  continue;
                }
                for (int j = i + 1; j < group.size(); ++j) {
                  if (mask[group[j]]) {
                    continue;
                  }
                  try_merge(group[i], group[j]);
                }
              }
            }
          } else {
            for (int i = 0; i < N; ++i) {
              if (mask[i]) {
                continue;
              }
              for (int j = i + 1; j < N; ++j) {
                if (mask[j]) {
                  continue;
                }
                try_merge(i, j);
              }
            }
          }

          // Erase orphaned parents so we don't keep fusing with them
          for (int i = N - 1; i >= 0; --i) {
            if (mask[i]) {
              parents->second.erase(parents->second.begin() + i);
            }
          }
          return false;
        } else {
          return output_set.find(a.id()) == output_set.end();
        }
      };
      bool discard = maybe_merge_parents(arr);
      for (auto& s : arr.siblings()) {
        discard &= maybe_merge_parents(s);
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

// Extract sub-graphs of the graph that can be compiled
// and replace them with a Compiled Primitive.
void compile_fuse(
    std::vector<array>& tape,
    ParentsMap& parents_map,
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  // Track outputs to replace with new compiled outputs
  std::unordered_map<uintptr_t, array> output_map;
  for (auto& o : outputs) {
    output_map.insert({o.id(), o});
  }

  // Set of inputs to distinguish constants
  std::unordered_set<uintptr_t> input_ids;
  for (auto& in : inputs) {
    input_ids.insert(in.id());
  }

  // Go through the tape in reverse order and check for fusable sub-graphs
  std::vector<array> new_tape;
  std::unordered_set<uintptr_t> global_cache;
  for (int i = tape.size() - 1; i >= 0; --i) {
    auto& arr = tape[i];

    // Already compiled
    if (global_cache.find(arr.id()) != global_cache.end()) {
      continue;
    }

    // Two pass recursion:
    // First pass:
    //  - Collect all the primitives which we can fuse with
    //  - Keeps a cache of fusable primitives which may be added out of
    //    DAG order. We have to determine if all of a fused primitive's
    //    outputs are also in the fused section, and this may not be the
    //    case the first time we visit it.
    // Second pass:
    //  - Collect inputs to the new compiled primitive
    //  - Add fusable primitives to a tape in the correct order

    std::function<void(const array&, int, const Stream&, const Shape&)> recurse;
    std::unordered_set<uintptr_t> cache;
    std::unordered_set<uintptr_t> input_set;
    recurse = [&](const array& a,
                  int depth,
                  const Stream& s,
                  const Shape& shape) {
      if (cache.find(a.id()) != cache.end()) {
        return;
      }

      // Stop fusing if:
      // - Depth limit exceeded
      // - Constant input
      // - Stream mismatch
      // - Non fusable primitive
      // - Is global output but has a different shape
      if (depth >= max_compile_depth || !a.has_primitive() ||
          a.primitive().stream() != s || !is_fusable(a.primitive()) ||
          (output_map.find(a.id()) != output_map.end() && a.shape() != shape)) {
        // Possible input
        input_set.insert(a.id());
        return;
      }

      bool all_parents_in = true;
      if (depth > 0) {
        // Guaranteed to have a parent since nested in the
        // recursion.
        auto& parents = parents_map.at(a.id());
        for (auto& [p, idx] : parents) {
          auto in_cache = cache.find(p.id()) != cache.end();
          if (!in_cache) {
            all_parents_in = false;
            break;
          }
        }
      }

      // Arrays with a mix of parents outside the compilable section
      // are not fusable except for broadcast which we can split to avoid
      // stopping fusion
      if (!all_parents_in) {
        if (a.has_primitive() && is_broadcast(a.primitive())) {
          array b = split_one(a, parents_map, cache);
          recurse(b, depth, s, shape);
        } else {
          // Possible input
          input_set.insert(a.id());
        }
        return;
      }

      if (output_map.find(a.id()) != output_map.end()) {
        input_set.insert(a.id());
      } else {
        // Not an input anymore since fusing it
        input_set.erase(a.id());
      }
      if (input_set.size() >= max_compile_arrays) {
        return;
      }
      cache.insert({a.id()});

      for (auto& in : a.inputs()) {
        recurse(in, depth + 1, s, shape);
      }
    };

    // This will be the result of the fused operation so it needs
    //   a) to not be already computed ie have a primitive
    //   b) that primitive to not be a broadcast since it will unnecessarily
    //      cast to a contiguous array potentially blowing up memory
    if (arr.has_primitive() && !is_broadcast(arr.primitive())) {
      Stream s = arr.primitive().stream();
      recurse(arr, 0, s, arr.shape());
    }

    // Not worth fusing a single primitive
    if (cache.size() <= 1) {
      new_tape.push_back(arr);
      continue;
    }

    // Recurse a second time to build the tape in the right
    // order and collect the inputs
    input_set.clear();
    std::vector<array> inputs;
    std::vector<array> fused_tape;
    std::unordered_set<uintptr_t> tape_set;
    std::function<void(const array&)> recurse_tape;
    recurse_tape = [&](const array& a) {
      if (cache.find(a.id()) == cache.end()) {
        if (input_set.find(a.id()) == input_set.end()) {
          input_set.insert(a.id());
          inputs.push_back(a);
        }
        return;
      }
      if (tape_set.find(a.id()) != tape_set.end()) {
        return;
      }
      tape_set.insert(a.id());
      for (auto& in : a.inputs()) {
        recurse_tape(in);
      }
      fused_tape.push_back(a);
    };
    recurse_tape(arr);

    std::vector<array> old_outputs;
    // Add to global cache and add any global outputs to outputs
    // of new primitive
    for (int j = 0; j < fused_tape.size() - 1; ++j) {
      auto& f = fused_tape[j];
      if (output_map.find(f.id()) != output_map.end()) {
        old_outputs.push_back(f);
        // Parents are now siblings, update the parent map
        auto& pairs = parents_map[f.id()];
        pairs.erase(
            std::remove_if(
                pairs.begin(),
                pairs.end(),
                [&](auto& p) {
                  return cache.find(p.first.id()) != cache.end();
                }),
            pairs.end());
      } else {
        // Remove inner fused arrays parents from the parents map
        // to keep the parents map in a valid state
        parents_map.erase(f.id());
      }
      global_cache.insert({f.id()});
    }
    old_outputs.push_back(arr);

    std::vector<Shape> shapes;
    std::vector<Dtype> types;
    for (auto& o : old_outputs) {
      if (o.shape() != old_outputs.back().shape()) {
        throw std::runtime_error(
            "[compile] Compilation failed. Tried to fuse operations with different output shapes");
      }
      shapes.push_back(o.shape());
      types.push_back(o.dtype());
    }
    std::unordered_set<uintptr_t> constant_ids;
    for (auto& in : inputs) {
      // Scalar constant
      if (in.size() == 1 && !in.has_primitive() &&
          input_ids.find(in.id()) == input_ids.end()) {
        constant_ids.insert(in.id());
      }
    }
    auto compiled_outputs = array::make_arrays(
        std::move(shapes),
        types,
        std::make_shared<Compiled>(
            old_outputs.back().primitive().stream(),
            inputs,
            old_outputs,
            std::move(fused_tape),
            std::move(constant_ids)),
        inputs);

    // One output per primitive
    new_tape.push_back(compiled_outputs.back());

    // Replace inputs old parents with compiled_outputs
    for (int i = 0; i < inputs.size(); ++i) {
      auto& pairs = parents_map[inputs[i].id()];
      pairs.erase(
          std::remove_if(
              pairs.begin(),
              pairs.end(),
              [&](auto& p) { return cache.find(p.first.id()) != cache.end(); }),
          pairs.end());
      for (auto& o : compiled_outputs) {
        pairs.push_back({o, i});
      }
    }

    // - Update outputs parents to point to compiled outputs
    // - Update any overall graph outputs to be compiled outputs
    for (int o = 0; o < old_outputs.size(); ++o) {
      merge_one(compiled_outputs[o], old_outputs[o], parents_map);
      if (auto it = output_map.find(old_outputs[o].id());
          it != output_map.end()) {
        it->second = compiled_outputs[o];
      }
    }
  }

  std::reverse(new_tape.begin(), new_tape.end());
  tape = std::move(new_tape);

  // Replace output with potentially compiled output
  for (auto& o : outputs) {
    o = output_map.at(o.id());
  }
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

bool skip_compile() {
  return compile_mode() == CompileMode::disabled ||
      !(compile_available_for_device(default_device()));
}

ArrayFnWithExtra compile(
    ArrayFnWithExtra fun,
    std::uintptr_t fun_id,
    bool shapeless /* = false */,
    std::vector<uint64_t> constants /* = {} */) {
  if (skip_compile()) {
    return fun;
  }
  if (!fun) {
    throw std::invalid_argument(
        "[compile] Cannot compile a function without a target.");
  }

  return [fun = std::move(fun),
          fun_id,
          shapeless,
          constants = std::move(constants)](const std::vector<array>& inputs) {
    // If the inputs are tracers, trace the original graph
    if (std::any_of(inputs.begin(), inputs.end(), [](auto& in) {
          return in.is_tracer();
        })) {
      return fun(inputs);
    }

    // Find a cache entry with the correct inputs
    auto& entry = compiler_cache().find(fun_id, inputs, shapeless, constants);

    // No matching cache entry existed, so compile
    if (entry.empty) {
      // Mark the entry as not empty since we are about to fill it
      entry.empty = false;
      // Set the constants
      entry.constants = std::move(constants);
      // Trace to build the graph
      std::tie(entry.inputs, entry.outputs, entry.extra) =
          compile_trace(fun, inputs, shapeless);

      // DFS the graph and get a tape, and a map of array id to (parent,
      // position in parent inputs)
      std::unordered_map<uintptr_t, std::vector<std::pair<array, int>>>
          parents_map;
      std::tie(entry.tape, parents_map) =
          compile_dfs(entry.inputs, entry.outputs, inputs);

      // Simplify the tape
      if (compile_mode() != CompileMode::no_simplify) {
        compile_simplify(
            entry.tape, parents_map, entry.outputs, /* passes */ 3);
      }

      // Kernel fusion to generate Compiled primitives. The tape and
      // new outputs must be updated accordingly
      if (compile_mode() != CompileMode::no_fuse) {
        compile_fuse(entry.tape, parents_map, entry.inputs, entry.outputs);
      }
    }

    // At this point we must have a tape, now replace the placeholders
    // with real arrays that can be evaluated
    return ArraysAndExtra{
        compile_replace(
            entry.tape, entry.inputs, entry.outputs, inputs, shapeless),
        entry.extra};
  };
}

std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    std::uintptr_t fun_id,
    bool shapeless /* = false */,
    std::vector<uint64_t> constants /* = {} */) {
  if (skip_compile()) {
    return fun;
  }
  if (!fun) {
    throw std::invalid_argument(
        "[compile] Cannot compile a function without a target.");
  }

  ArrayFnWithExtra fun_with_extra =
      [fun = std::move(fun)](const std::vector<array>& inputs) {
        return ArraysAndExtra{fun(inputs), nullptr};
      };

  auto compiled_fun = compile(
      std::move(fun_with_extra), fun_id, shapeless, std::move(constants));

  return [compiled_fun =
              std::move(compiled_fun)](const std::vector<array>& inputs) {
    return compiled_fun(inputs).first;
  };
}

void compile_erase(std::uintptr_t fun_id) {
  detail::compiler_cache().erase(fun_id);
}

void compile_clear_cache() {
  detail::compiler_cache().clear();
}

} // namespace detail

std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    bool shapeless /* false */) {
  if (detail::skip_compile()) {
    return fun;
  }
  auto fun_id = detail::get_function_address(fun);
  if (fun_id) {
    // If the function has an addressable target then no need to manage it's
    // lifetime
    return detail::compile(std::move(fun), fun_id, shapeless);
  } else {
    auto pfun = std::shared_ptr<
        std::function<std::vector<array>(const std::vector<array>&)>>(
        new std::function<std::vector<array>(const std::vector<array>&)>{fun},
        [](auto* p) {
          detail::compile_erase(reinterpret_cast<std::uintptr_t>(p));
          delete p;
        });
    fun_id = reinterpret_cast<std::uintptr_t>(pfun.get());
    return detail::compile(
        [pfun = std::move(pfun)](const auto& inputs) {
          return (*pfun)(inputs);
        },
        fun_id,
        shapeless);
  }
}

std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::vector<array> (*fun)(const std::vector<array>&),
    bool shapeless /* = false */) {
  if (detail::skip_compile()) {
    return fun;
  }
  return detail::compile(fun, reinterpret_cast<std::uintptr_t>(fun), shapeless);
}

void disable_compile() {
  detail::compile_mode() = CompileMode::disabled;
}

void enable_compile() {
  detail::compile_mode() = CompileMode::enabled;
}

void set_compile_mode(CompileMode mode) {
  detail::compile_mode() = mode;
}

} // namespace mlx::core
