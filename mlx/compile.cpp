// Copyright © 2023-2024 Apple Inc.

#include <atomic>
#include <cstdlib>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <shared_mutex>
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

using ArrayFunction =
    std::function<std::vector<array>(const std::vector<array>&)>;

constexpr int max_compile_depth = 11;
constexpr int max_compile_arrays = 24;
constexpr int max_fuse_distance = 0;

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
  return typeid(p) == typeid(Add) || typeid(p) == typeid(ArcTan2) ||
      typeid(p) == typeid(BitwiseBinary) || typeid(p) == typeid(Divide) ||
      typeid(p) == typeid(Equal) || typeid(p) == typeid(FloorDivide) ||
      typeid(p) == typeid(Greater) || typeid(p) == typeid(GreaterEqual) ||
      typeid(p) == typeid(Less) || typeid(p) == typeid(LessEqual) ||
      typeid(p) == typeid(LogicalNot) || typeid(p) == typeid(LogicalAnd) ||
      typeid(p) == typeid(LogicalOr) || typeid(p) == typeid(LogAddExp) ||
      typeid(p) == typeid(Maximum) || typeid(p) == typeid(Minimum) ||
      typeid(p) == typeid(Multiply) || typeid(p) == typeid(NotEqual) ||
      typeid(p) == typeid(Power) || typeid(p) == typeid(Subtract);
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
  // Iterate the moved-into members; the parameters are moved-from above.
  for (const auto& x : inputs_) {
    if (constant_ids_.find(x.id()) != constant_ids_.end()) {
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

std::atomic<CompileMode>& compile_mode() {
  auto get_val = []() {
    if (std::getenv("MLX_DISABLE_COMPILE")) {
      return CompileMode::disabled;
    } else {
      return CompileMode::enabled;
    }
  };
  static std::atomic<CompileMode> compile_mode_ = get_val();
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

class CompileCache {
 public:
  struct CacheEntry {
    CacheEntry(std::optional<Stream> stream, bool shapeless)
        : stream(stream), shapeless(shapeless) {}
    std::optional<Stream> stream;
    bool shapeless;
    std::vector<array> inputs;
    std::vector<array> outputs;
    std::vector<array> tape;
    bool empty{true};
    std::vector<uint64_t> constants;
    std::shared_ptr<void> extra;
  };

  CompileCache() {
    // Make sure the allocator is fully initialized before the compiler cache.
    allocator::allocator();
  }

  // Returns a reference to a CacheEntry which can be updated by the caller to
  // avoid copying large tapes / inputs / outputs, with the shared_ptr of
  // entries to avoid getting erased during compilation.
  std::tuple<CacheEntry&, std::shared_ptr<std::vector<CacheEntry>>> find(
      std::uintptr_t fun_id,
      const std::vector<array>& inputs,
      bool shapeless,
      const std::vector<uint64_t>& constants) {
    // Find the cache entries for |fun_id| in a thread-safe way.
    auto entries_ptr = [&]() {
      // Lookup with shared lock.
      {
        std::shared_lock lock(mutex_);
        auto it = cache_.find(fun_id);
        if (it != cache_.end()) {
          return it->second;
        }
      }
      // Insertion with exclusive lock.
      std::unique_lock lock(mutex_);
      auto& ptr = cache_[fun_id];
      if (!ptr) {
        ptr = std::make_shared<std::vector<CacheEntry>>();
      }
      return ptr;
    }();
    auto& entries = *entries_ptr;

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
    auto stream = peek_default_stream(default_device());
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
        return {entry, std::move(entries_ptr)};
      }
    }
    // Otherwise append a new cache entry
    entries.push_back(CacheEntry{stream, shapeless});
    return {entries.back(), std::move(entries_ptr)};
  }

  void erase(std::uintptr_t fun_id) {
    std::unique_lock lock(mutex_);
    cache_.erase(fun_id);
  }

  void clear() {
    std::unique_lock lock(mutex_);
    cache_.clear();
  }

 private:
  // The cache may get its key erased from a separate thread, but its value is
  // only added and modified in the thread of creation.
  // Put value in a shared_ptr to avoid race condition when erasing happened
  // during compilation for the same function.
  std::unordered_map<std::uintptr_t, std::shared_ptr<std::vector<CacheEntry>>>
      cache_;
  std::shared_mutex mutex_;
};

std::shared_ptr<CompileCache>& compile_cache_unsafe() {
  static thread_local auto cache = std::make_shared<CompileCache>();
  return cache;
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

  std::unordered_map<uintptr_t, array> output_map;
  for (auto& o : outputs) {
    output_map.insert({o.id(), o});
  }

  // Merge scalars, except outputs
  std::vector<array> new_tape;
  for (auto& arr : tape) {
    // Check if we can merge scalars
    if (is_scalar(arr) && output_map.find(arr.id()) == output_map.end()) {
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
                output_map.find(src.id()) == output_map.end()) {
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
          return output_map.find(a.id()) == output_map.end();
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

// Max-flow solver (Dinic) for compile_recompute.
class MaxFlow {
 public:
  static constexpr int64_t inf = std::numeric_limits<int64_t>::max() / 4;

  explicit MaxFlow(int n) : adj_(n), level_(n), next_(n) {}

  void add_edge(int u, int v, int64_t cap) {
    adj_[u].push_back(edges_.size());
    edges_.push_back({v, cap});
    adj_[v].push_back(edges_.size());
    edges_.push_back({u, 0});
  }

  // Returns the max flow, or inf if there is no finite cut.
  int64_t solve(int s, int t) {
    int64_t flow = 0;
    while (make_levels(s, t)) {
      std::fill(next_.begin(), next_.end(), 0);
      while (int64_t f = augment(s, t)) {
        flow += f;
        if (flow >= inf) {
          return inf;
        }
      }
    }
    return flow;
  }

  // Vertices that s still reaches in the residual graph.
  std::vector<bool> source_side(int s) const {
    std::vector<bool> seen(adj_.size(), false);
    std::vector<int> stack = {s};
    seen[s] = true;
    while (!stack.empty()) {
      int u = stack.back();
      stack.pop_back();
      for (int e : adj_[u]) {
        int v = edges_[e].to;
        if (edges_[e].cap > 0 && !seen[v]) {
          seen[v] = true;
          stack.push_back(v);
        }
      }
    }
    return seen;
  }

 private:
  struct Edge {
    int to;
    int64_t cap;
  };

  bool make_levels(int s, int t) {
    std::fill(level_.begin(), level_.end(), -1);
    std::queue<int> queue;
    level_[s] = 0;
    queue.push(s);
    while (!queue.empty()) {
      int u = queue.front();
      queue.pop();
      for (int e : adj_[u]) {
        int v = edges_[e].to;
        if (edges_[e].cap > 0 && level_[v] < 0) {
          level_[v] = level_[u] + 1;
          queue.push(v);
        }
      }
    }
    return level_[t] >= 0;
  }

  // Push flow along one path of the level graph. A stack is used instead of
  // recursion because compiled graphs can be deep.
  int64_t augment(int s, int t) {
    std::vector<int> path;
    int u = s;
    while (u != t) {
      bool moved = false;
      for (; next_[u] < adj_[u].size(); ++next_[u]) {
        int e = adj_[u][next_[u]];
        int v = edges_[e].to;
        if (edges_[e].cap > 0 && level_[v] == level_[u] + 1) {
          path.push_back(e);
          u = v;
          moved = true;
          break;
        }
      }
      if (!moved) {
        if (path.empty()) {
          return 0;
        }
        // Dead end: remove u from the level graph and step back.
        level_[u] = -1;
        u = edges_[path.back() ^ 1].to;
        path.pop_back();
      }
    }
    int64_t f = inf;
    for (int e : path) {
      f = std::min(f, edges_[e].cap);
    }
    for (int e : path) {
      edges_[e].cap -= f;
      edges_[e ^ 1].cap += f;
    }
    return f;
  }

  std::vector<Edge> edges_;
  std::vector<std::vector<int>> adj_;
  std::vector<int> level_;
  std::vector<size_t> next_;
};

void compile_recompute(std::vector<array>& tape, ParentsMap& parents_map) {
  int n = tape.size();
  std::unordered_map<std::uintptr_t, int> vid;
  for (int i = 0; i < n; ++i) {
    vid.insert({tape[i].id(), i});
    for (auto& s : tape[i].siblings()) {
      vid.insert({s.id(), i});
    }
  }

  // Backward nodes are made by autodiff, or computed from such nodes, like an
  // optimizer update.
  std::vector<bool> backward(n, false);
  bool has_backward = false;
  for (int i = 0; i < n; ++i) {
    auto& a = tape[i];
    if (!a.has_primitive()) {
      continue;
    }
    bool b = a.is_backward();
    for (auto& in : a.inputs()) {
      b = b || backward[vid.at(in.id())];
    }
    backward[i] = b;
    has_backward |= b;
  }
  if (!has_backward) {
    return;
  }

  auto readers = [&](int i) {
    std::vector<int> rs;
    for (auto& o : tape[i].outputs()) {
      if (auto it = parents_map.find(o.id()); it != parents_map.end()) {
        for (auto& [p, _] : it->second) {
          if (auto r = vid.find(p.id()); r != vid.end()) {
            rs.push_back(r->second);
          }
        }
      }
    }
    return rs;
  };
  auto fusable = [&](int i) {
    return tape[i].has_primitive() && is_fusable(tape[i].primitive());
  };
  auto is_view = [&](int i) {
    if (!tape[i].has_primitive()) {
      return false;
    }
    auto& p = tape[i].primitive();
    return typeid(p) == typeid(Broadcast) || typeid(p) == typeid(Reshape) ||
        typeid(p) == typeid(Transpose) || typeid(p) == typeid(ExpandDims) ||
        typeid(p) == typeid(Squeeze);
  };

  // True if the backward reads node i, directly or through views
  std::function<bool(int)> backward_reads;
  backward_reads = [&](int i) -> bool {
    for (int r : readers(i)) {
      if (backward[r] || (is_view(r) && backward_reads(r))) {
        return true;
      }
    }
    return false;
  };

  // True if the backward reads node i only with fusable ops and not through a
  // view. After a transpose or a reshape the copy does not fuse, for example a
  // weight cast read by a matmul. After a broadcast the copy runs once for each
  // element, for example a scalar learning rate in each optimizer update.
  auto fuses_into_backward = [&](int i) {
    for (int r : readers(i)) {
      if (is_view(r) ? (backward[r] || backward_reads(r))
                     : (backward[r] && !fusable(r))) {
        return false;
      }
    }
    return true;
  };

  MaxFlow graph(2 * n + 2);
  int source = 2 * n;
  int sink = 2 * n + 1;
  for (int i = 0; i < n; ++i) {
    auto& a = tape[i];
    int in = 2 * i;
    int out = 2 * i + 1;
    if (backward[i]) {
      graph.add_edge(in, sink, MaxFlow::inf);
      continue;
    }
    auto rs = readers(i);
    for (int r : rs) {
      graph.add_edge(out, 2 * r, MaxFlow::inf);
    }
    if (!a.has_primitive()) {
      // Inputs and constants stay alive anyway, so keeping them is free.
      graph.add_edge(source, in, MaxFlow::inf);
      graph.add_edge(in, out, 0);
      continue;
    }
    if (is_view(i)) {
      // Views are free to recompute, so they are never kept.
      graph.add_edge(in, out, MaxFlow::inf);
      continue;
    }

    // Keeping a value costs its size. The cost doubles when a fused kernel
    // would otherwise keep the value in registers.
    bool in_registers =
        fusable(i) && !rs.empty() && std::all_of(rs.begin(), rs.end(), fusable);
    int64_t bytes = 0;
    for (auto& o : a.outputs()) {
      bytes += o.nbytes();
    }
    graph.add_edge(in, out, in_registers ? 2 * bytes : bytes);

    // Only cheap elementwise ops are recomputed, and only when the backward
    // can fuse them. Other ops must run in the forward.
    bool recompute =
        fusable(i) && a.siblings().empty() && fuses_into_backward(i);
    if (!recompute) {
      graph.add_edge(source, in, MaxFlow::inf);
    }
  }

  auto kept_bytes = graph.solve(source, sink);
  if (kept_bytes >= MaxFlow::inf) {
    return;
  }
  auto source_side = graph.source_side(source);

  // The value the backward reads for `x`: x itself if it is kept, otherwise a
  // copy computed from kept values.
  std::unordered_map<int, array> copies;
  std::vector<array> new_nodes;
  std::function<array(const array&)> backward_value;
  backward_value = [&](const array& x) -> array {
    int j = vid.at(x.id());
    if (backward[j] || source_side[2 * j]) {
      return x;
    }
    if (auto it = copies.find(j); it != copies.end()) {
      return it->second;
    }
    auto& a = tape[j];
    std::vector<array> inputs;
    for (auto& in : a.inputs()) {
      inputs.push_back(backward_value(in));
    }
    array c(a.shape(), a.dtype(), a.primitive_ptr(), std::move(inputs));
    c.set_backward(true);
    for (int k = 0; k < c.inputs().size(); ++k) {
      parents_map[c.inputs()[k].id()].push_back({c, k});
    }
    copies.insert({j, c});
    new_nodes.push_back(c);
    return c;
  };

  // Point the backward readers at the copies.
  for (int i = 0; i < n; ++i) {
    if (!backward[i]) {
      continue;
    }
    // Give each backward node its own copies, so each kernel can fuse them.
    copies.clear();
    for (int k = 0; k < tape[i].inputs().size(); ++k) {
      auto x = tape[i].inputs()[k];
      auto c = backward_value(x);
      if (c.id() == x.id()) {
        continue;
      }
      auto& from = parents_map[x.id()];
      auto& to = parents_map[c.id()];
      for (auto it = from.begin(); it != from.end();) {
        if (it->second == k && vid.at(it->first.id()) == i) {
          it->first.inputs()[k] = c;
          to.push_back(std::move(*it));
          it = from.erase(it);
        } else {
          ++it;
        }
      }
    }
  }

  if (new_nodes.empty()) {
    return;
  }

  // Put each copy in the tape before the first node that reads it.
  std::unordered_map<std::uintptr_t, array> copy_of_id;
  for (auto& c : new_nodes) {
    copy_of_id.insert({c.id(), c});
  }
  std::vector<array> new_tape;
  new_tape.reserve(tape.size() + new_nodes.size());
  std::unordered_set<std::uintptr_t> placed;
  std::function<void(const array&)> place_inputs;
  place_inputs = [&](const array& a) {
    for (auto& in : a.inputs()) {
      auto it = copy_of_id.find(in.id());
      if (it != copy_of_id.end() && placed.insert(in.id()).second) {
        place_inputs(it->second);
        new_tape.push_back(it->second);
      }
    }
  };
  for (auto& a : tape) {
    place_inputs(a);
    new_tape.push_back(a);
  }
  tape = std::move(new_tape);
}

// Extract sub-graphs of the graph that can be compiled
// and replace them with a Compiled Primitive.
// Replace the arrays in `fused_tape` with one Compiled primitive. The arrays in
// `old_outputs` become the outputs of the primitive.
std::vector<array> make_compiled(
    std::vector<array> fused_tape,
    const std::vector<array>& inputs,
    std::vector<array> old_outputs,
    const std::unordered_set<uintptr_t>& input_ids,
    std::unordered_map<uintptr_t, array>& output_map,
    ParentsMap& parents_map) {
  std::unordered_set<uintptr_t> fused_ids;
  for (auto& f : fused_tape) {
    fused_ids.insert(f.id());
  }
  std::unordered_set<uintptr_t> output_ids;
  for (auto& o : old_outputs) {
    output_ids.insert(o.id());
  }

  for (auto& f : fused_tape) {
    if (output_ids.find(f.id()) != output_ids.end()) {
      // Parents are now siblings, update the parent map
      auto& pairs = parents_map[f.id()];
      pairs.erase(
          std::remove_if(
              pairs.begin(),
              pairs.end(),
              [&](auto& p) {
                return fused_ids.find(p.first.id()) != fused_ids.end();
              }),
          pairs.end());
    } else {
      // Remove inner fused arrays parents from the parents map
      // to keep the parents map in a valid state
      parents_map.erase(f.id());
    }
  }

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

  // Replace inputs old parents with compiled_outputs
  for (int i = 0; i < inputs.size(); ++i) {
    auto& pairs = parents_map[inputs[i].id()];
    pairs.erase(
        std::remove_if(
            pairs.begin(),
            pairs.end(),
            [&](auto& p) {
              return fused_ids.find(p.first.id()) != fused_ids.end();
            }),
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
  return compiled_outputs;
}

// Groups of arrays to fuse. Each group with more than one array becomes one
// Compiled primitive.
struct FuseGroups {
  std::vector<std::vector<array>> members;
  std::unordered_map<uintptr_t, int> group_of;

  // The group of `a`, or -1 if `a` is not fusable
  int find(const array& a) const {
    auto it = group_of.find(a.id());
    return (it == group_of.end()) ? -1 : it->second;
  }

  void add(const array& a, int g) {
    group_of[a.id()] = g;
    members[g].push_back(a);
  }

  void make_group(const array& a) {
    members.emplace_back();
    add(a, members.size() - 1);
  }

  // Move the arrays of group `src` to group `dst`
  void merge(int dst, int src) {
    for (auto& a : members[src]) {
      add(a, dst);
    }
    members[src].clear();
  }

  // Undo merge(dst, src), where `n` is the size of `dst` before the merge
  void unmerge(int dst, int src, int n) {
    for (int i = n; i < members[dst].size(); ++i) {
      add(members[dst][i], src);
    }
    members[dst].erase(members[dst].begin() + n, members[dst].end());
  }
};

// True if `a` is read outside of its group or is an output of the graph
bool is_group_output(
    const array& a,
    const FuseGroups& groups,
    const ParentsMap& parents_map,
    const std::unordered_map<uintptr_t, array>& output_map) {
  if (output_map.find(a.id()) != output_map.end()) {
    return true;
  }
  int g = groups.find(a);
  if (auto it = parents_map.find(a.id()); it != parents_map.end()) {
    for (auto& [p, idx] : it->second) {
      if (groups.find(p) != g) {
        return true;
      }
    }
  }
  return false;
}

// True if group `g` can become one Compiled primitive
bool is_valid_group(
    int g,
    const FuseGroups& groups,
    const ParentsMap& parents_map,
    const std::unordered_map<uintptr_t, array>& output_map) {
  // All outputs have the same shape. Do not write out a broadcast.
  std::vector<array> outputs;
  for (auto& a : groups.members[g]) {
    if (!is_group_output(a, groups, parents_map, output_map)) {
      continue;
    }
    if (is_broadcast(a.primitive()) &&
        output_map.find(a.id()) == output_map.end()) {
      return false;
    }
    if (!outputs.empty() && a.shape() != outputs[0].shape()) {
      return false;
    }
    outputs.push_back(a);
  }

  // Count the inputs and find the longest chain of arrays in the group
  std::unordered_set<uintptr_t> inputs;
  std::unordered_map<uintptr_t, int> depths;
  std::function<int(const array&)> recurse;
  recurse = [&](const array& a) -> int {
    if (groups.find(a) != g) {
      inputs.insert(a.id());
      return 0;
    }
    if (auto it = depths.find(a.id()); it != depths.end()) {
      return it->second;
    }
    int depth = 0;
    for (auto& in : a.inputs()) {
      depth = std::max(depth, recurse(in));
    }
    depths.insert({a.id(), depth + 1});
    return depth + 1;
  };
  int depth = 0;
  for (auto& a : groups.members[g]) {
    depth = std::max(depth, recurse(a));
  }
  return depth <= max_compile_depth &&
      inputs.size() + outputs.size() <= max_compile_arrays;
}

// Give group `g` its own copy of the broadcast `b` with split_one. Return
// true if the copy is fused into the group.
bool split_broadcast(
    array b,
    int g,
    FuseGroups& groups,
    ParentsMap& parents_map,
    const std::unordered_map<uintptr_t, array>& output_map) {
  std::unordered_set<uintptr_t> divider;
  for (auto& a : groups.members[g]) {
    divider.insert(a.id());
  }
  array c = split_one(b, parents_map, divider);
  for (int i = 0; i < c.inputs().size(); ++i) {
    parents_map[c.inputs()[i].id()].push_back({c, i});
  }
  groups.add(c, g);
  if (is_valid_group(g, groups, parents_map, output_map)) {
    return true;
  }

  // Undo the copy
  groups.members[g].pop_back();
  groups.group_of.erase(c.id());
  for (auto& in : c.inputs()) {
    auto& pairs = parents_map[in.id()];
    pairs.erase(
        std::remove_if(
            pairs.begin(),
            pairs.end(),
            [&](auto& p) { return p.first.id() == c.id(); }),
        pairs.end());
  }
  merge_one(b, c, parents_map);
  return false;
}

// Fuse a group into the group that reads it, if all the readers of the group
// are in that group. Return true if a group changed.
bool compile_fuse_vertical(
    FuseGroups& groups,
    const std::vector<array>& tape,
    ParentsMap& parents_map,
    const std::unordered_map<uintptr_t, array>& output_map) {
  bool changed = false;

  // Go through the tape in reverse order so readers are fused first
  for (int i = tape.size() - 1; i >= 0; --i) {
    auto& a = tape[i];
    int src = groups.find(a);
    if (src < 0) {
      continue;
    }

    // Find the groups that read the group of `a`
    std::vector<int> readers;
    bool all_fusable = true;
    for (auto& m : groups.members[src]) {
      auto it = parents_map.find(m.id());
      if (it == parents_map.end()) {
        continue;
      }
      for (auto& [p, idx] : it->second) {
        int dst = groups.find(p);
        if (dst < 0) {
          all_fusable = false;
        } else if (
            dst != src &&
            std::find(readers.begin(), readers.end(), dst) == readers.end()) {
          readers.push_back(dst);
        }
      }
    }

    // Split a broadcast with readers in more than one place, so that each
    // group can fuse its own copy
    if (is_broadcast(a.primitive()) && groups.members[src].size() == 1 &&
        (readers.size() > 1 || (readers.size() == 1 && !all_fusable))) {
      for (int dst : readers) {
        changed |= split_broadcast(a, dst, groups, parents_map, output_map);
      }
      continue;
    }

    if (!all_fusable || readers.size() != 1) {
      continue;
    }
    int dst = readers[0];
    if (a.primitive().stream() != groups.members[dst][0].primitive().stream()) {
      continue;
    }
    int n = groups.members[dst].size();
    groups.merge(dst, src);
    if (is_valid_group(dst, groups, parents_map, output_map)) {
      changed = true;
    } else {
      groups.unmerge(dst, src, n);
    }
  }
  return changed;
}

// True if a path through arrays outside of groups `g1` and `g2` connects them.
// The two groups cannot be one kernel then.
bool is_connected(
    int g1,
    int g2,
    const FuseGroups& groups,
    const std::unordered_map<uintptr_t, int>& position,
    const ParentsMap& parents_map) {
  auto in_groups = [&](const array& a) {
    int g = groups.find(a);
    return g == g1 || g == g2;
  };

  // Readers come after their inputs in the tape, so stop the search after the
  // last array of the two groups
  int last = 0;
  for (int g : {g1, g2}) {
    for (auto& a : groups.members[g]) {
      if (auto it = position.find(a.id()); it != position.end()) {
        last = std::max(last, it->second);
      }
    }
  }

  // Start from the readers outside of the two groups
  std::vector<array> stack;
  for (int g : {g1, g2}) {
    for (auto& a : groups.members[g]) {
      if (auto it = parents_map.find(a.id()); it != parents_map.end()) {
        for (auto& [p, idx] : it->second) {
          if (!in_groups(p)) {
            stack.push_back(p);
          }
        }
      }
    }
  }

  std::unordered_set<uintptr_t> visited;
  while (!stack.empty()) {
    auto a = std::move(stack.back());
    stack.pop_back();
    if (in_groups(a)) {
      return true;
    }
    if (!visited.insert(a.id()).second) {
      continue;
    }
    if (auto it = position.find(a.id());
        it != position.end() && it->second > last) {
      continue;
    }
    if (auto it = parents_map.find(a.id()); it != parents_map.end()) {
      for (auto& [p, idx] : it->second) {
        stack.push_back(p);
      }
    }
  }
  return false;
}

// Fuse two groups which read the same arrays, if no path through other arrays
// connects them and they run at about the same time in the eval order. Return
// true if a group changed.
bool compile_fuse_horizontal(
    FuseGroups& groups,
    const std::vector<array>& tape,
    const std::unordered_map<uintptr_t, int>& position,
    ParentsMap& parents_map,
    const std::unordered_map<uintptr_t, array>& output_map) {
  // Score each pair of groups with the bytes of the arrays they both read
  std::map<std::pair<int, int>, size_t> scores;
  for (auto& x : tape) {
    // Scalars become constants in the kernel, so they are not read
    if (x.size() == 1) {
      continue;
    }
    auto it = parents_map.find(x.id());
    if (it == parents_map.end()) {
      continue;
    }
    std::vector<int> readers;
    for (auto& [p, idx] : it->second) {
      int g = groups.find(p);
      if (g >= 0 && g != groups.find(x) &&
          std::find(readers.begin(), readers.end(), g) == readers.end()) {
        readers.push_back(g);
      }
    }
    for (int i = 0; i < readers.size(); ++i) {
      for (int j = i + 1; j < readers.size(); ++j) {
        auto key = std::minmax(readers[i], readers[j]);
        scores[{key.first, key.second}] += x.nbytes();
      }
    }
  }

  // Eval runs an array as late as possible, just before its readers. So the
  // level of an array, its longest path to an output, sets when it runs.
  std::unordered_map<uintptr_t, int> levels;
  std::function<int(const array&)> level;
  level = [&](const array& a) -> int {
    if (auto it = levels.find(a.id()); it != levels.end()) {
      return it->second;
    }
    int l = 0;
    if (auto it = parents_map.find(a.id()); it != parents_map.end()) {
      for (auto& [p, idx] : it->second) {
        l = std::max(l, level(p) + 1);
      }
    }
    levels.insert({a.id(), l});
    return l;
  };

  // A group runs at the level of its deepest output
  auto group_level = [&](int g) {
    int l = 0;
    for (auto& a : groups.members[g]) {
      if (is_group_output(a, groups, parents_map, output_map)) {
        l = std::max(l, level(a));
      }
    }
    return l;
  };
  auto distance = [&](int g1, int g2) {
    return std::abs(group_level(g1) - group_level(g2));
  };
  auto is_used = [&](int g) {
    for (auto& a : groups.members[g]) {
      if (is_group_output(a, groups, parents_map, output_map)) {
        return true;
      }
    }
    return false;
  };

  // Try the pairs with the most shared bytes first, then the closest pairs.
  // Keep one array of each group, since groups change when they merge.
  struct Candidate {
    size_t score;
    int distance;
    array a;
    array b;
  };
  std::vector<Candidate> candidates;
  for (auto& [key, score] : scores) {
    auto [g1, g2] = key;
    candidates.push_back(
        {score,
         distance(g1, g2),
         groups.members[g1][0],
         groups.members[g2][0]});
  }
  std::sort(candidates.begin(), candidates.end(), [](auto& x, auto& y) {
    return (x.score != y.score) ? x.score > y.score : x.distance < y.distance;
  });

  // Merging groups at different levels moves a kernel in the eval order, and
  // this can keep large arrays longer in memory
  bool changed = false;
  for (auto& c : candidates) {
    int g1 = groups.find(c.a);
    int g2 = groups.find(c.b);
    if (g1 == g2 || c.a.primitive().stream() != c.b.primitive().stream() ||
        distance(g1, g2) > max_fuse_distance || !is_used(g1) || !is_used(g2) ||
        is_connected(g1, g2, groups, position, parents_map)) {
      continue;
    }
    int size = groups.members[g1].size();
    groups.merge(g1, g2);
    if (is_valid_group(g1, groups, parents_map, output_map)) {
      changed = true;
    } else {
      groups.unmerge(g1, g2, size);
    }
  }
  return changed;
}

// Fuse arrays in groups with the fusion rules, then replace each group with
// one Compiled primitive.
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

  // Start with one group for each fusable array
  FuseGroups groups;
  std::unordered_map<uintptr_t, int> position;
  for (int i = 0; i < tape.size(); ++i) {
    auto& a = tape[i];
    position.insert({a.id(), i});
    if (a.has_primitive() && is_fusable(a.primitive()) &&
        input_ids.find(a.id()) == input_ids.end()) {
      groups.make_group(a);
    }
  }

  // Merge groups until no rule changes them. A merge can make a new merge
  // possible, so repeat the rules.
  while (compile_fuse_vertical(groups, tape, parents_map, output_map) ||
         compile_fuse_horizontal(
             groups, tape, position, parents_map, output_map)) {
  }

  // Replace each group of more than one array with a Compiled primitive
  for (int g = 0; g < groups.members.size(); ++g) {
    auto& group = groups.members[g];
    if (group.size() <= 1) {
      continue;
    }

    // The roots are the arrays which are not read in the group
    std::vector<array> roots;
    for (auto& a : group) {
      bool is_root = true;
      if (auto it = parents_map.find(a.id()); it != parents_map.end()) {
        for (auto& [p, idx] : it->second) {
          if (groups.find(p) == g) {
            is_root = false;
            break;
          }
        }
      }
      if (is_root) {
        roots.push_back(a);
      }
    }
    std::sort(roots.begin(), roots.end(), [&](auto& a, auto& b) {
      return position.at(a.id()) < position.at(b.id());
    });

    // Build the tape of the group in the right order and collect the inputs
    std::vector<array> inputs;
    std::vector<array> fused_tape;
    std::unordered_set<uintptr_t> input_set;
    std::unordered_set<uintptr_t> tape_set;
    std::function<void(const array&)> recurse_tape;
    recurse_tape = [&](const array& a) {
      if (groups.find(a) != g) {
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
    for (auto& r : roots) {
      recurse_tape(r);
    }

    std::vector<array> old_outputs;
    for (auto& f : fused_tape) {
      if (is_group_output(f, groups, parents_map, output_map)) {
        old_outputs.push_back(f);
      }
    }

    // Nothing reads the group, so it is not used
    if (old_outputs.empty()) {
      continue;
    }
    make_compiled(
        std::move(fused_tape),
        inputs,
        std::move(old_outputs),
        input_ids,
        output_map,
        parents_map);
  }

  // Build the new tape in execution order
  std::vector<array> new_tape;
  std::unordered_set<uintptr_t> cache;
  std::function<void(const array&)> recurse;
  recurse = [&](const array& a) {
    if (cache.find(a.id()) != cache.end()) {
      return;
    }
    // Don't recurse on inputs
    if (input_ids.find(a.id()) == input_ids.end()) {
      for (auto& in : a.inputs()) {
        recurse(in);
      }
    }
    cache.insert(a.id());
    for (auto& s : a.siblings()) {
      cache.insert(s.id());
    }
    new_tape.push_back(a);
  };
  for (auto& o : outputs) {
    o = output_map.at(o.id());
    recurse(o);
  }
  tape = std::move(new_tape);
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
    auto [entry, entries_ptr] =
        compile_cache_unsafe()->find(fun_id, inputs, shapeless, constants);
    static_assert(std::is_reference_v<decltype(entry)>);

    // No matching cache entry existed, so compile
    if (entry.empty) {
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
      auto mode = compile_mode().load();
      if (mode != CompileMode::no_simplify) {
        compile_simplify(
            entry.tape, parents_map, entry.outputs, /* passes */ 3);
      }

      // Experimental: recompute cheap forward values in the backward
      // instead of keeping them. Enable with MLX_COMPILE_RECOMPUTE=1.
      if (config::get("MLX_COMPILE_RECOMPUTE", 0)) {
        compile_recompute(entry.tape, parents_map);
      }

      // Kernel fusion to generate Compiled primitives. The tape and
      // new outputs must be updated accordingly
      if (mode != CompileMode::no_fuse) {
        compile_fuse(entry.tape, parents_map, entry.inputs, entry.outputs);
      }

      // Mark the entry as filled only after every step above completed, so
      // a throwing first trace leaves the entry empty and a later call
      // re-traces cleanly instead of hitting a half-filled entry
      entry.empty = false;
    }

    // At this point we must have a tape, now replace the placeholders
    // with real arrays that can be evaluated
    return ArraysAndExtra{
        compile_replace(
            entry.tape, entry.inputs, entry.outputs, inputs, shapeless),
        entry.extra};
  };
}

ArrayFunction compile(
    ArrayFunction fun,
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

CompileCacheWeakPtr compile_cache() {
  return compile_cache_unsafe();
}

void compile_erase(const CompileCacheWeakPtr& cache, std::uintptr_t fun_id) {
  if (auto p = cache.lock()) {
    p->erase(fun_id);
  }
}

void compile_clear_cache(const CompileCacheWeakPtr& cache) {
  if (auto p = cache.lock()) {
    p->clear();
  }
}

} // namespace detail

ArrayFunction compile(ArrayFunction fun, bool shapeless /* false */) {
  if (detail::skip_compile()) {
    return fun;
  }
  auto fun_id = detail::get_function_address(fun);
  if (fun_id) {
    // If the function has an addressable target then no need to manage it's
    // lifetime
    return detail::compile(std::move(fun), fun_id, shapeless);
  } else {
    auto cleanup = [cache = detail::compile_cache()](auto* p) {
      detail::compile_erase(cache, reinterpret_cast<std::uintptr_t>(p));
      delete p;
    };
    auto pfun = std::shared_ptr<ArrayFunction>(new ArrayFunction{fun}, cleanup);
    fun_id = reinterpret_cast<std::uintptr_t>(pfun.get());
    return detail::compile(
        [pfun = std::move(pfun)](const auto& inputs) {
          return (*pfun)(inputs);
        },
        fun_id,
        shapeless);
  }
}

ArrayFunction compile(
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
