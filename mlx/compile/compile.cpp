// Copyright © 2023-2024 Apple Inc.

#include <atomic>
#include <cstdlib>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "mlx/backend/common/compiled.h"
#include "mlx/compile.h"
#include "mlx/compile_impl.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"
#include "mlx/utils.h"

namespace mlx::core {

using ArrayFunction =
    std::function<std::vector<array>(const std::vector<array>&)>;

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
