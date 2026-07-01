# Dynamic Backend Loading for MLX

## Motivation

Today MLX compiles backends statically via CMake flags (`MLX_BUILD_METAL`,
`MLX_BUILD_CPU`, `MLX_BUILD_CUDA`).  Each build of `libmlx` is locked to the
exact set of backends chosen at compile time.  As new hardware targets emerge
— SIMD-tuned x86 CPU variants (SSE, AVX2, AVX-512), multiple CUDA Compute
Capabilities, vendor-specific accelerators — it becomes impractical to ship
a single binary that covers all targets.

Dynamic backend loading solves this by:

1. Letting a single `libmlx` load backend shared libraries at runtime.
2. Enabling independent versioning and distribution of backends.
3. Allowing the runtime (or the application) to select the best backend for
   the hardware it discovers at startup.
4. Supporting multiple GPU backends simultaneously (e.g. CUDA alongside
   future backends for non-NVIDIA GPUs).
5. Enabling third-party/out-of-tree backends without rebuilding MLX.
6. Decouples transitive dependencies from the primary application linked
   to libmlx which allows startup and graceful detection of missing
   dependencies (e.g., CUDA)

---

## Design

### 1. Backend Plugin Interface

Each backend shared library implements a C++ abstract interface and exports
C entry points for `dlopen`-based loading.

#### 1.1 `mlx/backend_interface.h` — The Backend Contract

```cpp
#pragma once

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"
#include "mlx/allocator.h"

// Incremented when the interface changes in an ABI-incompatible way.
#define MLX_BACKEND_API_VERSION 1

namespace mlx::core {

struct BackendInfo {
  std::string name;                   // e.g. "metal", "cuda", "vulkan"
  Device::DeviceType device_type;     // cpu or gpu
  int score = 0;                      // Higher is better; 0 = unsupported on this hardware
  int device_count = 1;

  using PropertyMap =
      std::unordered_map<std::string, std::variant<std::string, size_t>>;
  std::vector<PropertyMap> device_properties;  // Per-device properties
};

class BackendInterface {
 public:
  virtual ~BackendInterface() = default;

  virtual int api_version() const = 0;
  virtual BackendInfo info() const = 0;

  // Stream lifecycle
  virtual void new_stream(Stream stream) = 0;
  virtual void synchronize(Stream stream) = 0;
  virtual void finalize(Stream stream) = 0;

  // Evaluation
  virtual void eval(array& arr) = 0;

  // Memory allocator
  virtual allocator::Allocator& allocator(int device_index = 0) = 0;

  // Interpret opaque buffer pointer for this backend's allocator layout.
  // E.g. Metal returns MTL::Buffer->contents(), CPU returns ptr + sizeof(size_t).
  virtual void* buffer_raw_ptr(void* ptr) const = 0;

  // Device queries
  virtual bool is_available() const = 0;
  virtual int device_count() const = 0;
  virtual const BackendInfo::PropertyMap& device_info(
      int device_index = 0) const = 0;

  // Memory management
  virtual size_t get_active_memory() const = 0;
  virtual size_t get_peak_memory() const = 0;
  virtual void reset_peak_memory() = 0;
  virtual size_t get_cache_memory() const = 0;
  virtual size_t set_cache_limit(size_t limit) = 0;
  virtual size_t set_memory_limit(size_t limit) = 0;
  virtual size_t get_memory_limit() const = 0;
  virtual size_t set_wired_limit(size_t limit) = 0;
  virtual void clear_cache() = 0;

  // Event operations (for GPU synchronization)
  virtual std::shared_ptr<void> make_event(Stream stream) = 0;
  virtual void event_wait(std::shared_ptr<void>& event, uint64_t value) = 0;
  virtual void event_wait_stream(
      std::shared_ptr<void>& event, Stream stream, uint64_t value) = 0;
  virtual void event_signal(
      std::shared_ptr<void>& event, Stream stream, uint64_t value) = 0;
  virtual bool event_is_signaled(
      std::shared_ptr<void>& event, uint64_t value) const = 0;

  // Fence operations (for cross-device synchronization)
  virtual std::shared_ptr<void> make_fence(Stream stream) = 0;
  virtual void fence_wait(
      std::shared_ptr<void>& fence, Stream stream, const array& x) = 0;
  virtual void fence_update(
      std::shared_ptr<void>& fence, Stream stream, const array& x,
      bool cross_device) = 0;

  // Cross-backend copy (for multi-backend support)
  virtual void copy_to_host(const void* src, void* dst, size_t size) = 0;
  virtual void copy_from_host(const void* src, void* dst, size_t size) = 0;
};

}  // namespace mlx::core
```

#### 1.2 C Entry Points (exported from each `.so` / `.dylib`)

```cpp
// mlx/backend_plugin.h

// Required.  Construct and return the backend.  nullptr = init failed.
extern "C" MLX_API mlx::core::BackendInterface* mlx_backend_init();

// Optional.  Return a capability score without fully initializing.
// 0 = unsupported.  Called before mlx_backend_init() so the loader can
// pick the best candidate cheaply.
extern "C" MLX_API int mlx_backend_score();
```

Helper macros for backend authors.  `MLX_BACKEND_IMPL` wraps the factory
in a `try`/`catch` (returning `nullptr` on exception) and also generates
the `mlx_backend_abi_info` entry point automatically:

```cpp
#define MLX_BACKEND_IMPL(factory_fn)                                     \
  extern "C" MLX_API mlx::core::BackendInterface* mlx_backend_init() {   \
    try { return factory_fn(); } catch (...) { return nullptr; }         \
  }                                                                      \
  extern "C" MLX_API MLX_BackendABIInfo mlx_backend_abi_info() {         \
    return mlx_make_abi_info();                                          \
  }

#define MLX_BACKEND_SCORE_IMPL(score_fn)                     \
  extern "C" MLX_API int mlx_backend_score() { return score_fn(); }
```

### 2. Backend Registry

A singleton `BackendRegistry` manages discovery, loading, and lookup.

```cpp
namespace mlx::core {

struct LoadedBackend {
  BackendInterface* backend;
  void* dl_handle;
  BackendInfo info;
  std::string library_path;
};

class BackendRegistry {
 public:
  /// Load a single backend from an explicit path.
  BackendInterface* load(const std::string& path);

  /// Search for libmlx-{name}-*.{so,dylib} candidates, score them,
  /// and load the best one.
  BackendInterface* load_best(const std::string& name);

  /// Discover and load all backends from search paths.
  void load_all(const BackendFilter* filter = nullptr);

  /// Look up the backend that owns the given device.
  /// Uses the device type and index to resolve across multiple loaded
  /// backends of the same type (e.g. CUDA on gpu:0, Vulkan on gpu:1).
  BackendInterface* get(const Device& d) const;

  /// Look up a backend by name.
  BackendInterface* get(const std::string& name) const;

  /// Whether any backend of the given device type is available.
  bool is_available(Device::DeviceType type) const;

  /// Total device count for a given type (summed across all backends).
  int device_count(Device::DeviceType type) const;

  /// All loaded backends.
  const std::vector<LoadedBackend>& backends() const;

  /// Enumerate search paths (modifiable).
  std::vector<std::string>& search_paths();

  static BackendRegistry& instance();
};

}  // namespace mlx::core
```

### 3. Client-Side Filtering

Applications can steer which backends get loaded:

```cpp
struct BackendFilter {
  std::vector<std::string> allowed_names;   // Glob patterns, e.g. "cpu-*"
  std::vector<std::string> blocked_names;   // Glob patterns, e.g. "vulkan-*"
  std::function<bool(const BackendInfo&)> custom_filter;  // Escape hatch
};
```

`allowed_names` and `blocked_names` cover the common cases (restrict to a
specific backend family, exclude an experimental backend, etc.).
`custom_filter` is available for anything more specific — callers who need
to inspect device properties or enforce a minimum score can do so there.

### 4. Search Paths and Naming Convention

| Platform | Pattern | Example |
|----------|---------|---------|
| macOS | `libmlx-{name}[-variant].dylib` | `libmlx-metal.dylib` |
| Linux | `libmlx-{name}[-variant].so` | `libmlx-cuda-v13.so` |
| Windows | `mlx-{name}[-variant].dll` | `mlx-cuda.dll` |

Search path resolution (mutually exclusive):

**If `MLX_BACKEND_PATH` is set:** Use ONLY the paths in this environment variable
(colon-separated on Unix; semicolon on Windows). This provides complete control
over which backends are loaded, useful for testing and deployment scenarios.

**If `MLX_BACKEND_PATH` is not set:** Use built-in defaults in order:
1. Compile-time `MLX_BACKEND_DIR` (set by CMake install rules).
2. A `backends/` subdirectory relative to `libmlx` location.

This "either/or" design ensures predictable behavior: setting the environment
variable gives complete control, while the default provides sensible fallbacks
for standard installations.

### 5. Incompatibility Detection

| Check | When | Action |
|-------|------|--------|
| `dlopen` failure | Load time | Log warning, skip |
| `mlx_backend_abi_info()` mismatch | Before init | Log error with details (compiler, stdlib, sizes), `dlclose`, skip |
| `mlx_backend_score()` returns 0 | Before init | Log debug, skip |
| `mlx_backend_init()` returns nullptr | Init time | Log warning, skip |
| `api_version() != MLX_BACKEND_API_VERSION` | After init | Log error with both versions, destroy, skip |
| Rejected by BackendFilter | Before init | Skip |

Error messages include the library path and specific failure reason.
See the **ABI Safety** section for details on the ABI compatibility check.

---

## Integration with MLX Core

### Dispatch

Today's static dispatch in `transforms.cpp:236`:

```cpp
if (arr.primitive().device() == Device::gpu) {
    gpu::eval(arr);
} else {
    cpu::eval(arr);
}
```

With dynamic backends enabled, the `gpu::eval` and `cpu::eval` free functions
become thin dispatchers that forward to the registry.  The full `Device`
(type + index) is used so that multiple backends of the same type are
routed correctly:

```cpp
auto& device = arr.primitive().device();
auto* backend = BackendRegistry::instance().get(device);
if (!backend) {
    throw std::runtime_error("No backend loaded for device ...");
}
backend->eval(arr);
```

The same pattern applies to `new_stream`, `finalize`, `synchronize`,
`is_available`, `device_count`, and `device_info` — all currently resolved
at link time, all become registry lookups.

### Device Index Mapping

Multiple backends of the same device type each contribute to a shared
device index space.  When a backend is registered, the registry assigns
it a contiguous range of global indices based on its `device_count`.

For example, if a CUDA backend reports 2 GPUs and a Vulkan backend
reports 1 GPU, the index mapping becomes:

| Global index | Backend | Backend-local index |
|-------------|---------|-------------------|
| `gpu:0` | cuda | 0 |
| `gpu:1` | cuda | 1 |
| `gpu:2` | vulkan | 0 |

`BackendRegistry::get(Device{gpu, 2})` returns the Vulkan backend and
the registry translates the global index to a backend-local index before
calling into the backend.  `device_count(Device::gpu)` returns 3.

Scoring determines which *variant* wins within a backend family (e.g.
`cuda-v12` vs `cuda-v13`), but backends of different families (e.g.
`cuda` and `vulkan`) are not in competition — they are all loaded and
each owns its devices.

### Allocator and Memory Routing

When `MLX_DYNAMIC_BACKENDS=ON`, the allocator subsystem must route
operations to the correct backend in a multi-backend environment (e.g.,
CUDA on gpu:0-1, Vulkan on gpu:2).  Three pieces require dynamic routing:
`allocator::allocator()`, `Buffer::raw_ptr()`, and the memory functions
in `memory.h`.

`backend/dynamic/allocator.cpp` replaces `no_gpu/allocator.cpp` and
provides this routing.

#### Buffer origin tracking

Each `Buffer` tracks the `Device` that allocated it.  When an array is
created on a specific device, the allocator tags the buffer with that
device.  This allows `raw_ptr()` and other operations to route to the
correct backend without a global "current allocator" assumption.

```cpp
class Buffer {
  void* ptr_;
  Device device_;  // Tracks which device allocated this buffer
  // ...
};
```

#### `allocator::allocator(const Device&)` — Device-Specific Allocator

The device-specific allocator API routes to the correct backend based on
the device argument:

```cpp
// mlx/allocator.h — public API
MLX_API Allocator& allocator(const Device& d);

// Implementation in backend/dynamic/allocator.cpp
Allocator& allocator(const Device& d) {
  if (d.type == Device::cpu) {
    return common_allocator();
  }
  auto* backend = BackendRegistry::instance().get(d);
  if (backend) return backend->allocator();
  throw std::runtime_error("No backend loaded for device");
}
```

#### `allocator::allocator()` — Deprecated No-Argument Form

The no-argument form is **deprecated** because it cannot distinguish between
multiple GPU backends.  It returns the allocator for gpu:0 if any GPU backend
is loaded, otherwise falls back to the CPU allocator:

```cpp
// mlx/allocator.h — deprecated
[[deprecated("Use allocator(const Device&) for explicit device routing")]]
MLX_API Allocator& allocator();

// Implementation — maps to gpu:0 for backward compatibility
Allocator& allocator() {
  auto* backend = BackendRegistry::instance().get(Device::gpu);  // gpu:0
  if (backend) return backend->allocator();
  return common_allocator();
}
```

#### `Buffer::raw_ptr()`

Each backend stores different data in the opaque `Buffer::ptr_` — Metal
stores `MTL::Buffer*`, CUDA stores `CudaBuffer*`, CPU stores a
`size_t`-prefixed allocation.  `Buffer::raw_ptr()` uses the buffer's
tracked device to route to the correct backend:

```cpp
void* Buffer::raw_ptr() {
  if (!ptr_) return nullptr;
  if (device_.type == Device::gpu) {
    auto* backend = BackendRegistry::instance().get(device_);
    if (backend) return backend->buffer_raw_ptr(ptr_);
  }
  return static_cast<size_t*>(ptr_) + 1;  // CPU fallback
}
```

The hot-path cost is one singleton access + one map lookup + one virtual
call.  This is acceptable — `raw_ptr()` is called when reading results
back to the host, not in the GPU eval loop.

#### Memory functions

The 9 process-global memory functions (`get_active_memory`,
`get_peak_memory`, `reset_peak_memory`, `get_cache_memory`,
`set_cache_limit`, `set_memory_limit`, `get_memory_limit`,
`set_wired_limit`, `clear_cache`) have two forms:

**Per-device (with device argument):** The recommended API for
multi-backend configurations.  Returns or sets values for a specific
device:

```cpp
size_t get_active_memory(const Device& d) {
  auto* backend = BackendRegistry::instance().get(d);
  if (backend) return backend->get_active_memory();
  return 0;
}

size_t set_cache_limit(const Device& d, size_t limit) {
  auto* backend = BackendRegistry::instance().get(d);
  if (backend) return backend->set_cache_limit(limit);
  return 0;
}
```

**Aggregate getters (no device argument):** Return the sum across all
loaded GPU backends.  Useful for monitoring total memory usage:

```cpp
size_t get_active_memory() {
  size_t total = 0;
  for (auto* backend : BackendRegistry::instance().gpu_backends()) {
    total += backend->get_active_memory();
  }
  return total;
}
```

The aggregate getter form ensures that `transforms.cpp`'s memory pressure
check (`get_active_memory() > get_memory_limit()`) reflects total GPU
memory usage across all backends.

**Aggregate setters (no device argument):** Deprecated.  Broadcasting the
same limit to GPUs with different VRAM sizes (e.g., setting the same
cache limit to a 24GB NVIDIA GPU and a 16GB Intel Arc) is unlikely to be
correct.  These functions throw `std::runtime_error` if multiple GPU
backends are loaded, guiding users to the per-device API:

```cpp
[[deprecated("Use set_cache_limit(Device, size_t) for multi-backend")]]
size_t set_cache_limit(size_t limit) {
  auto backends = BackendRegistry::instance().gpu_backends();
  if (backends.size() > 1) {
    throw std::runtime_error(
        "set_cache_limit() without device argument is ambiguous with "
        "multiple GPU backends. Use set_cache_limit(device, limit).");
  }
  if (backends.empty()) return 0;
  return backends[0]->set_cache_limit(limit);
}
```

This deprecation path ensures single-backend users see no behavior change
while multi-backend users get a clear error rather than silent incorrect
behavior.

**Note:** `reset_peak_memory()` and `clear_cache()` without a device argument
apply to all backends, which is reasonable behavior for these operations
(resetting stats or clearing caches globally is a sensible default).

#### Event and Fence Routing

Events and fences are GPU synchronization primitives.  When
`MLX_DYNAMIC_BACKENDS=ON`, `backend/dynamic/event.cpp` and
`backend/dynamic/fence.cpp` route GPU stream operations through the
registry using the stream's device:

```cpp
// dynamic/event.cpp
Event::Event(Stream stream) : stream_(stream) {
  if (stream.device.type() == Device::gpu) {
    auto* backend = BackendRegistry::instance().get(stream.device);
    if (backend) {
      event_ = backend->make_event(stream);
    }
  }
  // CPU fallback uses condition variable
}

void Event::wait() {
  if (event_) {
    auto* backend = BackendRegistry::instance().get(stream_.device);
    backend->event_wait(event_, value_);
  } else {
    // CPU condition variable wait
  }
}
```

The `BackendInterface` event/fence methods wrap native synchronization
primitives (e.g. `cudaEvent_t` for CUDA) in `std::shared_ptr<void>`,
which the dynamic routing code holds opaquely and passes back to the
backend.

#### Constraints

Backends must be loaded before any arrays are allocated.  `load_all()`
(or explicit `load()` calls) should complete before any `eval()` call.
This ensures the registry can route operations to the correct backend
from the start and avoids synchronization complexity in the hot path.

A backend's `allocator()` method must return its own allocator instance
directly — it must **not** call the global `allocator::allocator()`.
When `MLX_DYNAMIC_BACKENDS=ON`, the global function routes through the
registry back to the backend, creating infinite recursion.  Backend
authors must be aware of this constraint.

### Plugin Implementation Constraints

Backend plugins face symbol resolution constraints that differ from
statically compiled backends:

1. **Symbol interposition:** Plugin code that calls `gpu::` or `cpu::`
   namespace functions may resolve to libmlx's registry-routing versions
   rather than the plugin's own implementations, causing deadlocks or
   infinite recursion. On Linux, `RTLD_DEEPBIND` mitigates this. GPU
   plugins should use direct runtime API calls (e.g., `cudaGetDeviceCount`)
   for device queries.

2. **Own allocator instance:** A plugin's `allocator()` must return its
   own allocator, not call `allocator::allocator()`.

3. **Self-contained eval chain:** Plugins must compile in their own
   complete eval implementation, not rely on libmlx's GPU code paths.

See `backend/cuda/plugin.cpp` for a reference implementation.

### Backward Compatibility

```cmake
option(MLX_DYNAMIC_BACKENDS "Enable runtime backend loading" OFF)
```

- **OFF (default):** Everything works exactly as today.  No `dlopen` code
  compiled.  GPU backends (Metal/CUDA) are statically linked into libmlx
  based on `MLX_BUILD_METAL` and `MLX_BUILD_CUDA`.
- **ON:** No GPU backend is statically linked into libmlx.  The `dynamic/`
  directory provides routing stubs that forward all GPU operations through
  the registry.  GPU functionality is unavailable until a backend plugin
  is loaded via `load()` or `load_all()`.  Metal and CUDA are built as
  separate plugin libraries (`libmlx-metal.dylib`, `libmlx-cuda.so`).

### CPU Backend

A CPU backend is always statically linked into libmlx. **CPU operations
must never fail** — unlike GPU backends where missing plugins correctly
result in "GPU unavailable," CPU eval must always work.

The architecture supports optional CPU backend plugins (e.g., for x86 SIMD
optimization in future work). If CPU plugins are registered, they will be
used; otherwise the built-in implementation handles all CPU operations.
This ensures the foundation is ready for future SIMD plugin work without
requiring changes to the core infrastructure.

---

## Backend Variants and Scoring

Multiple variants of the same backend can coexist (e.g., `libmlx-cuda-v12.so`
and `libmlx-cuda-v13.so`). The loader groups candidates by name stem and
selects the highest-scoring compatible variant per group.

The `mlx_backend_score()` entry point enables runtime capability detection.
Common variant axes include:

- **Toolkit version:** Score based on installed driver version
- **Architecture:** Score based on device compute capability (CUDA) or
  SIMD level (future x86 CPU plugins using CPUID to detect AVX-512/AVX2/SSE)

The scoring contract: higher score wins, zero means incompatible. See
`backend/cuda/plugin.cpp` for a reference implementation.

---

## Client Usage Examples

### Automatic discovery (C++)

```cpp
#include <mlx/mlx.h>

int main() {
    mlx::core::BackendRegistry::instance().load_all();

    // Normal MLX usage — dispatch is transparent.
    auto x = mlx::core::ones({3, 3});
    auto y = mlx::core::matmul(x, x);
    mlx::core::eval(y);
}
```

### Load a specific backend

```cpp
mlx::core::BackendRegistry::instance().load("/opt/mlx/backends/libmlx-cuda-v13.so");
auto x = mlx::core::ones({3, 3}, mlx::core::float32, mlx::core::Device::gpu);
```

### Filtered loading

```cpp
mlx::core::BackendFilter filter;
filter.allowed_names = {"cpu-*"};
mlx::core::BackendRegistry::instance().load_all(&filter);
```

### Python API

```python
import mlx.core as mx

mx.backends.load_all()

for b in mx.backends.list():
    print(f"{b.name}  score={b.score}  type={b.device_type}")

mx.backends.load("/opt/mlx/libmlx-cuda-v13.so")
mx.backends.load_all(allowed=["cpu-*"], blocked=["vulkan-*"])
```

### Multiple GPU backends (C++)

```cpp
mlx::core::BackendRegistry::instance().load_all();

// System has an NVIDIA GPU (CUDA) and an Intel Arc GPU (Vulkan)
// gpu:0 and gpu:1 are CUDA, gpu:2 is Vulkan
auto cuda_device = mlx::core::Device(mlx::core::Device::gpu, 0);
auto vulkan_device = mlx::core::Device(mlx::core::Device::gpu, 2);

auto x = mlx::core::ones({3, 3}, mlx::core::float32, cuda_device);
auto y = mlx::core::ones({3, 3}, mlx::core::float32, vulkan_device);
```

### Environment variable

```bash
export MLX_BACKEND_PATH=/home/user/mlx-backends
python my_app.py
```

---

## Loading Sequence

```
1. BackendRegistry::load_all(filter)
2. Enumerate search paths (env var → compile-time dir → libmlx dir)
3. Glob for libmlx-*.{so,dylib,dll}
4. Group candidates by name stem
   (libmlx-cuda-v12.so + libmlx-cuda-v13.so → group "cuda"
    libmlx-vulkan.so → group "vulkan")
5. Apply BackendFilter (allowed/blocked name globs, custom callback)
6. For each surviving candidate:
   a. dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND)
      (RTLD_DEEPBIND on Linux ensures plugins prefer their own symbols
       over host symbols, preventing infinite recursion)
   b. dlsym("mlx_backend_abi_info") → compare compiler, stdlib,
      struct sizes against MLX core's own values — mismatch → dlclose, skip
   c. dlsym("mlx_backend_score") — returns 0 → dlclose, skip
7. Select highest-scoring candidate per group
   (cuda-v13 beats cuda-v12 within the "cuda" group;
    "cuda" and "vulkan" are separate groups — both survive)
8. For each group winner:
   a. dlsym("mlx_backend_init") → call it (wrapped in try/catch)
   b. Check api_version() == MLX_BACKEND_API_VERSION
   c. Register in BackendRegistry
9. Build device index mapping: for each device type, assign contiguous
   global indices across all loaded backends of that type
   (e.g. cuda gets gpu:0..1, vulkan gets gpu:2)
```

---

## Build System Changes

### CMake Options

```cmake
option(MLX_DYNAMIC_BACKENDS "Enable runtime backend loading" OFF)
```

- **`MLX_DYNAMIC_BACKENDS=OFF` (default):** Unchanged from today. Backends
  are statically linked into libmlx based on `MLX_BUILD_METAL`/`MLX_BUILD_CUDA`.

- **`MLX_DYNAMIC_BACKENDS=ON`:** GPU backends are built as separate plugin
  libraries. GPU is unavailable until plugins are loaded at runtime. CPU
  backend remains statically linked and always works.

### Building with Dynamic Backends

```bash
# CUDA plugin (Linux)
cmake -B build -DMLX_DYNAMIC_BACKENDS=ON -DMLX_BUILD_CUDA=ON

# Metal plugin (macOS)
cmake -B build -DMLX_DYNAMIC_BACKENDS=ON -DMLX_BUILD_METAL=ON
```

This produces `libmlx` plus plugin libraries (`libmlx-cuda.so`,
`libmlx-metal.dylib`) in `${CMAKE_BINARY_DIR}/backends/`. Install rules
place plugins into `${CMAKE_INSTALL_LIBDIR}/mlx/backends/`.

### Platform Notes

| Platform | Static libmlx + Plugins | Shared libmlx + Plugins |
|----------|------------------------|------------------------|
| Linux    | Supported              | Supported              |
| macOS    | Supported              | Supported              |
| Windows  | Not supported          | Required               |

**Windows** requires shared libmlx because Windows DLLs need all symbols
resolved at link time. `BUILD_SHARED_LIBS` is automatically forced ON.

**Linux/macOS with static libmlx:** Link against `mlx::plugin_host` instead
of `mlx` to ensure all symbols are available to plugins:

```cmake
if(TARGET mlx::plugin_host)
  target_link_libraries(my_app PRIVATE mlx::plugin_host)
else()
  target_link_libraries(my_app PRIVATE mlx)
endif()
```

---

## ABI Safety

Dynamic loading across shared library boundaries raises several C++ ABI
concerns that the design must handle explicitly.

### Compiler and standard library matching

`BackendInterface` uses C++ virtual dispatch, and `BackendInfo` contains
`std::string`, `std::vector`, and `std::unordered_map`.  These types have
ABI-specific layouts that differ between:

- libstdc++ vs libc++ (e.g. `std::string` SSO layout differs)
- Major compiler versions that change ABI (GCC's `_GLIBCXX_USE_CXX11_ABI`)
- MSVC runtime versions (`/MD` vs `/MT`, different VS versions)

**Mitigation:** Each backend embeds an ABI descriptor that the loader checks
before calling `mlx_backend_init`.  This is a third C entry point:

```cpp
struct MLX_BackendABIInfo {
  int    abi_info_size;       // sizeof(this struct), for forward compat
  int    api_version;         // MLX_BACKEND_API_VERSION
  char   compiler_id[32];     // e.g. "Clang", "GNU", "MSVC"
  int    compiler_major;
  int    compiler_minor;
  char   stdlib_id[32];       // e.g. "libc++", "libstdc++", "msvc"
  int    sizeof_string;       // sizeof(std::string)
  int    sizeof_BackendInfo;  // sizeof(BackendInfo)
};

// Required.  Must be callable before mlx_backend_init.
extern "C" MLX_API MLX_BackendABIInfo mlx_backend_abi_info();
```

The macro `MLX_BACKEND_IMPL` auto-populates this from compiler predefined
macros.  The loader compares these fields against its own build and rejects
mismatches with a clear diagnostic (e.g. "backend libmlx-cuda-v13.so was
built with libstdc++ but MLX uses libc++").

This is a plain C struct returned by value through an `extern "C"` function,
so it is safe to call across any compiler combination — the check happens
before any C++ objects are exchanged.

### Exception safety across the C boundary

The `extern "C"` entry points (`mlx_backend_init`, `mlx_backend_score`,
`mlx_backend_abi_info`) must not propagate C++ exceptions — doing so is
undefined behavior.  The `MLX_BACKEND_IMPL` macro wraps the factory function
in a `try`/`catch` that returns `nullptr` on exception, and the loader
treats `nullptr` as an initialization failure.

Once a backend is initialized and the ABI check passes, subsequent calls
go through `BackendInterface` virtual methods.  These are normal C++ calls
within a single ABI, so exceptions propagate naturally.  Backend authors
are expected to throw standard MLX exceptions (e.g. `std::runtime_error`)
from `eval()` and other methods, and MLX's existing error handling applies.

### vtable layout

`BackendInterface` is a pure abstract class defined in an MLX header that
both the core library and the backend compile against.  Provided both sides
use the same compiler ABI (enforced by the `mlx_backend_abi_info` check),
the vtable layout is deterministic — the Itanium C++ ABI (used by
Clang/GCC on Linux/macOS) and the MSVC ABI both guarantee stable vtable
ordering for single-inheritance hierarchies.

### C++ symbol resolution between backend and core

Backend code calls extensively into core MLX C++ APIs — it receives
`array&` objects, calls `arr.primitive().eval_gpu()`, allocates memory
through `allocator::Allocator`, interacts with `Stream`, `Device`, etc.
These are all mangled C++ symbols that the backend `.so` must resolve
against the core library.

This works through normal dynamic linker mechanisms:

- **`libmlx` built as a shared library:** The backend `.so` links against
  `libmlx.so`/`.dylib` and the dynamic linker resolves mangled symbols at
  load time, the same as any two shared libraries.
- **`libmlx` built as a static library (Linux/macOS only):** The MLX symbols
  live in the host executable.  The executable must be linked with
  `-rdynamic` (`--export-dynamic`) and `--whole-archive` so that all symbols
  are included and visible to `dlopen`-ed libraries.  The `mlx::plugin_host`
  CMake interface target provides these flags automatically.
- **Windows:** Static libmlx + dynamic plugins is not supported.  Windows
  DLLs require all symbols resolved at link time via import libraries, and
  there's no practical way for plugins to resolve symbols from an arbitrary
  host executable.  `BUILD_SHARED_LIBS` is forced ON when
  `MLX_DYNAMIC_BACKENDS=ON` on Windows.

### Plugin symbol isolation

On Linux, plugins are loaded with `RTLD_DEEPBIND` to ensure they prefer
their own internal symbols over symbols from the host. macOS two-level
namespaces and Windows DLL semantics provide similar isolation by default.

The `RTLD_NOW` flag forces all symbols to resolve at load time, so missing
or mismatched symbols produce a clear `dlopen` failure rather than a
runtime crash.

---

## No backend unloading

Once a backend is loaded and initialized it remains loaded for the lifetime of
the process.  `dlclose` is unsafe when threads may hold function pointers or
vtable references into backend code, and the complexity of reference-counting or
safe teardown is not justified. Process exit handles cleanup.  If this is ever
revisited, it would require a full quiesce of all streams using that backend
before unloading.

---

## Multi-Backend Support

When multiple GPU backends are loaded simultaneously (e.g., CUDA + Vulkan),
users have full control over data placement.  Cross-backend operations
error with clear messages; users explicitly copy data between backends.

### Technical Considerations

**Device index assignment:** Indices are assigned based on backend score
(descending), so higher-priority backends consistently get lower indices.
For example, if CUDA (score=100) and Vulkan (score=50) both load, CUDA
gets `gpu:0-N` and Vulkan gets the subsequent indices.

**Cross-backend data transfer:** Data cannot move directly between backends
(CUDA device memory is not accessible from Vulkan).  Transfers between
backends must stage through host memory.  Intra-backend transfers (e.g.,
CUDA GPU 0 ↔ CUDA GPU 1) can use backend-native mechanisms like unified
memory.

**NCCL and distributed operations:** NCCL handles collective operations
within the CUDA ecosystem and is orthogonal to single-tensor cross-device
copies.  NCCL cannot transfer data between CUDA and non-CUDA backends.

### User Experience

```python
import mlx.core as mx

mx.backends.load_all()
# Loads CUDA (gpu:0-1) and Vulkan (gpu:2)

a = mx.ones((1000, 1000), device=mx.gpu(0))   # CUDA
b = mx.ones((1000, 1000), device=mx.gpu(2))   # Vulkan

# Errors with clear message - can't mix backends
c = mx.matmul(a, b)
# ValueError: Inputs are on different backends (cuda:gpu:0, vulkan:gpu:2).
#             Use mx.copy(x, device) to move data to the same backend.

# Explicit transfer via host staging
a_vulkan = mx.copy(a, mx.gpu(2))
c = mx.matmul(a_vulkan, b)  # Works - both on Vulkan
```

### Future Possibilities

This design intentionally keeps multi-backend support explicit.  Future
enhancements could include automatic cross-backend transfers (opt-in),
optimized intra-backend transfers, or model partitioning APIs, but these
are not scoped for the initial API design.

---

## Open Questions

1. **Python packaging.**  `pip install mlx` needs a strategy for
   platform-appropriate backend libraries — likely separate packages
   like `mlx-backend-cuda`.
