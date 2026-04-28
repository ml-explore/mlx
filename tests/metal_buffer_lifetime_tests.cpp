// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"

#include <atomic>
#include <thread>
#include <vector>

#include "mlx/fast.h"
#include "mlx/memory.h"
#include "mlx/mlx.h"

using namespace mlx::core;

// Smoke test: concurrent eval on built-in primitives must not crash.
//
// Built-in primitives like matmul have well-rooted shared_ptr chains
// from the standard ops layer, so the buffer lifetime race rarely
// surfaces on this code path; this test just guards against obvious
// regressions in the eval path under concurrent threads.
TEST_CASE("test concurrent eval smoke") {
  if (!gpu::is_available()) {
    return;
  }

  constexpr int kThreads = 16;
  constexpr int kIters = 8;

  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&]() {
      try {
        for (int i = 0; i < kIters; ++i) {
          auto x = random::normal({256, 256});
          auto w = random::normal({256, 256});
          auto y = matmul(x, w);
          eval(y);
        }
      } catch (...) {
        failures.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  CHECK_EQ(failures.load(), 0);
}

// Regression test for the buffer-lifetime contract that surfaces under
// Metal Validator (run with METAL_DEVICE_WRAPPER_TYPE=1 MTL_DEBUG_LAYER=1)
// as: "Metal object is being destroyed while still required to be alive
// by the command buffer".
//
// The MetalAllocator releases the underlying MTL::Buffer when the C++
// shared_ptr count hits zero. Allocator buffers use
// MTLResourceHazardTrackingModeUntracked and command buffers use
// commandBufferWithUnretainedReferences(); both Apple APIs require the
// application to keep bound buffers alive until the CB completes.
//
// The race needs THREE conditions stacked to surface deterministically:
//   1) high concurrency (many threads dispatching custom kernels in parallel),
//   2) custom Metal kernels (built-in primitives have well-rooted lifetime),
//   3) cache pressure (cache near full so frees route to direct release
//      rather than recycle-to-cache; without this the buffer stays alive
//      in the cache even though refcount logic is wrong).
//
// We reproduce (3) by setting `set_cache_limit(0)` so every free hits
// the direct-release path. With the patch, retain on bind keeps the
// buffer alive until CB completion; without it, Metal Validator flags
// the destroyed-buffer error.
TEST_CASE(
    "test custom kernel concurrent buffer lifetime under cache pressure") {
  if (!gpu::is_available()) {
    return;
  }

  // Force frees to skip the cache so refcount-incorrect paths surface.
  size_t saved_limit = set_cache_limit(0);
  clear_cache();

  constexpr int kThreads = 32;
  constexpr int kIters = 64;
  constexpr int kSize = 1024;

  // Trivial kernel: copy + accumulate inputs to output. Multiple inputs
  // make the bind path do real work per dispatch.
  auto kernel = fast::metal_kernel(
      "buffer_lifetime_pressure",
      std::vector<std::string>{"a", "b", "c"},
      std::vector<std::string>{"out"},
      "uint elem = thread_position_in_grid.x;\n"
      "if (elem < a_shape[0]) { out[elem] = a[elem] + b[elem] + c[elem]; }\n");

  std::atomic<int> failures{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&]() {
      try {
        for (int i = 0; i < kIters; ++i) {
          auto a = random::normal({kSize});
          auto b = random::normal({kSize});
          auto c = random::normal({kSize});
          std::vector<Shape> shapes{Shape{kSize}};
          std::vector<Dtype> dtypes{float32};
          auto outs = kernel(
              {a, b, c},
              shapes,
              dtypes,
              std::make_tuple(kSize, 1, 1),
              std::make_tuple(64, 1, 1),
              {},
              std::nullopt,
              false,
              {});
          eval(outs[0]);
          // `outs`, `a`, `b`, `c` go out of scope here. Without the
          // retain-on-bind fix and with cache pressure (limit=0),
          // their MTL::Buffers are released immediately while the
          // command buffer may still reference them.
        }
      } catch (...) {
        failures.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  set_cache_limit(saved_limit);
  CHECK_EQ(failures.load(), 0);
}
