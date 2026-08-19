// Copyright © 2026 Apple Inc.

#include <cstdlib>
#include <string>

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

bool contains(const std::string& haystack, const char* needle) {
  return haystack.find(needle) != std::string::npos;
}

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

} // namespace

TEST_CASE("cuda graph cache thrashing keeps the encoder usable") {
  if (!is_available(Device::gpu)) {
    return;
  }

  // The graph cache capacity is read when the CommandEncoder of a stream is
  // constructed, so shrink it before creating the stream this test runs on.
  set_env("MLX_CUDA_GRAPH_CACHE_SIZE", "1");
  set_env("MLX_ENABLE_CACHE_THRASHING_CHECK", "1");
  set_env("MLX_USE_CUDA_GRAPHS", "1");

  auto s = new_stream(Device::gpu);

  int thrashing = 0;
  int corrupted = 0;
  int succeeded = 0;

  // The cache key is topological: one "K-" per node plus the dependency edges.
  // Shapes do not appear in it, so the number of chained ops is varied instead
  // to produce a distinct key on every iteration. With capacity 1 the check
  // fires once more than 2 * capacity misses have accumulated.
  for (int i = 1; i <= 8; ++i) {
    try {
      auto a = ones({4, 4}, float32, s);
      auto b = a;
      for (int k = 0; k < i; ++k) {
        b = add(b, a, s);
      }
      eval(b);
      ++succeeded;
    } catch (const std::exception& e) {
      const std::string msg = e.what();
      if (contains(msg, "Cache thrashing")) {
        ++thrashing;
      } else if (contains(msg, "cudaGraphAddDependencies")) {
        ++corrupted;
      } else {
        FAIL_CHECK("unexpected error: " << msg);
      }
    }
  }

  MESSAGE(
      "succeeded=" << succeeded << " thrashing=" << thrashing
                   << " corrupted=" << corrupted);

  // The check has to fire, otherwise the test is not exercising anything.
  CHECK(thrashing > 0);

  // Before the fix the encoder kept the failed graph's nodes and dependencies,
  // so every commit after the first thrashing exception failed here instead of
  // reporting the thrashing error again.
  CHECK(corrupted == 0);

  set_env("MLX_ENABLE_CACHE_THRASHING_CHECK", "0");
}
