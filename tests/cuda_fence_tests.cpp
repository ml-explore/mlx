// Copyright © 2026 Apple Inc.

#include <chrono>

#include <doctest/doctest.h>

#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

using clk = std::chrono::steady_clock;

double ms_since(clk::time_point t) {
  return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

// Producer work that is long enough to measure on any GPU.
array matmul_chain(const array& seed, Stream s, int iters) {
  array a = seed;
  for (int i = 0; i < iters; ++i) {
    a = matmul(a, seed, s);
  }
  return a;
}

} // namespace

// A GPU to GPU dependency must be resolved in the consumer stream. If the
// fence waits on the host instead, async_eval cannot return until the
// producer kernels have run.
TEST_CASE("cross stream async_eval does not block the host") {
  if (!gpu::is_available()) {
    return;
  }

  auto producer = new_stream(Device::gpu);
  auto consumer = new_stream(Device::gpu);

  int n = 2048;
  int iters = 100;
  array seed = full({n, n}, 1.0f / n, float32, producer);
  eval(seed);

  // Compile the kernels so the timed runs do not pay for the JIT.
  eval(abs(matmul(seed, seed, producer), consumer));

  auto t = clk::now();
  eval(matmul_chain(seed, producer, iters));
  double compute_ms = ms_since(t);

  array out = abs(matmul_chain(seed, producer, iters), consumer);
  t = clk::now();
  async_eval(out);
  double submit_ms = ms_since(t);

  // Submission enqueues the work, it must not wait for it.
  CHECK_LT(submit_ms, 0.5 * compute_ms);

  eval(out);
  synchronize(producer);
  synchronize(consumer);
}
