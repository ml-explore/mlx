// Copyright © 2026 Apple Inc.

#pragma once

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "mlx/api.h"

// Keep a kernel out of line. Used for inner loops that are called both
// directly and from the worker pool: when such a loop gets inlined into the
// caller, GCC may fail to vectorize the inlined copy.
#if defined(_MSC_VER)
#define MLX_CPU_NOINLINE __declspec(noinline)
#else
#define MLX_CPU_NOINLINE __attribute__((noinline))
#endif

namespace mlx::core::cpu {

// Maximum number of threads (including the calling thread) used by
// intra-op parallel CPU kernels. Defaults to the number of hardware threads
// and can be overridden with the MLX_CPU_NUM_THREADS environment variable.
// A value of 1 disables intra-op parallelism.
MLX_API int max_threads();

namespace detail {

using RangeFn = void (*)(void* ctx, int64_t begin, int64_t end);

// Splits [0, n) into `nchunks` contiguous ranges and runs `fn` on each of
// them using the calling thread and the CPU worker pool. Blocks until all
// ranges are done. Falls back to a single serial call when the pool is
// already in use (e.g. nested parallel regions or concurrent streams).
MLX_API void parallel_for_impl(int64_t n, int nchunks, RangeFn fn, void* ctx);

} // namespace detail

// Run f(begin, end) over the range [0, n). The range is split into at most
// max_threads() chunks, each with at least `grain` elements. For small `n`
// this is a single direct call of f(0, n) with no synchronization overhead.
template <typename F>
void parallel_for(int64_t n, int64_t grain, F&& f) {
  if (n <= 0) {
    return;
  }
  grain = std::max<int64_t>(grain, 1);
  int64_t nchunks = std::min<int64_t>(max_threads(), n / grain);
  if (nchunks <= 1) {
    f(int64_t(0), n);
    return;
  }
  using Fn = std::remove_reference_t<F>;
  detail::parallel_for_impl(
      n,
      static_cast<int>(nchunks),
      [](void* ctx, int64_t b, int64_t e) { (*static_cast<Fn*>(ctx))(b, e); },
      const_cast<void*>(static_cast<const void*>(&f)));
}

// Minimum amount of work (in bytes touched) per thread for memory-bound
// elementwise kernels. Below this the cost of waking up workers dominates.
constexpr int64_t kElementwiseGrainBytes = 128 * 1024;

template <typename T>
constexpr int64_t elementwise_grain() {
  return std::max<int64_t>(kElementwiseGrainBytes / sizeof(T), 1);
}

} // namespace mlx::core::cpu
