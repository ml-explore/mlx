// Copyright © 2026 Apple Inc.

#include "mlx/backend/cpu/threading/accelerate/thread_pool.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <thread>

#include <dispatch/dispatch.h>

namespace mlx::core::cpu {

namespace {
// Matches the openblas backend: nested parallel_for calls from inside a
// running task execute inline as f(0, 1).
thread_local bool g_in_parallel_region = false;
} // namespace

AccelerateThreadPool::AccelerateThreadPool()
    : max_threads_(std::max(1u, std::thread::hardware_concurrency())) {}

void AccelerateThreadPool::parallel_for(
    int n_threads,
    std::function<void(int tid, int nth)> f) {
  if (n_threads <= 1 || g_in_parallel_region) {
    f(0, 1);
    return;
  }

  // Clamp to max_threads
  n_threads = std::min(n_threads, max_threads_);

  // Use GCD's dispatch_apply for parallel execution.
  // This uses the system-wide thread pool managed by the kernel.
  // Capture any exception from worker blocks since GCD blocks cannot
  // propagate C++ exceptions (they would std::terminate).
  // Use heap-allocated state because __block storage copies non-copyable types.
  struct ExState {
    std::exception_ptr eptr{nullptr};
    std::atomic<bool> taken{false};
  };
  auto ex = std::make_shared<ExState>();

  dispatch_apply(
      n_threads,
      dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
      ^(size_t i) {
        bool was_nested = g_in_parallel_region;
        g_in_parallel_region = true;
        try {
          f(static_cast<int>(i), n_threads);
        } catch (...) {
          bool expected = false;
          if (ex->taken.compare_exchange_strong(expected, true)) {
            ex->eptr = std::current_exception();
          }
        }
        g_in_parallel_region = was_nested;
      });

  if (ex->eptr) {
    std::rethrow_exception(ex->eptr);
  }
}

int AccelerateThreadPool::max_threads() const {
  return max_threads_;
}

// Factory function implementation
std::unique_ptr<ThreadPoolBackend> create_thread_pool_backend() {
  return std::make_unique<AccelerateThreadPool>();
}

} // namespace mlx::core::cpu
