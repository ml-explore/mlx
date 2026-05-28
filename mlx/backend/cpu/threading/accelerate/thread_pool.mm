// Copyright © 2026 Apple Inc.

#include "mlx/backend/cpu/threading/accelerate/thread_pool.h"

#include <dispatch/dispatch.h>
#include <atomic>
#include <memory>
#include <thread>

namespace mlx::core::cpu {

AccelerateThreadPool::AccelerateThreadPool()
    : max_threads_(std::thread::hardware_concurrency()) {}

void AccelerateThreadPool::parallel_for(
    int n_threads,
    std::function<void(int tid, int nth)> f) {
  if (n_threads <= 1) {
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
        try {
          f(static_cast<int>(i), n_threads);
        } catch (...) {
          bool expected = false;
          if (ex->taken.compare_exchange_strong(expected, true)) {
            ex->eptr = std::current_exception();
          }
        }
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
