// Copyright © 2026 Apple Inc.

#include "mlx/backend/cpu/threading/common.h"

namespace mlx::core::cpu {

ThreadPool& ThreadPool::instance() {
  // Leak on all platforms - see Scheduler singleton comment in scheduler.cpp.
  static ThreadPool* pool = new ThreadPool;
  return *pool;
}

ThreadPool::ThreadPool() : backend_(create_thread_pool_backend()) {}

void ThreadPool::parallel_for(
    int n_threads,
    std::function<void(int tid, int nth)> f) {
  if (n_threads <= 0) {
    return;
  }
  backend_->parallel_for(n_threads, std::move(f));
}

ThreadPool::~ThreadPool() = default;

int ThreadPool::max_threads() const {
  return backend_->max_threads();
}

} // namespace mlx::core::cpu
