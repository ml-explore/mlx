// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/backend/cpu/threading/base.h"

#include <cstdlib>
#include <memory>

namespace mlx::core::cpu {

/**
 * Thread pool for CPU backend parallelism.
 *
 * Provides a simple interface for parallel execution of work across multiple
 * threads. Uses platform-native thread pools for optimal performance:
 * - macOS: Grand Central Dispatch (GCD)
 * - Linux/Windows: OpenBLAS-coordinated threading
 *
 * Usage:
 *   auto& pool = ThreadPool::instance();
 *   int nth = std::min(pool.max_threads(), (int)(size / 4096));
 *   if (nth > 1) {
 *     pool.parallel_for(nth, [&](int tid, int nth) {
 *       size_t chunk = (size + nth - 1) / nth;
 *       size_t start = chunk * tid;
 *       size_t end = std::min(start + chunk, size);
 *       process_chunk(start, end);
 *     });
 *   } else {
 *     process_chunk(0, size);
 *   }
 */
class ThreadPool {
 public:
  /// Get the singleton instance.
  static ThreadPool& instance();

  /// Get the maximum number of threads available.
  int max_threads() const;

  /**
   * Execute a function in parallel across n_threads.
   *
   * @param n_threads Number of threads to use (1 to max_threads())
   * @param f Function to execute, called as f(thread_id, n_threads)
   *          where thread_id is in [0, n_threads) and n_threads is the
   *          total number of threads participating.
   *
   * Blocks until all threads complete their work.
   */
  void parallel_for(int n_threads, std::function<void(int tid, int nth)> f);

  ~ThreadPool();

  // Non-copyable
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

 private:
  ThreadPool();

  std::unique_ptr<ThreadPoolBackend> backend_;
};

// Minimum total elements before considering parallelization.
// Below this threshold, threading overhead exceeds benefits.
constexpr size_t MIN_TOTAL_ELEMENTS = 262144; // 256K elements

// Minimum elements per thread to avoid too many threads
constexpr size_t MIN_ELEMENTS_PER_THREAD = 32768; // 32K elements

/**
 * Helper to compute effective thread count for a given workload size.
 *
 * @param size Total number of elements to process
 * @param max_threads Maximum threads to use (typically pool.max_threads())
 * @return Number of threads to use (1 if size is too small)
 */
inline int effective_threads(size_t size, int max_threads) {
  // Don't parallelize small workloads
  if (size < MIN_TOTAL_ELEMENTS) {
    return 1;
  }
  // Compute threads based on per-thread minimum
  int n = static_cast<int>(size / MIN_ELEMENTS_PER_THREAD);
  return std::min(std::max(1, n), max_threads);
}

} // namespace mlx::core::cpu
