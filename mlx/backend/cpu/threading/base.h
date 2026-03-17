// Copyright © 2026 Apple Inc.

#pragma once

#include <functional>
#include <memory>

namespace mlx::core::cpu {

/**
 * Pure virtual interface for platform-specific thread pool implementations.
 *
 * This interface abstracts the threading mechanism so different platforms
 * can use their respective thread pools:
 * - macOS: Grand Central Dispatch (GCD)
 * - Linux/Windows: OpenBLAS-coordinated threading
 */
class ThreadPoolBackend {
 public:
  virtual ~ThreadPoolBackend() = default;

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
  virtual void parallel_for(
      int n_threads,
      std::function<void(int tid, int nth)> f) = 0;

  /**
   * Get the maximum number of threads available.
   */
  virtual int max_threads() const = 0;
};

/**
 * Factory function to create the platform-specific thread pool backend.
 * Implemented in platform-specific files (accelerate/thread_pool.mm,
 * openblas/thread_pool.cpp).
 */
std::unique_ptr<ThreadPoolBackend> create_thread_pool_backend();

} // namespace mlx::core::cpu
