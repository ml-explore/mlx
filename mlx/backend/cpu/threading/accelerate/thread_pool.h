// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/backend/cpu/threading/base.h"

namespace mlx::core::cpu {

/**
 * macOS thread pool implementation using Grand Central Dispatch (GCD).
 *
 * GCD is a system-wide thread pool managed by the macOS kernel.
 * Benefits:
 * - No thread management overhead in userspace
 * - Automatic thread count optimization
 * - Integration with system scheduler
 * - No conflicts with Accelerate framework
 */
class AccelerateThreadPool : public ThreadPoolBackend {
 public:
  AccelerateThreadPool();
  ~AccelerateThreadPool() override = default;

  void parallel_for(int n_threads, std::function<void(int tid, int nth)> f)
      override;

  int max_threads() const override;

 private:
  int max_threads_;
};

} // namespace mlx::core::cpu
