// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/backend/cpu/threading/base.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace mlx::core::cpu {

/**
 * Persistent thread pool for Linux/Windows with optional OpenBLAS coordination.
 *
 * This pool manages its own std::thread workers and pins OpenBLAS to
 * single-threaded mode at startup when OpenBLAS symbols are available. This
 * avoids over-subscription between the pool and OpenBLAS. Batched or larger
 * GEMMs can still run in parallel across pool workers via cblas.cpp.
 *
 * Worker wakeup uses per-worker cache-line-aligned generation flags. Workers
 * spin briefly on their private flag (no cross-core cache contention), then
 * fall back to cv_.wait for long idle periods.
 */
class CPUThreadPool : public ThreadPoolBackend {
 public:
  CPUThreadPool();
  ~CPUThreadPool() override;

  void parallel_for(int n_threads, std::function<void(int tid, int nth)> f)
      override;
  int max_threads() const override;

 private:
  void worker_loop(int worker_id);
  // Runs one slot of the current task, capturing the first exception so it
  // can be rethrown on the dispatching thread. Always counts the slot as
  // done -- parallel_for must keep the task alive until every claimed slot
  // has finished, even when one throws.
  void run_task_slot(int slot, int nth);

  // Per-worker wake flag on its own cache line to avoid false sharing.
  // Workers spin on their private flag -- no cross-core cache contention.
  struct alignas(64) WorkerSlot {
    std::atomic<uint64_t> wake_gen{0};
  };
  // Maximum background workers. The main caller also participates, so
  // max_threads_ can be at most MAX_WORKERS + 1.
  static constexpr int MAX_WORKERS = 128;

  std::vector<std::thread> workers_;
  int max_threads_;

  // Serializes concurrent parallel_for calls from different CPU streams.
  // MLX's stream_generate uses a separate generation_stream, so two
  // StreamThreads may call parallel_for simultaneously. Since all task
  // state (task_ptr_, started_, done_, etc.) is shared, concurrent calls
  // must be serialized. The GCD backend (macOS) handles this implicitly
  // via dispatch_apply; we need an explicit mutex.
  std::mutex dispatch_mtx_;

  std::mutex mtx_; // worker sleep/wake coordination
  std::condition_variable cv_;

  const std::function<void(int, int)>* task_ptr_ = nullptr;
  std::atomic<int> task_n_threads_{0};
  std::atomic<int> started_{0};
  std::atomic<int> done_{0};
  std::atomic<int> ready_{0};
  std::atomic<uint64_t> gen_{0};
  std::atomic<uint64_t> task_gen_{0};
  std::atomic<int> sleeping_count_{0}; // workers currently in cv_.wait
  bool stop_ = false;

  // First exception thrown by any slot of the current task; rethrown by
  // parallel_for after all slots complete. One task runs at a time
  // (dispatch_mtx_), so a single slot suffices.
  std::mutex exception_mtx_;
  std::exception_ptr task_exception_;

  WorkerSlot worker_slots_[MAX_WORKERS];
};

} // namespace mlx::core::cpu
