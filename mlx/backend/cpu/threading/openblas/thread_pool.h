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
 * Persistent thread pool for Linux/Windows with OpenBLAS coordination.
 *
 * This pool manages its own std::thread workers and pins OpenBLAS to
 * single-threaded mode at startup to avoid over-subscription (e.g. our 32
 * threads + OpenBLAS's 32 threads fighting for 32 cores). BLAS performance
 * is unaffected because matmul (cblas.cpp) calls cblas_sgemm from within
 * parallel_for -- each worker invokes single-threaded BLAS on its row slice,
 * achieving full core utilization without internal BLAS threading.
 *
 * Worker wakeup uses per-worker cache-line-aligned generation flags to avoid
 * mutex serialization on the hot path. Workers spin briefly on their private
 * flag (no cross-core cache contention), then fall back to cv_.wait for long
 * idle periods. This reduces wakeup latency from ~10-50us (mutex + OS wakeup)
 * to ~100ns (L3 cache hit on flag change).
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

  // Per-worker wake flag on its own cache line to avoid false sharing.
  // Workers spin on their private flag -- no cross-core cache contention.
  struct alignas(64) WorkerSlot {
    std::atomic<uint64_t> wake_gen{0};
  };
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
  std::atomic<int> sleeping_count_{0}; // workers currently in cv_.wait
  bool stop_ = false;

  WorkerSlot worker_slots_[MAX_WORKERS];

  // Session mode: when workers are spinning between frequent parallel_for
  // calls (e.g., 130 ops/token during generation), avoid per-op mutex
  // acquisition, N wake_gen writes, and started_ contention. Workers spin
  // on a shared session_gen_ counter instead of per-worker wake_gen flags.
  // Dispatch cost drops from ~5us (mutex + N writes) to ~0.1us (one atomic).
  alignas(64) std::atomic<uint64_t> session_gen_{0};
  alignas(64) std::atomic<int> session_count_{0}; // workers in session loop
  std::atomic<bool> in_session_{false};

  void session_worker_loop(int worker_id);
};

} // namespace mlx::core::cpu
