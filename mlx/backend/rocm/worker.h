// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace mlx::core::rocm {

// Forward declarations
class HipEvent;

// Run tasks in worker thread, synchronized with HIP stream.
class Worker {
 public:
  Worker();
  ~Worker();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // Add a pending |task| that will run when consumed or committed.
  void add_task(std::function<void()> task);

  // Inform worker thread to run current batches after kernels in |stream|
  // finish running.
  void commit(hipStream_t stream);

 private:
  static void signal(void*);

  void thread_fn();
  std::mutex mtx_;
  std::condition_variable cond_;

  uint64_t committed_batch_{0};
  uint64_t signaled_batch_{0};

  bool stop_{false};

  // Tasks are put in |pending_tasks_| first, and then moved to
  // |worker_tasks_| when end_batch() is called.
  using Tasks = std::vector<std::function<void()>>;
  Tasks pending_tasks_;
  std::map<uint64_t, Tasks> worker_tasks_;
  std::thread worker_;
};

} // namespace mlx::core::rocm
