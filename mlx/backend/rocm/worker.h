// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

namespace mlx::core::rocm {

// Simple worker for async task execution synchronized with HIP streams.
class Worker {
 public:
  Worker();
  ~Worker();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  // Add a task to be executed
  void add_task(std::function<void()> task);

  // Run pending tasks immediately in current thread.
  void consume_in_this_thread();

  // Commit tasks to be run after stream completion
  void commit(hipStream_t stream);

  // Simple commit without stream dependency
  void commit();

 private:
  void worker_loop();

  std::thread worker_thread_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_{false};
};

} // namespace mlx::core::rocm