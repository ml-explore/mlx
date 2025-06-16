// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>
#include <functional>
#include <future>
#include <queue>
#include <thread>

namespace mlx::core::rocm {

using HipStream = hipStream_t;

class Worker {
 public:
  Worker();
  ~Worker();

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  void enqueue(std::function<void()> task);
  void commit();
  void join();

 private:
  void worker_loop();

  std::thread worker_thread_;
  std::queue<std::function<void()>> tasks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool stop_{false};
  bool committed_{false};
};

} // namespace mlx::core::rocm