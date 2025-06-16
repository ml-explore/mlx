// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/worker.h"

namespace mlx::core::rocm {

Worker::Worker() : worker_thread_(&Worker::worker_loop, this) {}

Worker::~Worker() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  if (worker_thread_.joinable()) {
    worker_thread_.join();
  }
}

void Worker::enqueue(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push(task);
  }
  cv_.notify_one();
}

void Worker::commit() {
  std::lock_guard<std::mutex> lock(mutex_);
  committed_ = true;
}

void Worker::join() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return tasks_.empty() && committed_; });
}

void Worker::worker_loop() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

      if (stop_) {
        break;
      }

      if (!tasks_.empty()) {
        task = tasks_.front();
        tasks_.pop();
      }
    }

    if (task) {
      task();
    }
  }
}

} // namespace mlx::core::rocm