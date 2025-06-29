// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/worker.h"
#include "mlx/backend/rocm/utils.h"

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

void Worker::add_task(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    tasks_.push(task);
  }
  cv_.notify_one();
}

void Worker::consume_in_this_thread() {
  std::queue<std::function<void()>> local_tasks;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    local_tasks.swap(tasks_);
  }

  while (!local_tasks.empty()) {
    auto task = local_tasks.front();
    local_tasks.pop();
    task();
  }
}

void Worker::commit(hipStream_t stream) {
  // Synchronize with stream and then process tasks
  CHECK_HIP_ERROR(hipStreamSynchronize(stream));
  consume_in_this_thread();
}

void Worker::commit() {
  cv_.notify_all();
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