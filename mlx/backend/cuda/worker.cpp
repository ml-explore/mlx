// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/worker.h"
#include "mlx/backend/cuda/device.h"

namespace mlx::core::cu {

Worker::Worker()
    : signal_stream_(device(mlx::core::Device::gpu)),
      signal_event_(cudaEventDisableTiming | cudaEventBlockingSync),
      worker_(&Worker::thread_fn, this) {}

Worker::~Worker() {
  {
    std::lock_guard lock(mtx_);
    stop_ = true;
  }
  cond_.notify_one();
  worker_.join();
}

void Worker::add_task(std::function<void()> task) {
  pending_tasks_.push_back(std::move(task));
}

void Worker::signal(void* data) {
  auto w = static_cast<Worker*>(data);
  {
    std::lock_guard lock(w->mtx_);
    w->signaled_batch_++;
  }
  w->cond_.notify_one();
}

void Worker::commit(cudaStream_t stream) {
  // Move pending tasks into tasks
  if (pending_tasks_.empty()) {
    return;
  }
  {
    std::lock_guard lock(mtx_);
    // Move pending tasks into ready tasks
    worker_tasks_[++committed_batch_] = std::move(pending_tasks_);
  }
  signal_event_.record(stream);
  signal_event_.wait(signal_stream_);
  cudaLaunchHostFunc(signal_stream_, signal, this);
}

void Worker::thread_fn() {
  while (!stop_) {
    uint64_t current_batch = 0;
    Tasks tasks;
    {
      std::unique_lock<std::mutex> lk(mtx_);
      cond_.wait(lk, [this, &current_batch] {
        return this->signaled_batch_ > current_batch || this->stop_;
      });
      current_batch = signaled_batch_;
      auto end = worker_tasks_.upper_bound(current_batch);
      for (auto it = worker_tasks_.begin(); it != end; ++it) {
        if (tasks.empty()) {
          tasks = std::move(it->second);
        } else {
          std::move(
              it->second.begin(), it->second.end(), std::back_inserter(tasks));
        }
      }
      worker_tasks_.erase(worker_tasks_.begin(), end);
    }
    // Make sure tasks are cleared before the next wait
    for (int i = 0; i < tasks.size(); ++i) {
      auto task = std::move(tasks[i]);
      task();
    }
  }
}

} // namespace mlx::core::cu
