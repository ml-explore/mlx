// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/worker.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"

namespace mlx::core::cu {

Worker::Worker()
    : signal_stream_(device(mlx::core::Device::gpu)),
      worker_(&Worker::thread_fn, this) {}

Worker::~Worker() {
  {
    std::lock_guard lock(worker_mutex_);
    stop_ = true;
  }
  worker_event_.signal(batch_ + 1);
  worker_.join();
}

void Worker::add_task(std::function<void()> task) {
  pending_tasks_.push_back(std::move(task));
}

void Worker::consume_in_this_thread() {
  for (auto& task : pending_tasks_) {
    task();
  }
  pending_tasks_.clear();
}

void Worker::end_batch() {
  batch_++;
  {
    std::lock_guard lock(worker_mutex_);
    worker_tasks_[batch_] = std::move(pending_tasks_);
  }
  uncommited_batches_++;
}

void Worker::commit() {
  if (uncommited_batches_ == 0) {
    return;
  }
  uncommited_batches_ = 0;
  worker_event_.signal(batch_);
}

void Worker::commit(cudaStream_t stream) {
  if (uncommited_batches_ == 0) {
    return;
  }
  uncommited_batches_ = 0;
  // Signal the |worker_event_| in |signal_stream_| after the kernels in
  // |stream_| finish running.
  signal_event_.record(stream);
  signal_event_.wait(signal_stream_);
  worker_event_.signal(signal_stream_, batch_);
}

void Worker::thread_fn() {
  // The worker thread is safe to free buffers.
  allocator().register_this_thread();

  while (!stop_) {
    uint64_t batch = worker_event_.value();
    Tasks tasks;
    {
      std::lock_guard lock(worker_mutex_);
      // Move tasks in signaled batches.
      auto end = worker_tasks_.upper_bound(batch);
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
    for (auto& task : tasks) {
      task();
    }
    worker_event_.wait(batch + 1);
  }
}

} // namespace mlx::core::cu
