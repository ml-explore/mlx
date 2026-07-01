// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/worker.h"
#include "mlx/backend/rocm/utils.h"

#include <atomic>

namespace mlx::core::rocm {

// Defined in device.cpp. True during a full decode-step stream capture.
extern std::atomic<bool> g_decode_capturing;

Worker::Worker(int device) : device_(device), worker_(&Worker::thread_fn, this) {}

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

void Worker::commit(hipStream_t stream) {
  // Move pending tasks into tasks
  if (pending_tasks_.empty()) {
    return;
  }
  {
    std::lock_guard lock(mtx_);
    // Move pending tasks into ready tasks
    worker_tasks_[++committed_batch_] = std::move(pending_tasks_);
  }
  // During a full decode-step capture, hipLaunchHostFunc can't be recorded into
  // the captured graph (and would re-fire on every replay). Signal completion
  // inline so batch accounting stays consistent and tasks (frees) still run.
  // The captured kernels haven't executed yet, but the deterministic decode
  // arena makes freeing/reusing those buffers safe across the record token.
  if (g_decode_capturing.load(std::memory_order_relaxed)) {
    signal(this);
    return;
  }
  // Use hipLaunchHostFunc to signal when stream operations complete
  (void)hipLaunchHostFunc(stream, signal, this);
}

void Worker::thread_fn() {
  // Bind this thread to the encoder's device before running any task. Completion
  // handlers free temporaries / return buffers to the pool and may issue HIP
  // calls; they must hit the same device the stream lives on, not the default
  // device 0. Without this the discrete-GPU queue wedges on a multi-GPU host.
  (void)hipSetDevice(device_);
  uint64_t current_batch = 0;
  while (!stop_) {
    Tasks tasks;
    {
      std::unique_lock<std::mutex> lk(mtx_);
      cond_.wait(lk, [this, current_batch] {
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
    for (size_t i = 0; i < tasks.size(); ++i) {
      auto task = std::move(tasks[i]);
      task();
    }
  }
}

} // namespace mlx::core::rocm
