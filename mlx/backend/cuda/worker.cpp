// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/worker.h"
#include "mlx/backend/cuda/device.h"

#include <thread>

namespace mlx::core::cu {

Worker::Worker(Device& d)
    : signal_stream_(d),
      signal_event_(d, cudaEventDisableTiming | cudaEventBlockingSync),
      state_(std::make_shared<State>()) {
  // Detach the thread and let it free the state after finishing tasks.
  // This is to avoid deadlock when joining threads on exit on Windows:
  // https://developercommunity.visualstudio.com/t/1654756
  std::thread(&State::thread_fn, state_).detach();
}

Worker::~Worker() {
  {
    std::lock_guard lock(state_->mtx);
    state_->stop = true;
  }
  state_->cond.notify_one();
}

void Worker::add_task(std::function<void()> task) {
  pending_tasks_.push_back(std::move(task));
}

void Worker::commit(cudaStream_t stream) {
  // Move pending tasks into tasks
  if (pending_tasks_.empty()) {
    return;
  }
  {
    std::lock_guard lock(state_->mtx);
    // Move pending tasks into ready tasks
    state_->worker_tasks[++committed_batch_] = std::move(pending_tasks_);
  }
  signal_event_.record(stream);
  signal_event_.wait(signal_stream_);
  CHECK_CUDA_ERROR(
      cudaLaunchHostFunc(signal_stream_, State::signal, state_.get()));
}

// static
void Worker::State::signal(void* data) {
  auto state = static_cast<State*>(data);
  {
    std::lock_guard lock(state->mtx);
    state->signaled_batch++;
  }
  state->cond.notify_one();
}

void Worker::State::thread_fn() {
  uint64_t current_batch = 0;
  while (true) {
    Tasks tasks;
    {
      std::unique_lock<std::mutex> lk(mtx);
      cond.wait(lk, [this, current_batch] {
        return this->signaled_batch > current_batch || this->stop;
      });
      if (stop) {
        return;
      }
      current_batch = signaled_batch;
      auto end = worker_tasks.upper_bound(current_batch);
      for (auto it = worker_tasks.begin(); it != end; ++it) {
        if (tasks.empty()) {
          tasks = std::move(it->second);
        } else {
          std::move(
              it->second.begin(), it->second.end(), std::back_inserter(tasks));
        }
      }
      worker_tasks.erase(worker_tasks.begin(), end);
    }
    // Make sure tasks are cleared before the next wait
    for (int i = 0; i < tasks.size(); ++i) {
      auto task = std::move(tasks[i]);
      task();
    }
  }
}

} // namespace mlx::core::cu
